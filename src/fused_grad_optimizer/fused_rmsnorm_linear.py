"""
v6 prototype — FusedRMSNormLinear.

Idea: under AC, avoid materialising `norm_output` as a saved tensor.
Fold RMSNorm into a Linear's custom autograd Function: forward saves
(hidden_states, gamma) instead of the norm's output, and backward
recomputes `norm_output` just before calling the existing tile-wise
fused grad+AdamW kernel.

Theoretical memory win:
  - Removes the saved `norm_output` tensor from autograd's graph
    (~32 MB per norm at SEQ=4096, H=4096 bf16), across 2 norms × 32 layers
    ≈ 2 GB window. Paired with layer-level AC the window is already
    short, so expected peak savings is single-digit GB.

Compute overhead:
  - Each of q/k/v_proj (and gate/up_proj) re-applies RMSNorm once in
    forward and once in backward. With a shared-norm layout that's 3x
    (attn) or 2x (MLP) the RMSNorm cost — trivial compared to the matmul.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fused_grad_optimizer.autograd import _apply_fused, _apply_precomputed
from fused_grad_optimizer.module import FusedLinear
from fused_grad_optimizer.state import FusedOptimizerState, OptimizerConfig


def _rmsnorm_forward(hidden_states: torch.Tensor,
                     gamma: torch.Tensor,
                     eps: float) -> torch.Tensor:
    in_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (gamma.to(torch.float32) * x).to(in_dtype)


def _rmsnorm_backward(hidden_states: torch.Tensor,
                      gamma: torch.Tensor,
                      grad_norm_out: torch.Tensor,
                      eps: float):
    in_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    g = grad_norm_out.to(torch.float32)
    gm = gamma.to(torch.float32)
    H = x.shape[-1]

    variance = x.pow(2).mean(-1, keepdim=True)
    rrms = torch.rsqrt(variance + eps)
    x_hat = x * rrms

    sum_term = (g * gm * x).sum(-1, keepdim=True)
    grad_x = g * gm * rrms - x * sum_term * (rrms ** 3) / H

    grad_gamma = (g * x_hat).reshape(-1, H).sum(dim=0)
    return grad_x.to(in_dtype), grad_gamma.to(gamma.dtype)


class FusedRMSNormLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden_states, gamma, eps, weight, bias,
                optimizer_state, optimizer_config,
                is_accumulating, l2_manager, use_fused_backward=True):
        norm_output = _rmsnorm_forward(hidden_states, gamma, eps)
        output = norm_output @ weight.t()
        if bias is not None:
            output = output + bias

        ctx.has_bias = bias is not None
        if ctx.has_bias:
            ctx.save_for_backward(hidden_states, gamma, weight, bias)
        else:
            ctx.save_for_backward(hidden_states, gamma, weight)
        ctx.eps = eps
        ctx.optimizer_state = optimizer_state
        ctx.optimizer_config = optimizer_config
        ctx.is_accumulating = is_accumulating
        ctx.l2_manager = l2_manager
        ctx.use_fused_backward = use_fused_backward
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            hidden_states, gamma, weight, bias = ctx.saved_tensors
        else:
            hidden_states, gamma, weight = ctx.saved_tensors
            bias = None
        state = ctx.optimizer_state
        config = ctx.optimizer_config

        # Recompute norm_output — transient, never saved across backward.
        norm_output = _rmsnorm_forward(hidden_states, gamma, ctx.eps)

        # Chain rule into RMSNorm
        grad_norm_output = grad_output @ weight

        norm_2d = norm_output.reshape(-1, norm_output.shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        if config is None:
            grad_weight = grad_output_2d.t() @ norm_2d
        elif ctx.is_accumulating:
            state.accumulate_grad(grad_output_2d.t() @ norm_2d)
            grad_weight = None
        else:
            pending = state.pop_accumulated_grad()
            if pending is not None:
                grad_w_tot = (grad_output_2d.t() @ norm_2d).float()
                grad_w_tot.add_(pending)
                _apply_precomputed(grad_w_tot, weight, state, config)
            elif not ctx.use_fused_backward:
                grad_w = (grad_output_2d.t() @ norm_2d).float()
                _apply_precomputed(grad_w, weight, state, config)
            else:
                _apply_fused(grad_output_2d, norm_2d, weight, state, config,
                             ctx.l2_manager)
            grad_weight = None

        del norm_output, norm_2d

        grad_bias = (grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
                     if bias is not None else None)

        grad_hidden_states, grad_gamma = _rmsnorm_backward(
            hidden_states, gamma, grad_norm_output, ctx.eps)

        return (grad_hidden_states, grad_gamma, None,
                grad_weight, grad_bias,
                None, None, None, None, None)


class FusedRMSNormLinear(FusedLinear):
    """
    FusedLinear variant that additionally folds a preceding RMSNorm.
    The gamma Parameter is owned by the original RMSNorm module (still
    registered in model.parameters()); this class stores it by reference.

    Picked up by FusedOptimizerManager via FusedLinear isinstance check.
    """

    def __init__(self, linear: nn.Linear, gamma: nn.Parameter, eps: float,
                 optimizer_type: str = "adamw", state_mode: str = "bf16"):
        # Initialise parent with the linear's shape; copy weights afterward.
        super().__init__(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            optimizer_type=optimizer_type,
            state_mode=state_mode,
        )
        self.weight.data = linear.weight.data
        if linear.bias is not None and self.bias is not None:
            self.bias.data = linear.bias.data
        self._gamma = gamma
        self._eps = eps

    def forward(self, hidden_states):
        if not self.training:
            norm_out = _rmsnorm_forward(hidden_states, self._gamma, self._eps)
            return F.linear(norm_out, self.weight, self.bias)

        self._ensure_state()
        return FusedRMSNormLinearFunction.apply(
            hidden_states, self._gamma, self._eps,
            self.weight, self.bias,
            self._state, self._config, self._is_accumulating,
            self._l2_manager, self.use_fused_backward,
        )

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, shared_gamma=True, "
                f"eps={self._eps}, state_mode={self.state_mode}")


def fuse_rmsnorm_linear_pairs(model):
    """
    Walk a Llama model and replace (RMSNorm + child Linear) pairs with
    FusedRMSNormLinear. The original RMSNorm module's forward becomes an
    identity so the downstream Linear receives the pre-norm tensor.

    Targets per decoder layer:
      - input_layernorm           feeds q_proj, k_proj, v_proj
      - post_attention_layernorm  feeds gate_proj, up_proj

    Returns a callable that restores the original modules.
    """
    undo = []

    def _identity_forward(self, x):
        return x

    for layer in model.model.layers:
        norm1 = layer.input_layernorm
        norm2 = layer.post_attention_layernorm
        eps1 = getattr(norm1, "variance_epsilon",
                       getattr(norm1, "eps", 1e-6))
        eps2 = getattr(norm2, "variance_epsilon",
                       getattr(norm2, "eps", 1e-6))

        orig_fwd1 = norm1.forward
        orig_fwd2 = norm2.forward
        norm1.forward = _identity_forward.__get__(norm1, type(norm1))
        norm2.forward = _identity_forward.__get__(norm2, type(norm2))
        undo.append(("norm_fwd", norm1, orig_fwd1))
        undo.append(("norm_fwd", norm2, orig_fwd2))

        attn = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            orig = getattr(attn, name)
            if isinstance(orig, nn.Linear):
                fused = FusedRMSNormLinear(orig, norm1.weight, eps1)
                fused = fused.to(orig.weight.device).to(orig.weight.dtype)
                setattr(attn, name, fused)
                undo.append(("submod", attn, name, orig))

        mlp = layer.mlp
        for name in ("gate_proj", "up_proj"):
            orig = getattr(mlp, name)
            if isinstance(orig, nn.Linear):
                fused = FusedRMSNormLinear(orig, norm2.weight, eps2)
                fused = fused.to(orig.weight.device).to(orig.weight.dtype)
                setattr(mlp, name, fused)
                undo.append(("submod", mlp, name, orig))

    def _restore():
        for entry in reversed(undo):
            kind = entry[0]
            if kind == "norm_fwd":
                _, mod, fwd = entry
                mod.forward = fwd
            elif kind == "submod":
                _, parent, name, orig = entry
                setattr(parent, name, orig)

    return _restore


__all__ = [
    "FusedRMSNormLinear",
    "FusedRMSNormLinearFunction",
    "fuse_rmsnorm_linear_pairs",
]
