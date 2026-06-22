"""
Custom autograd Function that fuses weight gradient computation with optimizer
update during backward. Drop-in replacement for nn.Linear's autograd behavior.

Forward:  Y = X @ W.T  (standard matmul)
Backward:
  - grad_input = grad_output @ W          (cuBLAS — returned for chain rule)
  - grad_weight = grad_output.T @ X       (fused with optimizer via Triton)
  - Weight updated in-place; grad_weight is NEVER returned or stored

Level 2: If an L2CacheManager is provided, activation tensors are pinned in
L2 before the fused kernel launch for reduced read latency.
"""

import torch
import torch.nn.functional as F

from fused_grad_optimizer.kernel import (
    fused_grad_sgd, fused_grad_adamw, optimizer_only_adamw,
    fused_grad_adamw_int8state, optimizer_only_adamw_int8state,
)
from fused_grad_optimizer.kernel_fp8_state import fused_grad_adamw_fp8state
from fused_grad_optimizer.state import FusedOptimizerState, OptimizerConfig


class FusedLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, optimizer_state, optimizer_config,
                is_accumulating, l2_manager, use_fused_backward=True):
        # Use F.linear (single fused addmm) instead of `input @ weight.t() + bias`
        # which would do two separate kernels. Mathematically identical, but ~28 ms
        # faster per fwd at Llama-3.1-8B BS=1 SEQ=2048 (52 ms vs 80 ms).
        output = F.linear(input, weight, bias)

        ctx.save_for_backward(input, weight, bias)
        ctx.optimizer_state = optimizer_state
        ctx.optimizer_config = optimizer_config
        ctx.is_accumulating = is_accumulating
        ctx.l2_manager = l2_manager
        ctx.use_fused_backward = use_fused_backward
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        state = ctx.optimizer_state
        config = ctx.optimizer_config

        # grad_input MUST be computed against the pre-update weight — the
        # fused path below mutates `weight` in place, so any reordering that
        # defers grad_input past _apply_fused / _apply_precomputed silently
        # uses the post-step weight (verified to break the chain rule by
        # benchmarks/b200/_fused_correctness_check.py).
        grad_input = grad_output @ weight

        # Reshape to 2D for kernel
        input_2d = input.reshape(-1, input.shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        if config is None:
            # No fused optimizer: fall back to standard autograd
            grad_weight = grad_output_2d.t() @ input_2d
        elif ctx.is_accumulating:
            # Gradient accumulation micro-step: buffer gradient, defer update
            state.accumulate_grad(grad_output_2d.t() @ input_2d)
            grad_weight = None
        else:
            # Final step: apply optimizer
            pending = state.pop_accumulated_grad()

            if pending is not None:
                # Had accumulated gradients — materialize + sum + apply
                grad_weight_total = (grad_output_2d.t() @ input_2d).float()
                grad_weight_total.add_(pending)
                _apply_precomputed(grad_weight_total, weight, state, config)
            elif not ctx.use_fused_backward:
                # cuBLAS fallback: materialize grad_W, apply optimizer separately.
                # Faster for very large V (e.g. lm_head) where read amplification
                # makes the fused kernel slower than cuBLAS + separate optimizer.
                grad_weight = (grad_output_2d.t() @ input_2d).float()
                _apply_precomputed(grad_weight, weight, state, config)
            else:
                # No accumulation — use the fused kernel (no grad_W allocation)
                _apply_fused(grad_output_2d, input_2d, weight, state, config,
                             ctx.l2_manager)

            grad_weight = None

        grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


def _apply_fused(grad_output_2d, input_2d, weight, state, config, l2_manager):
    """Tile-wise fused gradient + optimizer. grad_W never exists in HBM."""
    state.ensure_buffers()
    state.increment_step()
    opt = config.optimizer_type

    # `.contiguous()` is a no-op when the tensor is already contiguous (which
    # is the common case after `reshape(-1, hidden)` on a contiguous source),
    # so guard explicitly to skip the rare copy and make the intent clear.
    go = grad_output_2d if grad_output_2d.is_contiguous() else grad_output_2d.contiguous()
    inp = input_2d if input_2d.is_contiguous() else input_2d.contiguous()

    # Level 2: pin activation tensors in L2 if manager available
    if l2_manager is not None:
        l2_manager.pin(inp, go)

    try:
        if opt == "sgd":
            fused_grad_sgd(go, inp, weight,
                           lr=config.lr, weight_decay=config.weight_decay)
        elif opt == "adamw":
            if state.state_mode == "int8":
                fused_grad_adamw_int8state(
                    go, inp, weight, state.m_q, state.v_q,
                    state.m_scale, state.v_scale,
                    lr=config.lr, beta1=config.beta1,
                    beta2=config.beta2, eps=config.eps,
                    weight_decay=config.weight_decay,
                    step=state.step, qblock=state.qblock_size)
            elif state.state_mode == "fp8":
                # Delayed per-tensor scaling: update this-step's scale from
                # last-step's absmax (populated by the previous kernel via
                # atomic_max), then reset the accumulator.
                state.pre_step_fp8()
                fused_grad_adamw_fp8state(
                    go, inp, weight, state,
                    lr=config.lr, beta1=config.beta1,
                    beta2=config.beta2, eps=config.eps,
                    weight_decay=config.weight_decay,
                    step=state.step)
            else:
                fused_grad_adamw(go, inp, weight, state.m, state.v,
                                 lr=config.lr, beta1=config.beta1,
                                 beta2=config.beta2, eps=config.eps,
                                 weight_decay=config.weight_decay,
                                 step=state.step)
        else:
            raise ValueError(f"Unknown optimizer: {opt}")
    finally:
        if l2_manager is not None:
            l2_manager.unpin()


def _apply_precomputed(grad_weight, weight, state, config):
    """Apply optimizer to a pre-computed gradient (gradient accumulation path)."""
    state.ensure_buffers()
    state.increment_step()
    opt = config.optimizer_type

    if opt == "sgd":
        # Simple SGD on pre-computed gradient
        weight.data.mul_(1.0 - config.lr * config.weight_decay)
        weight.data.add_(grad_weight.to(weight.dtype), alpha=-config.lr)
    elif opt == "adamw":
        if state.quantize_state:
            optimizer_only_adamw_int8state(
                grad_weight, weight, state.m_q, state.v_q,
                state.m_scale, state.v_scale,
                lr=config.lr, beta1=config.beta1,
                beta2=config.beta2, eps=config.eps,
                weight_decay=config.weight_decay,
                step=state.step, qblock=state.qblock_size,
            )
        else:
            optimizer_only_adamw(
                grad_weight, weight, state.m, state.v,
                lr=config.lr,
                beta1=config.beta1,
                beta2=config.beta2,
                eps=config.eps,
                weight_decay=config.weight_decay,
                step=state.step,
            )
    else:
        raise ValueError(f"Unknown optimizer: {opt}")
