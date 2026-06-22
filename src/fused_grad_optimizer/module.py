"""
FusedLinear: drop-in nn.Linear replacement with fused grad+optimizer.
FusedOptimizerManager: coordinates all FusedLinear modules in a model.

Usage:
    # Replace a layer
    model.lm_head = FusedLinear.from_linear(model.lm_head, optimizer_type="adamw")

    # Coordinate with training loop
    manager = FusedOptimizerManager(model)
    regular_optimizer = torch.optim.AdamW(manager.get_non_fused_params(), lr=1e-4)

    for step, batch in enumerate(dataloader):
        manager.pre_step(lr=get_lr(step), step=step + 1)
        loss = model(**batch).loss
        loss.backward()                     # fused layers update weights here
        regular_optimizer.step()            # update non-fused layers
        regular_optimizer.zero_grad()

Level 2 (L2 cache pinning):
    manager = FusedOptimizerManager(model, use_l2_pinning=True)
    # L2 pinning is transparent — activations are pinned before each fused kernel
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fused_grad_optimizer.autograd import FusedLinearFunction
from fused_grad_optimizer.state import FusedOptimizerState, OptimizerConfig


class FusedLinear(nn.Module):
    """
    Drop-in nn.Linear that fuses weight gradient + optimizer in backward.

    The weight gradient is computed tile-by-tile in Triton registers and
    immediately consumed by the optimizer. The full (out_features x in_features)
    gradient tensor is never allocated.
    """

    def __init__(self, in_features, out_features, bias=False,
                 optimizer_type="adamw", quantize_state=False,
                 state_mode=None, **optimizer_kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        self.optimizer_type = optimizer_type
        self.quantize_state = quantize_state
        # state_mode ∈ {"bf16", "int8", "fp8"}. Falls back to int8 if
        # quantize_state=True was used (legacy argument).
        self.state_mode = state_mode if state_mode is not None else (
            "int8" if quantize_state else "bf16")
        self.optimizer_kwargs = optimizer_kwargs
        self._state: FusedOptimizerState | None = None
        self._config: OptimizerConfig | None = None
        self._is_accumulating = False
        self._l2_manager = None
        # When False, use cuBLAS matmul + separate optimizer (faster for very large V)
        self.use_fused_backward = True

    @classmethod
    def from_linear(cls, linear: nn.Linear, optimizer_type="adamw",
                    quantize_state=False, state_mode=None, **optimizer_kwargs):
        """Create from an existing nn.Linear, sharing weight data."""
        fused = cls(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            optimizer_type=optimizer_type,
            quantize_state=quantize_state,
            state_mode=state_mode,
            **optimizer_kwargs,
        )
        fused.weight.data = linear.weight.data
        if linear.bias is not None and fused.bias is not None:
            fused.bias.data = linear.bias.data
        return fused

    def _ensure_state(self):
        if self._state is None:
            self._state = FusedOptimizerState(
                self.weight, self.optimizer_type,
                quantize_state=self.quantize_state,
                state_mode=self.state_mode)

    def update_optimizer_config(self, config=None, **kwargs):
        """Call before each forward pass to set lr, step, etc.

        If a pre-built config is passed, use it directly (shared across modules).
        Otherwise fall back to creating one from kwargs.
        """
        if config is not None:
            self._config = config
        else:
            merged = {**self.optimizer_kwargs, **kwargs}
            self._config = OptimizerConfig(optimizer_type=self.optimizer_type, **merged)

    def set_accumulating(self, is_accumulating: bool):
        """True during gradient accumulation micro-steps, False on final step."""
        self._is_accumulating = is_accumulating

    def set_l2_manager(self, l2_manager):
        """Attach L2 cache manager for Level 2 optimization."""
        self._l2_manager = l2_manager

    def forward(self, x):
        if not self.training:
            return F.linear(x, self.weight, self.bias)

        self._ensure_state()
        # If the C++ dispatch path is enabled (env: FUSED_LINEAR_CPP=1), call
        # into the C++ autograd Function. This eliminates the per-call Python
        # overhead of torch.autograd.Function.apply (saves ~10-20 ms on a
        # Llama-3.1-8B fwd with its ~225 nn.Linear calls).
        import os as _os
        if _os.environ.get("FUSED_LINEAR_CPP") == "1":
            from fused_grad_optimizer.fused_linear_cpp_bridge import fused_linear_apply_cpp
            return fused_linear_apply_cpp(
                x, self.weight, self.bias,
                self._state, self._config, self._is_accumulating,
            )
        return FusedLinearFunction.apply(
            x, self.weight, self.bias,
            self._state, self._config, self._is_accumulating,
            self._l2_manager, self.use_fused_backward,
        )

    def get_optimizer_state_dict(self):
        if self._state is None:
            return None
        return self._state.state_dict()

    def load_optimizer_state_dict(self, d):
        self._ensure_state()
        self._state.load_state_dict(d)

    def extra_repr(self):
        s = (f"in_features={self.in_features}, out_features={self.out_features}, "
             f"bias={self.bias is not None}, optimizer={self.optimizer_type}")
        if self.state_mode != "bf16":
            s += f", state_mode={self.state_mode}"
        return s


class FusedOptimizerManager:
    """
    Coordinates all FusedLinear modules in a model.

    Handles:
    - Updating optimizer config (lr, step) before each forward
    - Excluding fused parameters from the main optimizer
    - Gradient accumulation state
    - L2 cache pinning (Level 2)
    - Checkpoint save/load
    """

    def __init__(self, model: nn.Module, gradient_accumulation_steps: int = 1,
                 use_l2_pinning: bool = False, quantize_state: bool | None = None):
        self.model = model
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._global_step = 0
        self._micro_step = 0

        self._fused_modules: list[FusedLinear] = []
        self._fused_module_names: dict[int, str] = {}
        self._fused_param_ids: set[int] = set()

        # Level 2: L2 cache pinning
        self._l2_manager = None
        if use_l2_pinning:
            from fused_grad_optimizer.l2_cache import L2CacheManager
            self._l2_manager = L2CacheManager()

        for name, module in model.named_modules():
            if isinstance(module, FusedLinear):
                # Override quantize_state if specified at manager level
                if quantize_state is not None:
                    module.quantize_state = quantize_state
                self._fused_modules.append(module)
                self._fused_module_names[id(module)] = name
                self._fused_param_ids.add(id(module.weight))
                # bias is NOT added — the fused kernel only updates weights.
                # bias.grad flows through autograd normally and must be
                # stepped by the regular optimizer via get_non_fused_params().
                if self._l2_manager is not None:
                    module.set_l2_manager(self._l2_manager)
                # cuBLAS fallback: available via module.use_fused_backward = False
                # for layers where cuBLAS + separate optimizer beats the fused
                # kernel. Currently disabled by default — the fused kernel is
                # competitive on RTX PRO 6000 and the memory cost of materializing
                # grad_W for large layers (e.g. lm_head: 1GB) offsets the savings.

    def get_non_fused_params(self) -> list[nn.Parameter]:
        """Parameters NOT managed by fused kernels — pass these to your optimizer."""
        return [
            p for p in self.model.parameters()
            if id(p) not in self._fused_param_ids and p.requires_grad
        ]

    def pre_step(self, lr: float | None = None, step: int | None = None,
                 is_accumulating: bool | None = None, **kwargs):
        """Call before each forward pass."""
        self._micro_step += 1

        if is_accumulating is None:
            is_accumulating = (self._micro_step % self.gradient_accumulation_steps) != 0

        if not is_accumulating:
            self._global_step += 1

        if step is None:
            step = self._global_step
        if lr is None:
            lr = 1e-4

        # Build ONE config and share across all modules (avoids N allocations)
        config = OptimizerConfig(optimizer_type="adamw", lr=lr, **kwargs)

        for m in self._fused_modules:
            m._config = config
            m._is_accumulating = is_accumulating

    def optimizer_state_dict(self) -> dict:
        state = {
            "global_step": self._global_step,
            "micro_step": self._micro_step,
            "modules": {},
        }
        for m in self._fused_modules:
            name = self._fused_module_names[id(m)]
            sd = m.get_optimizer_state_dict()
            if sd is not None:
                state["modules"][name] = sd
        return state

    def load_optimizer_state_dict(self, state: dict):
        self._global_step = state["global_step"]
        self._micro_step = state["micro_step"]
        for m in self._fused_modules:
            name = self._fused_module_names[id(m)]
            if name in state["modules"]:
                m.load_optimizer_state_dict(state["modules"][name])

    @property
    def l2_manager(self):
        return self._l2_manager

    @property
    def num_fused_params(self) -> int:
        return len(self._fused_param_ids)

    @property
    def num_fused_modules(self) -> int:
        return len(self._fused_modules)
