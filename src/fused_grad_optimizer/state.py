"""
Optimizer state and configuration for fused gradient+optimizer kernels.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class OptimizerConfig:
    """
    Snapshot of optimizer hyperparameters for one step.
    Create a fresh one each step so LR schedulers work naturally.
    """
    optimizer_type: str = "adamw"    # "sgd" or "adamw"
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01

    def __init__(self, optimizer_type="adamw", lr=1e-4, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.01, **kwargs):
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay


class FusedOptimizerState:
    """
    Optimizer state buffers for a single weight parameter.

    For AdamW: stores first moment (m) and second moment (v).
    For SGD: no state buffers needed.

    m and v are allocated LAZILY on first optimizer update, not at init.
    This matches PyTorch AdamW behavior and avoids holding 2x fp32 copies
    of every fused weight before training even starts.
    """

    def __init__(self, weight: torch.Tensor, optimizer_type: str = "adamw",
                 state_dtype: Optional[torch.dtype] = None,
                 quantize_state: bool = False, qblock_size: int = 64,
                 state_mode: Optional[str] = None):
        self.optimizer_type = optimizer_type
        self.step = 0
        self._pending_grad: Optional[torch.Tensor] = None
        self._weight_ref = weight  # only used for shape/device at alloc time
        # Match param dtype by default (same as PyTorch AdamW).
        # Kernel upcasts to fp32 internally for the update math.
        self._state_dtype = state_dtype if state_dtype is not None else weight.dtype

        # state_mode ∈ {"bf16", "int8", "fp8"} — orthogonal to optimizer_type.
        # Legacy: quantize_state=True ⇒ state_mode="int8" (backward compat).
        if state_mode is None:
            state_mode = "int8" if quantize_state else "bf16"
        if state_mode not in ("bf16", "int8", "fp8"):
            raise ValueError(f"Unknown state_mode: {state_mode}")
        self.state_mode = state_mode
        self.quantize_state = (state_mode == "int8")
        self.qblock_size = qblock_size

        if optimizer_type == "adamw":
            if self.state_mode == "int8":
                self.m_q: Optional[torch.Tensor] = None   # int8 (V, H)
                self.v_q: Optional[torch.Tensor] = None   # int8 (V, H)
                self.m_scale: Optional[torch.Tensor] = None  # fp32 (V, H // qblock)
                self.v_scale: Optional[torch.Tensor] = None  # fp32 (V, H // qblock)
            elif self.state_mode == "fp8":
                # FP8 per-tensor scaled — see kernel_fp8_state.py
                self.m_fp8: Optional[torch.Tensor] = None
                self.v_fp8: Optional[torch.Tensor] = None
                self.m_scale: Optional[torch.Tensor] = None       # () fp32
                self.v_scale: Optional[torch.Tensor] = None       # () fp32
                self.m_absmax_next: Optional[torch.Tensor] = None # () fp32
                self.v_absmax_next: Optional[torch.Tensor] = None # () fp32
                # Per-tile absmax scratch (replaces atomic_max). Sized at
                # ensure_buffers() time using the smallest-BLOCK tiles-per-V-H.
                self.m_absmax_scratch: Optional[torch.Tensor] = None
                self.v_absmax_scratch: Optional[torch.Tensor] = None
            else:
                self.m: Optional[torch.Tensor] = None
                self.v: Optional[torch.Tensor] = None
        elif optimizer_type == "sgd":
            pass
        else:
            raise ValueError(
                f"Unknown optimizer: {optimizer_type}. Use 'sgd' or 'adamw'."
            )

    def ensure_buffers(self):
        """Allocate m/v on first use. Called right before the optimizer update."""
        if self.optimizer_type == "adamw":
            if self.state_mode == "int8":
                if self.m_q is None:
                    w = self._weight_ref
                    V, H = w.shape
                    device = w.device
                    self.m_q = torch.zeros(V, H, dtype=torch.int8, device=device)
                    self.v_q = torch.zeros(V, H, dtype=torch.int8, device=device)
                    scale_cols = H // self.qblock_size
                    self.m_scale = torch.ones(V, scale_cols, dtype=torch.float32, device=device)
                    self.v_scale = torch.ones(V, scale_cols, dtype=torch.float32, device=device)
            elif self.state_mode == "fp8":
                if self.m_fp8 is None:
                    from fused_grad_optimizer.kernel_fp8_state import max_scratch_tiles
                    w = self._weight_ref
                    V, H = w.shape
                    device = w.device
                    self.m_fp8 = torch.zeros(V, H, dtype=torch.float8_e4m3fn, device=device)
                    self.v_fp8 = torch.zeros(V, H, dtype=torch.float8_e5m2,   device=device)
                    # Per-tensor scalars. Init scale=1.0 (neutral for zero state).
                    self.m_scale       = torch.ones((),  dtype=torch.float32, device=device)
                    self.v_scale       = torch.ones((),  dtype=torch.float32, device=device)
                    self.m_absmax_next = torch.zeros((), dtype=torch.float32, device=device)
                    self.v_absmax_next = torch.zeros((), dtype=torch.float32, device=device)
                    # Per-tile absmax scratch — one fp32 per 128x128 tile.
                    # The main kernel writes tile-max here; a tiny reduce
                    # kernel folds it into _absmax_next.
                    NT = max_scratch_tiles(V, H)
                    self.m_absmax_scratch = torch.zeros(NT, dtype=torch.float32, device=device)
                    self.v_absmax_scratch = torch.zeros(NT, dtype=torch.float32, device=device)
            else:
                if self.m is None:
                    self.m = torch.zeros_like(self._weight_ref, dtype=self._state_dtype)
                    self.v = torch.zeros_like(self._weight_ref, dtype=self._state_dtype)

    def pre_step_fp8(self):
        """For fp8 state only. Promote last step's absmax to this step's
        scale, then zero the accumulator. No-op for bf16/int8. Called from
        autograd._apply_fused right before the kernel launch."""
        if self.state_mode != "fp8" or self.m_fp8 is None:
            return
        from fused_grad_optimizer.kernel_fp8_state import FP8_E4M3_MAX, FP8_E5M2_MAX
        MIN_AM = 1e-8
        self.m_scale.copy_(torch.clamp(self.m_absmax_next, min=MIN_AM) / FP8_E4M3_MAX)
        self.v_scale.copy_(torch.clamp(self.v_absmax_next, min=MIN_AM) / FP8_E5M2_MAX)
        self.m_absmax_next.zero_()
        self.v_absmax_next.zero_()

    def increment_step(self):
        self.step += 1

    def accumulate_grad(self, grad: torch.Tensor):
        """Buffer a micro-batch gradient for later fused update."""
        g = grad.detach().float()
        if self._pending_grad is None:
            self._pending_grad = g.clone()
        else:
            self._pending_grad.add_(g)

    def pop_accumulated_grad(self) -> Optional[torch.Tensor]:
        """Return and clear buffered gradients."""
        g = self._pending_grad
        self._pending_grad = None
        return g

    def has_accumulated_grad(self) -> bool:
        return self._pending_grad is not None

    def state_dict(self) -> dict:
        d = {"optimizer_type": self.optimizer_type, "step": self.step,
             "quantize_state": self.quantize_state}
        if self.optimizer_type == "adamw":
            if self.quantize_state:
                if self.m_q is not None:
                    d["m_q"] = self.m_q.clone()
                    d["v_q"] = self.v_q.clone()
                    d["m_scale"] = self.m_scale.clone()
                    d["v_scale"] = self.v_scale.clone()
            else:
                if self.m is not None:
                    d["m"] = self.m.clone()
                    d["v"] = self.v.clone()
        return d

    def load_state_dict(self, d: dict):
        self.step = d["step"]
        if self.optimizer_type == "adamw":
            if self.quantize_state and "m_q" in d:
                self.m_q = d["m_q"].clone()
                self.v_q = d["v_q"].clone()
                self.m_scale = d["m_scale"].clone()
                self.v_scale = d["v_scale"].clone()
            elif not self.quantize_state and "m" in d:
                self.m = d["m"].clone()
                self.v = d["v"].clone()
