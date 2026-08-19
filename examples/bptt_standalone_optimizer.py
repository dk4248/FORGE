"""bptt_standalone_optimizer.py — using FORGE's int8 optimizer with ordinary
backward(), for training loops that backprop through the same weight more
than once per step (BPTT / recurrent objectives).

Why this exists: `wrap_rwkv7()` + the fused-into-backward path (see
`rwkv7_example.py`) assumes each wrapped layer is touched by backward() at
most once per training step — the fused AdamW update runs *inside*
`backward()` itself. That assumption breaks for any objective that
backprops through a recurrent/shared weight across multiple timesteps in
one differentiated graph (e.g. RL over a multi-step "reasoning" chain, or
plain truncated-BPTT language modeling) — the same weight gets touched
more than once inside a single backward() call, which the fused path
can't tolerate no matter how the outer training loop is structured.

The fix: don't fuse anything into backward. Run ordinary autograd — full
BPTT gradient flow, exactly as correct as the non-FORGE path — then apply
FORGE's standalone `optimizer_only_adamw_int8state()` kernel to the
resulting `.grad` tensors afterward. This keeps the larger of FORGE's two
memory wins (int8-quantized optimizer moments: ~4x smaller than fp32
AdamW) without needing the fused-backward machinery at all.

`Int8AdamW` below is a minimal, architecture-agnostic optimizer wrapper
implementing that pattern. `offload_state=True` additionally keeps each
parameter's int8 moment state in CPU RAM, staging it to GPU one parameter
at a time inside the existing per-param step() loop — turns ~depends-on-
model-size GB of resident GPU state into tens of MB of transient use,
at the cost of a PCIe transfer per parameter per step (worthwhile when
you're optimizer-state-bound, not compute-bound).

Measured on a real BPTT-style RL training loop (RWKV-7, 2.9B params, one
continuous differentiable graph across a multi-step recurrent chain):
fixed cost (weights + first-backward grad buffers, before any
activation/BPTT memory) went from ~17.4GB without offload — already over
a 16GB card's budget regardless of any other tuning — to ~11.8GB with
this pattern + offload_state=True. That's what let the run fit on a T4
16GB at all.
"""
from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


class Int8AdamW:
    """AdamW using FORGE's int8-quantized moment state, applied to
    gradients from *ordinary* (non-fused) backward() — see module
    docstring for when this is the right tool instead of the fused
    autograd path.

    Only 2D parameters whose last dim is divisible by `qblock` are
    int8-optimized (the underlying kernel's own constraint — matches
    typical Linear-layer weight shapes). Everything else (biases,
    LayerNorm, embeddings, or any custom recurrent-state parameters)
    needs a regular optimizer alongside this one — see `other_params`.
    """

    def __init__(self, params: Iterable[nn.Parameter], lr: float = 1e-4,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.01, qblock: int = 64,
                 offload_state: bool = False):
        from fused_grad_optimizer.state import FusedOptimizerState, OptimizerConfig

        params = list(params)
        self.fused_params: List[nn.Parameter] = [
            p for p in params if p.dim() == 2 and p.shape[1] % qblock == 0
        ]
        fused_ids = {id(p) for p in self.fused_params}
        self.other_params = [p for p in params if id(p) not in fused_ids]

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.offload_state = offload_state
        self._OptimizerConfig = OptimizerConfig
        self._states = {
            id(p): FusedOptimizerState(p, optimizer_type="adamw",
                                        state_mode="int8", qblock_size=qblock)
            for p in self.fused_params
        }
        if offload_state:
            for state in self._states.values():
                state.ensure_buffers()  # allocates on the weight's device (GPU)
                state.m_q = state.m_q.to("cpu")
                state.v_q = state.v_q.to("cpu")
                state.m_scale = state.m_scale.to("cpu")
                state.v_scale = state.v_scale.to("cpu")

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.fused_params:
            p.grad = None if set_to_none else (
                p.grad.zero_() if p.grad is not None else None)

    def step(self) -> None:
        from fused_grad_optimizer.autograd import _apply_precomputed
        config = self._OptimizerConfig(
            optimizer_type="adamw", lr=self.lr, beta1=self.beta1,
            beta2=self.beta2, eps=self.eps, weight_decay=self.weight_decay,
        )
        for p in self.fused_params:
            if p.grad is None:
                continue
            state = self._states[id(p)]
            if self.offload_state:
                dev = p.device
                state.m_q = state.m_q.to(dev)
                state.v_q = state.v_q.to(dev)
                state.m_scale = state.m_scale.to(dev)
                state.v_scale = state.v_scale.to(dev)
            _apply_precomputed(p.grad.float(), p.data, state, config)
            if self.offload_state:
                state.m_q = state.m_q.to("cpu")
                state.v_q = state.v_q.to("cpu")
                state.m_scale = state.m_scale.to("cpu")
                state.v_scale = state.v_scale.to("cpu")


if __name__ == "__main__":
    # Toy BPTT loop: a shared Linear layer applied recurrently over T
    # timesteps in one differentiated graph — the exact shape of workload
    # the fused-into-backward path can't handle, and this pattern can.
    D, T = 256, 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cell = nn.Linear(D, D).to(device)
    non_fused = [p for p in cell.parameters()
                 if not (p.dim() == 2 and p.shape[1] % 64 == 0)]
    opt = Int8AdamW(cell.parameters(), lr=1e-4, offload_state=(device == "cuda"))
    non_fused_opt = torch.optim.AdamW(non_fused, lr=1e-4) if non_fused else None

    x = torch.randn(4, D, device=device)
    for step in range(3):
        h = x
        for _ in range(T):          # same weight touched T times per backward()
            h = torch.tanh(cell(h))
        loss = h.pow(2).mean()

        opt.zero_grad()
        if non_fused_opt:
            non_fused_opt.zero_grad()
        loss.backward()             # ordinary autograd — full BPTT, no truncation
        opt.step()
        if non_fused_opt:
            non_fused_opt.step()

        print(f"step {step}: loss={loss.item():.4f}")
