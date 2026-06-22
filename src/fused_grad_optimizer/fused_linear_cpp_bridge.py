"""Python bridge for the fused_linear C++ extension.

Loads the C++ extension (JIT-compiled via torch.utils.cpp_extension.load), and
registers a Python callback the C++ backward uses to dispatch the fused
grad+optimizer kernel (CUTLASS EVT / Triton v2 / FP8-state / int8-state).

Public entry point: `fused_linear_apply_cpp(input, weight, bias, state, config,
is_accumulating)` — drop-in replacement for `FusedLinearFunction.apply(...)`.
"""

import logging
import os

import torch

log = logging.getLogger("fused_linear_cpp")

_module = None


def _build_env():
    # Same arch flags as the EVT module (sm_100a for B200).
    return ["-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1"]


def _get_module():
    global _module
    if _module is not None:
        return _module
    from torch.utils.cpp_extension import load
    here = os.path.dirname(__file__)
    src = os.path.join(here, "csrc", "fused_linear_cpp", "fused_linear.cpp")
    log.info(f"fused_linear_cpp: building from {src}")
    _module = load(
        name="fused_linear_cpp",
        sources=[src],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )
    _module.set_bwd_callback(_bwd_callback)
    return _module


def _bwd_callback(grad_output_2d, input_2d, weight, state, config, is_accumulating):
    """C++ calls this once per linear during backward.

    `state` is a FusedOptimizerState; `config` is an OptimizerConfig.
    Dispatches to the same fused kernels the pure-Python FusedLinearFunction
    used (so the math is identical and tile selection is unchanged).
    """
    from fused_grad_optimizer.autograd import _apply_fused, _apply_precomputed
    if config is None:
        return  # no-op for non-fused case (shouldn't happen on this code path)
    if is_accumulating:
        # Gradient accumulation micro-step: buffer the grad, defer optimizer.
        state.accumulate_grad(grad_output_2d.t() @ input_2d)
        return
    pending = state.pop_accumulated_grad()
    if pending is not None:
        grad_weight_total = (grad_output_2d.t() @ input_2d).float()
        grad_weight_total.add_(pending)
        _apply_precomputed(grad_weight_total, weight, state, config)
    else:
        # No accumulation — use the fused kernel (no grad_W allocation).
        _apply_fused(grad_output_2d, input_2d, weight, state, config, l2_manager=None)


def fused_linear_apply_cpp(input, weight, bias, state, config, is_accumulating):
    """Drop-in for FusedLinearFunction.apply — uses the C++ autograd Function."""
    mod = _get_module()
    return mod.fused_linear_apply(input, weight, bias, state, config, is_accumulating)


def precompile():
    """Build the C++ extension eagerly (e.g. before benchmarks)."""
    _get_module()
