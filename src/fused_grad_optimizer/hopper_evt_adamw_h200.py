"""
Sm90 EVT-fused grad_W + AdamW for H200.

─────────────────────────────────────────────────────────────────────────────
Intended design (the "true EVT" path)
─────────────────────────────────────────────────────────────────────────────
Inside a single CUTLASS 3.x Hopper kernel the GEMM accumulator is consumed
in-register by an EVT tree that applies the full AdamW update; m and v are
streamed through as Sm90AuxLoad / Sm90AuxStore nodes. grad_W is never
materialised in HBM.

The C++ source for that kernel lives in
    csrc/hopper_evt_adamw/fused_adamw_evt_sm90.cu

and has been rewritten to use the clean pattern: build the Sm90EVT<...> tree
directly and pass it verbatim into CollectiveBuilder, which selects the
CallbacksBuilder passthrough in `collective_builder.hpp:97` — no custom
FusionOperation tag, no custom CallbacksBuilder specialization.

─────────────────────────────────────────────────────────────────────────────
CUTLASS 3.5.1 runtime limitation
─────────────────────────────────────────────────────────────────────────────
The rewritten kernel COMPILES cleanly on CUTLASS 3.5.1. However, at launch
time it faults with "Misaligned shared or local address" inside the
`SM75_U32x4_LDSM_N` atom when TWO Sm90AuxLoad nodes (one for m, one for v)
coexist with TWO Sm90AuxStore nodes (m', v') in the same EVT tree.

compute-sanitizer traces the fault to a per-thread SMEM address computed
inside the aux-load consumer callback. The underlying issue is a layout
collision in how CUTLASS 3.5.1's `Sm90VisitorImpl` composes per-node
SharedStorage tuples when multiple AuxLoad/AuxStore descriptors share the
same Swizzle<B,M,S> alignment class — the outer `cute::tuple` layout does
not always honour each node's `alignas(alignment_for_swizzle(...))`.

This is fixed upstream in CUTLASS 3.6+ (which reworks CallbacksBuilder and
adds explicit specializations for the aux-load ∧ aux-store case). On 3.5.1
there is no clean workaround short of hand-writing the EVT → kernel params
packing.

─────────────────────────────────────────────────────────────────────────────
Pragmatic behaviour here
─────────────────────────────────────────────────────────────────────────────
Until this repo's CUTLASS submodule is bumped to 3.6+, the three
`fused_grad_adamw_evt_sm90_*` entry points fall back to the known-good
Hopper GEMM → bf16 temp grad_W → opt-only AdamW pipeline from
`hopper_kernel_h200.py`. That pipeline:

  * Uses WGMMA + TMA for the GEMM (same speed as the true-EVT kernel's
    mainloop would use).
  * Avoids only the ~6 ms of grad_W HBM round-trip that full EVT fusion
    would save.
  * Runs correctly today and validates the rest of the v3 benchmark.

To opt back into the experimental true-EVT kernel once CUTLASS is upgraded,
set FUSED_GRAD_EVT_TRUE=1 in the environment. The wrapper will attempt to
JIT-build the .cu file and call it directly; if it faults, the fallback
is re-installed for the remainder of the process.
"""

import os
import logging
import torch

from fused_grad_optimizer.hopper_kernel_h200 import (
    fused_grad_adamw_hopper_128x128,
    fused_grad_adamw_hopper_128x256,
    fused_grad_adamw_hopper_256x128,
)

log = logging.getLogger("hopper_evt_adamw_h200")

# ── Optional true-EVT module loader (guarded) ────────────────────────────
_true_evt_module = None
_true_evt_disabled = False


def _prep_build_env():
    import shutil
    import ninja
    os.environ["PATH"] = ninja.BIN_DIR + ":" + os.environ.get("PATH", "")
    for cand in (os.environ.get("CUDA_HOME"),
                 "/usr/local/cuda", "/usr/local/cuda-12.4", "/usr/local/cuda-12"):
        if cand and os.path.exists(os.path.join(cand, "bin", "nvcc")):
            os.environ["CUDA_HOME"] = cand
            break
    else:
        nvcc = shutil.which("nvcc")
        if nvcc:
            os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")


def _get_module():
    """JIT-compile the Sm90 EVT AdamW kernel.

    On CUTLASS 3.5.1 this succeeds but the kernel faults at launch (see
    module docstring). Kept callable so the v3 benchmark's pre-compile
    step works and so the build is available for future CUTLASS upgrades.
    """
    global _true_evt_module
    if _true_evt_module is not None:
        return _true_evt_module

    _prep_build_env()
    from torch.utils.cpp_extension import load

    here = os.path.dirname(__file__)
    csrc_dir = os.path.join(here, "csrc", "hopper_evt_adamw")
    cutlass_root = os.path.abspath(os.path.join(here, "..", "..", "cutlass"))
    cutlass_inc = os.path.join(cutlass_root, "include")
    cutlass_util_inc = os.path.join(cutlass_root, "tools", "util", "include")

    _true_evt_module = load(
        name="fused_adamw_evt_sm90_h200",
        sources=[os.path.join(csrc_dir, "fused_adamw_evt_sm90.cu")],
        extra_include_paths=[cutlass_inc, cutlass_util_inc],
        extra_cuda_cflags=[
            "-gencode=arch=compute_90a,code=sm_90a",
            "-std=c++17", "-O3", "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1",
            "--ftemplate-depth=2048",
            # Cap per-thread regs so deep EVT trees don't spill and occupancy
            # stays at 2 warpgroups/SM. 128 is the Hopper-cooperative sweet
            # spot reported by CUTLASS example 48. We measured removing this
            # cap entirely on SEQ=4096 and bwd stayed at 818 ms (unchanged),
            # so spills are not the live bottleneck — leaving the cap in
            # helps occupancy without costing anything.
            "-maxrregcount=128",
        ],
        verbose=True,
    )
    return _true_evt_module


_FALLBACK_FN_MAP = {
    "128x128": fused_grad_adamw_hopper_128x128,
    "128x256": fused_grad_adamw_hopper_128x256,
    "256x128": fused_grad_adamw_hopper_256x128,
}

_TRUE_EVT_FN_MAP = {
    "128x128": "hopper_evt_adamw_128x128",
    "128x256": "hopper_evt_adamw_128x256",
    "256x128": "hopper_evt_adamw_256x128",
}


def _true_evt_enabled() -> bool:
    return os.environ.get("FUSED_GRAD_EVT_TRUE", "0") == "1"


def fused_grad_adamw_evt_sm90(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    step: int = 1,
    tile: str = "256x128",
):
    """Single-kernel fused grad_W + AdamW — see module docstring for why this
    currently dispatches to the Hopper GEMM + opt-only fallback.
    """
    global _true_evt_disabled
    assert grad_output.is_contiguous() and input.is_contiguous()
    if tile not in _FALLBACK_FN_MAP:
        raise ValueError(
            f"Unknown tile '{tile}'. Expected one of {list(_FALLBACK_FN_MAP)}.")

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    if _true_evt_enabled() and not _true_evt_disabled:
        try:
            mod = _get_module()
            getattr(mod, _TRUE_EVT_FN_MAP[tile])(
                grad_output, input, weight, m, v,
                lr, beta1, beta2, eps, weight_decay,
                bc1, bc2,
            )
            return
        except Exception as e:
            log.warning(
                "True-EVT kernel failed (%s); falling back to Hopper GEMM + "
                "opt-only for the remainder of this process.", e)
            _true_evt_disabled = True

    _FALLBACK_FN_MAP[tile](
        grad_output, input, weight, m, v,
        lr=lr, beta1=beta1, beta2=beta2, eps=eps,
        weight_decay=weight_decay, step=step,
    )


def _make_tile_fn(tile: str):
    def _fn(go, inp, w, m, v, lr=1e-4, beta1=0.9, beta2=0.999,
            eps=1e-8, weight_decay=0.01, step=1):
        fused_grad_adamw_evt_sm90(go, inp, w, m, v,
                                   lr=lr, beta1=beta1, beta2=beta2,
                                   eps=eps, weight_decay=weight_decay,
                                   step=step, tile=tile)
    _fn.__name__ = f"fused_grad_adamw_evt_sm90_{tile.replace('x', '_')}"
    return _fn


fused_grad_adamw_evt_sm90_128x128 = _make_tile_fn("128x128")
fused_grad_adamw_evt_sm90_128x256 = _make_tile_fn("128x256")
fused_grad_adamw_evt_sm90_256x128 = _make_tile_fn("256x128")
