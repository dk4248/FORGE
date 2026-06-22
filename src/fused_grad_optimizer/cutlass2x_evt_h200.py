"""
CUTLASS 2.x EVT fused grad_W + AdamW — H200 runner.

Wraps `csrc/evt_fused/fused_adamw_evt.cu`, which targets arch::Sm80 and uses
synchronous `mma.m16n8k16` tensor-core instructions. It runs on H200 (sm_90a)
via backward-compat — we explicitly gencode for sm_90a. It gives up WGMMA's
2× peak advantage but in return sidesteps the warp-specialized/async-pipeline
machinery that the 3.x Hopper EVT path uses, where deep visitor trees cause
ptxas to serialise `wgmma.mma_async` across function boundaries (C7510) and
drop effective throughput to ~8.5% of peak.

Measured on H200 (SEQ=4096 lm_head-shaped layer):
    3.x Hopper EVT (true-fused)        : 51.1 ms   8.5% of WGMMA peak
    2-kernel Hopper fallback           : 11.9 ms  36.7% of WGMMA peak   ← prod default
    CUTLASS 2.x EVT (this file)        : 14.6 ms  29.8% of Sm80 MMA peak
    Triton persistent                  : 14.1 ms  30.8% of peak

The 2.x EVT is slower than the two-kernel path on H200 because losing WGMMA
costs more than saving the grad_W HBM round-trip. It's kept here as a
verified alternative for benchmarking and as a reference for architectures
where the tradeoff tips the other way (e.g. RTX Blackwell, where the 2.x
path actually wins).

Enable at the benchmark level with:   FUSED_GRAD_EVT_2X=1
"""

import os
import logging

import torch

log = logging.getLogger("cutlass2x_evt_h200")

_module = None


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
    """JIT-compile the CUTLASS 2.x EVT fused AdamW kernel for sm_90a."""
    global _module
    if _module is not None:
        return _module

    _prep_build_env()
    from torch.utils.cpp_extension import load

    here = os.path.dirname(__file__)
    csrc_dir = os.path.join(here, "csrc", "evt_fused")
    cutlass_root = os.path.abspath(os.path.join(here, "..", "..", "cutlass"))
    cutlass_inc = os.path.join(cutlass_root, "include")
    cutlass_util_inc = os.path.join(cutlass_root, "tools", "util", "include")

    _module = load(
        name="fused_adamw_evt_2x_h200",
        sources=[os.path.join(csrc_dir, "fused_adamw_evt.cu")],
        extra_include_paths=[cutlass_inc, cutlass_util_inc, csrc_dir],
        extra_cuda_cflags=[
            "-gencode=arch=compute_90a,code=sm_90a",
            "-std=c++17", "-O3", "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--ftemplate-depth=2048",
        ],
        verbose=False,
    )
    return _module


# Map tile label -> pybind entry point name in the built module.
_TILE_FN_MAP = {
    "128x128": "cutlass_fused_adamw_evt",          # baseline 128×128×32
    "128x256": "cutlass_fused_adamw_evt_128x256",  # wide-N (down_proj)
    "256x128": "cutlass_fused_adamw_evt_256x128",  # wide-M (lm_head, gate/up_proj)
}


def fused_grad_adamw_evt2x(
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
    tile: str = "128x128",
):
    """Single-kernel CUTLASS 2.x EVT fused grad_W + AdamW on H200."""
    assert grad_output.is_contiguous() and input.is_contiguous()
    if tile not in _TILE_FN_MAP:
        raise ValueError(
            f"Unknown tile '{tile}'. Expected one of {list(_TILE_FN_MAP)}.")

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    mod = _get_module()
    getattr(mod, _TILE_FN_MAP[tile])(
        grad_output, input, weight, m, v,
        lr, beta1, beta2, eps, weight_decay,
        bc1, bc2,
    )


def _make_tile_fn(tile: str):
    def _fn(go, inp, w, m, v, lr=1e-4, beta1=0.9, beta2=0.999,
            eps=1e-8, weight_decay=0.01, step=1):
        fused_grad_adamw_evt2x(go, inp, w, m, v,
                                lr=lr, beta1=beta1, beta2=beta2,
                                eps=eps, weight_decay=weight_decay,
                                step=step, tile=tile)
    _fn.__name__ = f"fused_grad_adamw_evt2x_{tile.replace('x', '_')}"
    return _fn


fused_grad_adamw_evt2x_128x128 = _make_tile_fn("128x128")
fused_grad_adamw_evt2x_128x256 = _make_tile_fn("128x256")
fused_grad_adamw_evt2x_256x128 = _make_tile_fn("256x128")
