"""
Experimental Sm100 EVT AdamW dispatcher for B200.

Sibling of cutlass_evt_b200.py. Loads a separate JIT extension built from
csrc/cutlass_b200_evt/fused_adamw_evt_sm100_exp.cu with new variants:

  * Change C — `*_fast`: rsqrt+fma w_update functor (3-5x faster MFU chain
    on the AdamW critical path; numerics within bf16 quantisation noise).
  * Change E — `256x256_4cl[_fast]`: cluster<4,1,1> EVT variant. Re-tests
    the cluster<4,1,1> path (rejected on no-EVT in T2.6 with -14% on lm_head)
    with the heavier EVT epilogue and TMA multicast on B.
  * Change F — `fused_grad_adamw_evt_b200_exp_multistream`: per-layer
    multi-stream wrapper for the production EVT kernel, to fill SMs that go
    idle on tail waves of small layers.

Used by the experimental benchmark harness; production code keeps using
cutlass_evt_b200.py until a winner is selected.
"""

import os
import shutil
import logging
import torch

log = logging.getLogger("cutlass_evt_b200_exp")

_module = None
_B200_ARCH_FLAGS = ["-gencode=arch=compute_100a,code=sm_100a"]


def _detect_cuda_home() -> str:
    env = os.environ.get("CUDA_HOME")
    if env and os.path.exists(os.path.join(env, "bin", "nvcc")):
        return env
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(nvcc))
    raise RuntimeError("Could not locate CUDA 12.8+ toolkit (need it for sm_100a).")


def _find_cutlass_root() -> str:
    if os.environ.get("CUTLASS_PATH"):
        c = os.environ["CUTLASS_PATH"]
        if os.path.exists(os.path.join(c, "include", "cutlass", "cutlass.h")):
            return c
    here = os.path.dirname(__file__)
    repo = os.path.abspath(os.path.join(here, "..", "..", "cutlass"))
    if os.path.exists(os.path.join(repo, "include", "cutlass", "cutlass.h")):
        return repo
    raise RuntimeError("CUTLASS not found. Clone it via clone_cutlass.sh.")


def _prep_build_env():
    import ninja
    os.environ["PATH"] = ninja.BIN_DIR + ":" + os.environ.get("PATH", "")
    os.environ["CUDA_HOME"] = _detect_cuda_home()
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"


def _get_module():
    global _module
    if _module is not None:
        return _module
    _prep_build_env()
    from torch.utils.cpp_extension import load
    here = os.path.dirname(__file__)
    csrc_dir = os.path.join(here, "csrc", "cutlass_b200_evt")
    cutlass_root = _find_cutlass_root()
    log.info(f"CUTLASS B200 EVT (exp): building from {cutlass_root}")
    _module = load(
        name="fused_adamw_evt_sm100_exp",
        sources=[os.path.join(csrc_dir, "fused_adamw_evt_sm100_exp.cu")],
        extra_include_paths=[
            os.path.join(cutlass_root, "include"),
            os.path.join(cutlass_root, "tools", "util", "include"),
        ],
        extra_cuda_cflags=[
            *_B200_ARCH_FLAGS,
            "-std=c++17", "-O3", "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
            "--ftemplate-depth=2048",
        ],
        verbose=True,
    )
    return _module


# Map tile-string → bound module symbol name.
_TILE_FN_MAP = {
    # Production-equivalent tiles (re-built with _exp module name).
    "128x128_1sm":               "blackwell_evt_adamw_exp_128x128_1sm",
    "256x128_2sm":               "blackwell_evt_adamw_exp_256x128_2sm",
    "256x256_2sm":               "blackwell_evt_adamw_exp_256x256_2sm",
    "128x128_1sm_streamk":       "blackwell_evt_adamw_exp_128x128_1sm_streamk",
    "256x256_2sm_streamk":       "blackwell_evt_adamw_exp_256x256_2sm_streamk",
    # Change C (rsqrt+fma w_update)
    "128x128_1sm_fast":          "blackwell_evt_adamw_exp_128x128_1sm_fast",
    "256x128_2sm_fast":          "blackwell_evt_adamw_exp_256x128_2sm_fast",
    "256x256_2sm_fast":          "blackwell_evt_adamw_exp_256x256_2sm_fast",
    # Change E (cluster<4,1,1>)
    "256x256_4cl":               "blackwell_evt_adamw_exp_256x256_4cl",
    "256x256_4cl_fast":          "blackwell_evt_adamw_exp_256x256_4cl_fast",
}


ALL_TILES = list(_TILE_FN_MAP.keys())


def fused_grad_adamw_evt_b200_exp(
    grad_output, input, weight, m, v,
    lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=1,
    tile="256x128_2sm_fast",
):
    """Drop-in but lets caller pick any of the experimental tiles directly."""
    assert grad_output.is_contiguous() and input.is_contiguous()
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    mod = _get_module()
    if tile not in _TILE_FN_MAP:
        raise ValueError(f"unknown tile {tile!r}; valid: {sorted(_TILE_FN_MAP)}")
    getattr(mod, _TILE_FN_MAP[tile])(
        grad_output, input, weight, m, v,
        lr, beta1, beta2, eps, weight_decay, bc1, bc2,
    )


# ─── Change F: multi-stream per-layer dispatcher ──────────────────────────
# When the optimizer-step wraps a list of independent layers, dispatching them
# in round-robin on N CUDA streams lets the GPU overlap tail-wave SMs of layer
# K with the start of layer K+1. Each kernel is launched on its own stream;
# cross-stream dependence is None (each layer's GO/INP/W/m/v are disjoint).
def fused_grad_adamw_evt_b200_exp_multistream(
    layer_args,
    *,
    tile="256x128_2sm_fast",
    n_streams=2,
    lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=1,
):
    """
    layer_args: list of (grad_output, input, weight, m, v) tuples.
    Dispatches each layer to a CUDA stream (round-robin over n_streams),
    then synchronises all streams against the default stream before returning.
    """
    if not layer_args:
        return
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    mod = _get_module()
    fn = getattr(mod, _TILE_FN_MAP[tile])
    streams = [torch.cuda.Stream() for _ in range(n_streams)]
    default_stream = torch.cuda.current_stream()
    for s in streams:
        s.wait_stream(default_stream)
    for i, (go, inp, w, m, v) in enumerate(layer_args):
        s = streams[i % n_streams]
        with torch.cuda.stream(s):
            fn(go, inp, w, m, v, lr, beta1, beta2, eps, weight_decay, bc1, bc2)
    for s in streams:
        default_stream.wait_stream(s)


def precompile():
    _get_module()
