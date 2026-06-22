"""
Hopper-native kernel wrapper for H200 (sm_90a).

Builds the CUTLASS 3.x CollectiveBuilder-based GEMM (WGMMA + TMA +
persistent scheduler) from csrc/hopper_evt/fused_adamw_hopper.cu, and wires
it up as a drop-in replacement for fused_grad_adamw() in the backward pass:

    grad_W (bf16 temp)  = GO^T @ INP        # Hopper kernel, WGMMA+TMA
    W, m, v            <- AdamW(grad_W)     # existing opt_only CUDA kernel

Exposes four tile-shape variants — pick one via `fused_grad_adamw_hopper(...,
tile="128x128" | "128x256" | "256x128" | "64x128")`.  Defaults to 256x128
which fits lm_head / gate_proj / up_proj best (wide-M).

Blackwell callers should continue to use cutlass_kernel.py — nothing here
affects that path.
"""

import os
import shutil
import torch

_module_hopper = None
_module_opt_only = None


def _detect_cuda_home() -> str:
    env = os.environ.get("CUDA_HOME")
    if env and os.path.exists(os.path.join(env, "bin", "nvcc")):
        return env
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(nvcc))
    for cand in ("/usr/local/cuda", "/usr/local/cuda-12.4", "/usr/local/cuda-12"):
        if os.path.exists(os.path.join(cand, "bin", "nvcc")):
            return cand
    raise RuntimeError("Could not locate CUDA toolkit.")


_H200_ARCH_FLAGS = ["-gencode=arch=compute_90a,code=sm_90a"]


def _prep_build_env():
    import ninja
    os.environ["PATH"] = ninja.BIN_DIR + ":" + os.environ.get("PATH", "")
    os.environ["CUDA_HOME"] = _detect_cuda_home()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")


def _get_module_hopper():
    """JIT-compile the CUTLASS 3.x Hopper GEMM wrapper (sm_90a, WGMMA+TMA)."""
    global _module_hopper
    if _module_hopper is not None:
        return _module_hopper

    _prep_build_env()
    from torch.utils.cpp_extension import load

    here = os.path.dirname(__file__)
    csrc_dir = os.path.join(here, "csrc", "hopper_evt")
    cutlass_root = os.path.abspath(os.path.join(here, "..", "..", "cutlass"))
    cutlass_inc = os.path.join(cutlass_root, "include")
    cutlass_util_inc = os.path.join(cutlass_root, "tools", "util", "include")

    _module_hopper = load(
        name="fused_adamw_hopper_h200",
        sources=[os.path.join(csrc_dir, "fused_adamw_hopper.cu")],
        extra_include_paths=[cutlass_inc, cutlass_util_inc],
        extra_cuda_cflags=[
            *_H200_ARCH_FLAGS,
            "-std=c++17", "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            # Sm90 collective kernels compile much faster with relaxed
            # template depth; these flags also help the WGMMA mainloop emit.
            "-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1",
        ],
        verbose=True,
    )
    return _module_hopper


def _get_module_opt_only():
    """JIT-compile the 128-bit vectorized AdamW optimizer-only kernel (sm_90a)."""
    global _module_opt_only
    if _module_opt_only is not None:
        return _module_opt_only

    _prep_build_env()
    from torch.utils.cpp_extension import load

    csrc_dir = os.path.join(os.path.dirname(__file__), "csrc", "opt_only")
    _module_opt_only = load(
        name="optimizer_only_adamw_cuda_h200_hopper",
        sources=[os.path.join(csrc_dir, "optimizer_only_adamw.cu")],
        extra_cuda_cflags=[
            *_H200_ARCH_FLAGS,
            "-std=c++17", "-O3", "--use_fast_math",
        ],
        verbose=False,
    )
    return _module_opt_only


_TILE_FN_MAP = {
    "128x128": "hopper_grad_w_bf16_128x128",
    "128x256": "hopper_grad_w_bf16_128x256",
    "256x128": "hopper_grad_w_bf16_256x128",
    "64x128":  "hopper_grad_w_bf16_64x128",
}


def fused_grad_adamw_hopper(
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
    """Hopper WGMMA+TMA GEMM -> temp grad_W (bf16) -> CUDA AdamW step.

    tile:
      "128x128" | "128x256" | "256x128" (default) | "64x128"
    """
    assert grad_output.is_contiguous() and input.is_contiguous()
    if tile not in _TILE_FN_MAP:
        raise ValueError(
            f"Unknown tile '{tile}'. Expected one of {list(_TILE_FN_MAP)}.")

    V = grad_output.shape[1]
    H = input.shape[1]

    grad_w = torch.empty((V, H), dtype=torch.bfloat16, device=weight.device)
    mod_gemm = _get_module_hopper()
    getattr(mod_gemm, _TILE_FN_MAP[tile])(grad_output, input, grad_w)

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    mod_opt = _get_module_opt_only()
    mod_opt.optimizer_only_adamw_cuda(
        grad_w, weight, m, v,
        lr, beta1, beta2, eps, weight_decay,
        bc1, bc2,
    )


# ── Per-tile callables (so the benchmark can monkey-patch them individually) ──

def _make_tile_fn(tile: str):
    def _fn(go, inp, w, m, v, lr=1e-4, beta1=0.9, beta2=0.999,
            eps=1e-8, weight_decay=0.01, step=1):
        fused_grad_adamw_hopper(go, inp, w, m, v,
                                 lr=lr, beta1=beta1, beta2=beta2,
                                 eps=eps, weight_decay=weight_decay,
                                 step=step, tile=tile)
    _fn.__name__ = f"fused_grad_adamw_hopper_{tile.replace('x', '_')}"
    return _fn


fused_grad_adamw_hopper_128x128 = _make_tile_fn("128x128")
fused_grad_adamw_hopper_128x256 = _make_tile_fn("128x256")
fused_grad_adamw_hopper_256x128 = _make_tile_fn("256x128")
fused_grad_adamw_hopper_64x128  = _make_tile_fn("64x128")
