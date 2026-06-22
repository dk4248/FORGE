"""
Hopper Stream-K variant for H200 (sm_90a).

Same "GEMM -> temp grad_W (bf16) -> optimizer-only kernel" structure as
hopper_kernel_h200.py, but the CUTLASS 3.x GEMM uses `StreamKScheduler`
instead of `PersistentScheduler`.  At short K (SEQ_LEN=512 → K=512),
Stream-K splits the K reduction across multiple CTAs, which hides the
3-4-stage TMA+WGMMA pipeline fill cost that otherwise swamps tiny-K GEMMs.

Exposes four tile variants (matches hopper_kernel_h200.py):
    "128x128" | "128x256" | "256x128" | "64x128"
Default "256x128" is usually best for wide-M layers (lm_head / gate / up).
"""

import os
import shutil
import torch

_module = None
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


def _get_module():
    global _module
    if _module is not None:
        return _module

    _prep_build_env()
    from torch.utils.cpp_extension import load

    here = os.path.dirname(__file__)
    csrc_dir = os.path.join(here, "csrc", "hopper_streamk")
    cutlass_root = os.path.abspath(os.path.join(here, "..", "..", "cutlass"))
    cutlass_inc = os.path.join(cutlass_root, "include")
    cutlass_util_inc = os.path.join(cutlass_root, "tools", "util", "include")

    _module = load(
        name="fused_adamw_hopper_streamk_h200",
        sources=[os.path.join(csrc_dir, "fused_adamw_hopper_streamk.cu")],
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
        ],
        verbose=True,
    )
    return _module


def _get_module_opt_only():
    global _module_opt_only
    if _module_opt_only is not None:
        return _module_opt_only

    _prep_build_env()
    from torch.utils.cpp_extension import load

    csrc_dir = os.path.join(os.path.dirname(__file__), "csrc", "opt_only")
    _module_opt_only = load(
        name="optimizer_only_adamw_cuda_h200_streamk",
        sources=[os.path.join(csrc_dir, "optimizer_only_adamw.cu")],
        extra_cuda_cflags=[
            *_H200_ARCH_FLAGS,
            "-std=c++17", "-O3", "--use_fast_math",
        ],
        verbose=False,
    )
    return _module_opt_only


_TILE_FN_MAP = {
    "128x128": "hopper_streamk_grad_w_bf16_128x128",
    "128x256": "hopper_streamk_grad_w_bf16_128x256",
    "256x128": "hopper_streamk_grad_w_bf16_256x128",
}


def fused_grad_adamw_hopper_streamk(
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
    """Stream-K Hopper GEMM -> temp grad_W (bf16) -> CUDA AdamW step.

    tile: "128x128" | "128x256" | "256x128" (default) | "64x128"
    """
    assert grad_output.is_contiguous() and input.is_contiguous()
    if tile not in _TILE_FN_MAP:
        raise ValueError(f"Unknown tile '{tile}'. Expected {list(_TILE_FN_MAP)}.")

    V = grad_output.shape[1]
    H = input.shape[1]

    grad_w = torch.empty((V, H), dtype=torch.bfloat16, device=weight.device)
    mod_gemm = _get_module()
    getattr(mod_gemm, _TILE_FN_MAP[tile])(grad_output, input, grad_w)

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    mod_opt = _get_module_opt_only()
    mod_opt.optimizer_only_adamw_cuda(
        grad_w, weight, m, v,
        lr, beta1, beta2, eps, weight_decay,
        bc1, bc2,
    )


def _make_tile_fn(tile: str):
    def _fn(go, inp, w, m, v, lr=1e-4, beta1=0.9, beta2=0.999,
            eps=1e-8, weight_decay=0.01, step=1):
        fused_grad_adamw_hopper_streamk(go, inp, w, m, v,
                                         lr=lr, beta1=beta1, beta2=beta2,
                                         eps=eps, weight_decay=weight_decay,
                                         step=step, tile=tile)
    _fn.__name__ = f"fused_grad_adamw_hopper_streamk_{tile.replace('x', '_')}"
    return _fn


fused_grad_adamw_hopper_streamk_128x128 = _make_tile_fn("128x128")
fused_grad_adamw_hopper_streamk_128x256 = _make_tile_fn("128x256")
fused_grad_adamw_hopper_streamk_256x128 = _make_tile_fn("256x128")
