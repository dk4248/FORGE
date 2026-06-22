"""
H200 variant of cutlass_kernel.py.

Differences vs. the sm_120 (RTX PRO 6000 / Blackwell) original:
  * CUDA_HOME is auto-detected from `nvcc` on PATH (falls back to
    /usr/local/cuda), instead of the hard-coded /usr/local/cuda-13 which
    does not exist on this H200 box (CUDA 12.4 at /usr/local/cuda).
  * -gencode replaced with compute_90a / sm_90a (Hopper).  The kernels
    here use standard WMMA and CUTLASS with ArchTag=Sm80, so sm_90/sm_90a
    is sufficient — we do not emit any WGMMA/TMA-specific PTX.
  * Kept identical function names as cutlass_kernel.py so the benchmark
    can `from fused_grad_optimizer.cutlass_kernel_h200 import ...` with
    a one-line swap.

JIT-compiled via torch.utils.cpp_extension.
"""

import os
import shutil
import torch

_module_v3 = None
_module_v1 = None
_module_cutlass3x = None
_module_opt_only = None
_module_evt = None


def _detect_cuda_home() -> str:
    """Find a working CUDA toolkit root.

    Priority:
      1) $CUDA_HOME if it already points at an existing nvcc.
      2) nvcc on PATH -> its grandparent dir.
      3) /usr/local/cuda (symlink that most installs provide).
      4) /usr/local/cuda-12.4 (this machine's actual install).
    """
    env = os.environ.get("CUDA_HOME")
    if env and os.path.exists(os.path.join(env, "bin", "nvcc")):
        return env

    nvcc = shutil.which("nvcc")
    if nvcc:
        # /usr/local/cuda/bin/nvcc -> /usr/local/cuda
        return os.path.dirname(os.path.dirname(nvcc))

    for cand in ("/usr/local/cuda", "/usr/local/cuda-12.4", "/usr/local/cuda-12"):
        if os.path.exists(os.path.join(cand, "bin", "nvcc")):
            return cand

    raise RuntimeError(
        "Could not locate CUDA toolkit. Set CUDA_HOME or put nvcc on PATH."
    )


# H200 (Hopper) arch flags.  sm_90a enables Hopper-specific warpgroup/TMA
# intrinsics if any downstream CUTLASS code ever needs them; for the current
# kernels (WMMA + Sm80 CUTLASS tag) plain sm_90 would also work.
_H200_ARCH_FLAGS = [
    "-gencode=arch=compute_90a,code=sm_90a",
]


def _prep_build_env():
    """Common env setup shared by every _get_module_* function."""
    import ninja
    os.environ["PATH"] = ninja.BIN_DIR + ":" + os.environ.get("PATH", "")
    os.environ["CUDA_HOME"] = _detect_cuda_home()
    # PyTorch's cpp_extension reads TORCH_CUDA_ARCH_LIST to decide which
    # archs to compile.  Pinning it to 9.0a prevents ninja from also
    # injecting stray sm_90/sm_120 gencodes from visible-device auto-detect.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")


def _get_module_v3():
    """JIT-compile the v3 kernel (cp.async + WMMA pipeline) for H200."""
    global _module_v3
    if _module_v3 is not None:
        return _module_v3

    _prep_build_env()
    from torch.utils.cpp_extension import load

    csrc_dir = os.path.join(os.path.dirname(__file__), "csrc", "v3_wmma")
    _module_v3 = load(
        name="fused_adamw_v3_h200",
        sources=[os.path.join(csrc_dir, "fused_adamw_v3.cu")],
        extra_cuda_cflags=[
            *_H200_ARCH_FLAGS,
            "-std=c++17", "-O3",
            "--use_fast_math",
        ],
        verbose=False,
    )
    return _module_v3


def _get_module_v1():
    """JIT-compile the original WMMA kernel (fallback) for H200."""
    global _module_v1
    if _module_v1 is not None:
        return _module_v1

    _prep_build_env()
    from torch.utils.cpp_extension import load

    csrc_dir = os.path.join(os.path.dirname(__file__), "csrc", "legacy")
    cutlass_include = (
        "/raid/scratch/oti/miniconda3/lib/python3.13/"
        "site-packages/cutlass_library/source/include"
    )

    _module_v1 = load(
        name="fused_adamw_cutlass_h200",
        sources=[os.path.join(csrc_dir, "fused_adamw_cutlass.cu")],
        extra_include_paths=[cutlass_include] if os.path.exists(cutlass_include) else [],
        extra_cuda_cflags=[
            *_H200_ARCH_FLAGS,
            "-std=c++17", "-O3",
            "--use_fast_math",
        ],
        verbose=False,
    )
    return _module_v1


def fused_grad_adamw_cutlass(
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
):
    """Fused weight gradient + AdamW (v3: cp.async pipelined WMMA)."""
    assert grad_output.is_contiguous() and input.is_contiguous()
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    mod = _get_module_v3()
    mod.fused_grad_adamw_v3(
        grad_output, input, weight, m, v,
        lr, beta1, beta2, eps, weight_decay,
        bc1, bc2,
    )


def fused_grad_adamw_cutlass_v1(
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
):
    """Fused weight gradient + AdamW (v1: original WMMA, no pipeline)."""
    assert grad_output.is_contiguous() and input.is_contiguous()
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    mod = _get_module_v1()
    mod.fused_grad_adamw_cutlass(
        grad_output, input, weight, m, v,
        lr, beta1, beta2, eps, weight_decay,
        bc1, bc2,
    )


# ---------------------------------------------------------------------------
# CUTLASS 3.x path (Route 1, Step 1): CollectiveBuilder GEMM + custom AdamW
# ---------------------------------------------------------------------------

def _get_module_cutlass3x():
    """JIT-compile the CUTLASS 3.x bf16 GEMM wrapper for H200."""
    global _module_cutlass3x
    if _module_cutlass3x is not None:
        return _module_cutlass3x

    _prep_build_env()
    from torch.utils.cpp_extension import load

    here = os.path.dirname(__file__)
    csrc_dir = os.path.join(here, "csrc", "cutlass2x_gemm")
    cutlass_root = os.path.abspath(os.path.join(here, "..", "..", "cutlass"))
    cutlass_inc = os.path.join(cutlass_root, "include")
    cutlass_util_inc = os.path.join(cutlass_root, "tools", "util", "include")

    _module_cutlass3x = load(
        name="fused_adamw_cutlass3x_h200",
        sources=[os.path.join(csrc_dir, "fused_adamw_cutlass3x.cu")],
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
    return _module_cutlass3x


def _get_module_opt_only():
    """JIT-compile the 128-bit vectorized AdamW optimizer-only kernel for H200."""
    global _module_opt_only
    if _module_opt_only is not None:
        return _module_opt_only

    _prep_build_env()
    from torch.utils.cpp_extension import load

    csrc_dir = os.path.join(os.path.dirname(__file__), "csrc", "opt_only")
    _module_opt_only = load(
        name="optimizer_only_adamw_cuda_h200",
        sources=[os.path.join(csrc_dir, "optimizer_only_adamw.cu")],
        extra_cuda_cflags=[
            *_H200_ARCH_FLAGS,
            "-std=c++17", "-O3",
            "--use_fast_math",
        ],
        verbose=False,
    )
    return _module_opt_only


def fused_grad_adamw_cutlass3x(
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
):
    """CUTLASS 2.x GEMM -> temp grad_W (bf16) -> custom CUDA AdamW step."""
    assert grad_output.is_contiguous() and input.is_contiguous()

    V = grad_output.shape[1]
    H = input.shape[1]

    grad_w = torch.empty((V, H), dtype=torch.bfloat16, device=weight.device)
    mod_gemm = _get_module_cutlass3x()
    mod_gemm.cutlass3x_grad_w_bf16(grad_output, input, grad_w)

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    mod_opt = _get_module_opt_only()
    mod_opt.optimizer_only_adamw_cuda(
        grad_w, weight, m, v,
        lr, beta1, beta2, eps, weight_decay,
        bc1, bc2,
    )


# ---------------------------------------------------------------------------
# EVT (fused grad_W + AdamW in one kernel)
# ---------------------------------------------------------------------------

def _get_module_evt():
    """JIT-compile the EVT-fused grad_W+AdamW kernel for H200."""
    global _module_evt
    if _module_evt is not None:
        return _module_evt

    _prep_build_env()
    from torch.utils.cpp_extension import load

    here = os.path.dirname(__file__)
    csrc_dir = os.path.join(here, "csrc", "evt_fused")
    cutlass_root = os.path.abspath(os.path.join(here, "..", "..", "cutlass"))
    cutlass_inc = os.path.join(cutlass_root, "include")
    cutlass_util_inc = os.path.join(cutlass_root, "tools", "util", "include")

    _module_evt = load(
        name="fused_adamw_evt_h200",
        sources=[os.path.join(csrc_dir, "fused_adamw_evt.cu")],
        extra_include_paths=[cutlass_inc, cutlass_util_inc, csrc_dir],
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
    return _module_evt


def _evt_call(shape_tag, grad_output, input, weight, m, v,
              lr, beta1, beta2, eps, weight_decay, step):
    assert grad_output.is_contiguous() and input.is_contiguous()
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    mod = _get_module_evt()
    fn = {
        "128x128": mod.cutlass_fused_adamw_evt,
        "128x256": mod.cutlass_fused_adamw_evt_128x256,
        "256x128": mod.cutlass_fused_adamw_evt_256x128,
    }[shape_tag]
    fn(grad_output, input, weight, m, v,
       lr, beta1, beta2, eps, weight_decay, bc1, bc2)


def fused_grad_adamw_evt(
    grad_output, input, weight, m, v,
    lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=1,
):
    _evt_call("128x128", grad_output, input, weight, m, v,
              lr, beta1, beta2, eps, weight_decay, step)


def fused_grad_adamw_evt_128x256(
    grad_output, input, weight, m, v,
    lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=1,
):
    _evt_call("128x256", grad_output, input, weight, m, v,
              lr, beta1, beta2, eps, weight_decay, step)


def fused_grad_adamw_evt_256x128(
    grad_output, input, weight, m, v,
    lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=1,
):
    _evt_call("256x128", grad_output, input, weight, m, v,
              lr, beta1, beta2, eps, weight_decay, step)
