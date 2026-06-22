"""
Optimized fused grad+AdamW CUDA kernels.

v3: cp.async pipelined WMMA kernel for sm_120 (Blackwell workstation).
    3-stage software pipeline overlaps SMEM loads with tensor core compute.

v1: Original WMMA kernel (synchronous loads, for reference/fallback).

JIT-compiled via torch.utils.cpp_extension.
"""

import os
import torch

_module_v3 = None
_module_v1 = None
_module_cutlass3x = None
_module_opt_only = None
_module_evt = None


def _get_module_v3():
    """JIT-compile the v3 kernel (cp.async + WMMA pipeline)."""
    global _module_v3
    if _module_v3 is not None:
        return _module_v3

    import ninja
    os.environ["PATH"] = ninja.BIN_DIR + ":" + os.environ.get("PATH", "")
    os.environ["CUDA_HOME"] = "/usr/local/cuda-13"

    from torch.utils.cpp_extension import load

    csrc_dir = os.path.join(os.path.dirname(__file__), "csrc", "v3_wmma")

    _module_v3 = load(
        name="fused_adamw_v3",
        sources=[os.path.join(csrc_dir, "fused_adamw_v3.cu")],
        extra_cuda_cflags=[
            "-gencode=arch=compute_120,code=sm_120",
            "-std=c++17", "-O3",
            "--use_fast_math",
        ],
        verbose=False,
    )
    return _module_v3


def _get_module_v1():
    """JIT-compile the original WMMA kernel (fallback)."""
    global _module_v1
    if _module_v1 is not None:
        return _module_v1

    import ninja
    os.environ["PATH"] = ninja.BIN_DIR + ":" + os.environ.get("PATH", "")
    os.environ["CUDA_HOME"] = "/usr/local/cuda-13"

    from torch.utils.cpp_extension import load

    csrc_dir = os.path.join(os.path.dirname(__file__), "csrc", "legacy")
    cutlass_include = (
        "/raid/scratch/oti/miniconda3/lib/python3.13/"
        "site-packages/cutlass_library/source/include"
    )

    _module_v1 = load(
        name="fused_adamw_cutlass",
        sources=[os.path.join(csrc_dir, "fused_adamw_cutlass.cu")],
        extra_include_paths=[cutlass_include] if os.path.exists(cutlass_include) else [],
        extra_cuda_cflags=[
            "-gencode=arch=compute_120,code=sm_120",
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
# CUTLASS 3.x path (Route 1, Step 1): GEMM via CollectiveBuilder → temp bf16
# grad_W buffer, then Triton optimizer_only_adamw applies the AdamW update.
#
# This is a first cut to see whether CUTLASS 3.x's sm_120 GEMM is faster than
# our hand-written WMMA kernel. If yes, the next step is to replace the
# LinearCombination epilogue with a custom EVT visitor that fuses AdamW
# in-place (no grad_W materialization).
# ---------------------------------------------------------------------------

def _get_module_cutlass3x():
    """JIT-compile the CUTLASS 3.x bf16 GEMM wrapper."""
    global _module_cutlass3x
    if _module_cutlass3x is not None:
        return _module_cutlass3x

    import ninja
    os.environ["PATH"] = ninja.BIN_DIR + ":" + os.environ.get("PATH", "")
    os.environ["CUDA_HOME"] = "/usr/local/cuda-13"

    from torch.utils.cpp_extension import load

    here = os.path.dirname(__file__)
    csrc_dir = os.path.join(here, "csrc", "cutlass2x_gemm")
    cutlass_root = os.path.abspath(os.path.join(here, "..", "..", "cutlass"))
    cutlass_inc = os.path.join(cutlass_root, "include")
    cutlass_util_inc = os.path.join(cutlass_root, "tools", "util", "include")

    _module_cutlass3x = load(
        name="fused_adamw_cutlass3x",
        sources=[os.path.join(csrc_dir, "fused_adamw_cutlass3x.cu")],
        extra_include_paths=[cutlass_inc, cutlass_util_inc],
        extra_cuda_cflags=[
            "-gencode=arch=compute_120,code=sm_120",
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
    """JIT-compile the custom 128-bit vectorized AdamW optimizer-only kernel."""
    global _module_opt_only
    if _module_opt_only is not None:
        return _module_opt_only

    import ninja
    os.environ["PATH"] = ninja.BIN_DIR + ":" + os.environ.get("PATH", "")
    os.environ["CUDA_HOME"] = "/usr/local/cuda-13"

    from torch.utils.cpp_extension import load

    csrc_dir = os.path.join(os.path.dirname(__file__), "csrc", "opt_only")
    _module_opt_only = load(
        name="optimizer_only_adamw_cuda",
        sources=[os.path.join(csrc_dir, "optimizer_only_adamw.cu")],
        extra_cuda_cflags=[
            "-gencode=arch=compute_120,code=sm_120",
            "-std=c++17", "-O3",
            "--use_fast_math",
        ],
        verbose=False,
    )
    return _module_opt_only


def fused_grad_adamw_cutlass3x(
    grad_output: torch.Tensor,   # (BT, V) bf16
    input: torch.Tensor,         # (BT, H) bf16
    weight: torch.Tensor,        # (V, H)  bf16, in-place
    m: torch.Tensor,             # (V, H)  bf16, in-place
    v: torch.Tensor,             # (V, H)  bf16, in-place
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    step: int = 1,
):
    """Two-step: CUTLASS 2.x GEMM → temp grad_W (bf16) → custom CUDA AdamW step.

    The AdamW step is a 128-bit vectorized CUDA kernel (not Triton) to cut the
    per-layer optimizer cost by avoiding Triton's autotune and kernel-launch
    overhead × 225 layers.
    """
    assert grad_output.is_contiguous() and input.is_contiguous()

    V = grad_output.shape[1]
    H = input.shape[1]

    # Step 1: GEMM → bf16 grad_W in a temp buffer.
    grad_w = torch.empty((V, H), dtype=torch.bfloat16, device=weight.device)
    mod_gemm = _get_module_cutlass3x()
    mod_gemm.cutlass3x_grad_w_bf16(grad_output, input, grad_w)

    # Step 2: in-place AdamW update (custom 128-bit vectorized CUDA kernel).
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    mod_opt = _get_module_opt_only()
    mod_opt.optimizer_only_adamw_cuda(
        grad_w, weight, m, v,
        lr, beta1, beta2, eps, weight_decay,
        bc1, bc2,
    )


# ---------------------------------------------------------------------------
# Route 1 complete: fused grad_W + AdamW via CUTLASS 2.x EVT (Epilogue
# Visitor Tree). Single kernel, no grad_W materialization.
# ---------------------------------------------------------------------------

def _get_module_evt():
    """JIT-compile the EVT-fused grad_W+AdamW kernel."""
    global _module_evt
    if _module_evt is not None:
        return _module_evt

    import ninja
    os.environ["PATH"] = ninja.BIN_DIR + ":" + os.environ.get("PATH", "")
    os.environ["CUDA_HOME"] = "/usr/local/cuda-13"

    from torch.utils.cpp_extension import load

    here = os.path.dirname(__file__)
    csrc_dir = os.path.join(here, "csrc", "evt_fused")
    cutlass_root = os.path.abspath(os.path.join(here, "..", "..", "cutlass"))
    cutlass_inc = os.path.join(cutlass_root, "include")
    cutlass_util_inc = os.path.join(cutlass_root, "tools", "util", "include")

    _module_evt = load(
        name="fused_adamw_evt",
        sources=[os.path.join(csrc_dir, "fused_adamw_evt.cu")],
        extra_include_paths=[cutlass_inc, cutlass_util_inc, csrc_dir],
        extra_cuda_cflags=[
            "-gencode=arch=compute_120,code=sm_120",
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
    """EVT fused (ThreadblockShape=128×128×32, Stages=3) — baseline."""
    _evt_call("128x128", grad_output, input, weight, m, v,
              lr, beta1, beta2, eps, weight_decay, step)


def fused_grad_adamw_evt_128x256(
    grad_output, input, weight, m, v,
    lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=1,
):
    """EVT fused (ThreadblockShape=128×256×32, Stages=3) — favors wide N (down_proj)."""
    _evt_call("128x256", grad_output, input, weight, m, v,
              lr, beta1, beta2, eps, weight_decay, step)


def fused_grad_adamw_evt_256x128(
    grad_output, input, weight, m, v,
    lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=1,
):
    """EVT fused (ThreadblockShape=256×128×32, Stages=3) — favors wide M (lm_head, gate/up)."""
    _evt_call("256x128", grad_output, input, weight, m, v,
              lr, beta1, beta2, eps, weight_decay, step)
