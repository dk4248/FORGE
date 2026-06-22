"""
Blackwell-native CUTLASS GEMM wrapper for B200 (sm_100a).

Builds the CUTLASS 4.x CollectiveBuilder GEMM (tcgen05.mma + TMA + Sm100
persistent scheduler) from csrc/cutlass_b200_gemm/fused_adamw_cutlass_b200.cu
and wires it up as a drop-in replacement for fused_grad_adamw() in the
backward pass:

    grad_W (bf16 temp) = GO^T @ INP        # CUTLASS Sm100 kernel
    W, m, v           <- AdamW(grad_W)     # existing opt_only CUDA kernel

This is the "step 1 fast GEMM" path on B200 — same trade-off as the
hopper_kernel_h200.py wrapper. EVT-fusing the AdamW math into the epilogue
is a follow-up; getting the dense Sm100 GEMM into the benchmark is the
first useful milestone.

Why a separate file rather than reusing hopper_kernel_h200.py:
  * The .cu must instantiate the Sm100 builder, not Sm90 — different
    template path, different cubin gencode flag.
  * `-gencode=arch=compute_100a,code=sm_100a` requires nvcc 12.8+. The
    H200 wrapper uses the older 9.0a flag.
  * Need to point `extra_include_paths` at a CUTLASS install with sm_100
    headers (this repo's vendored cutlass/ submodule isn't checked out;
    flashinfer ships CUTLASS 4.2.1 with full Sm100 support).
"""

import os
import shutil
import logging
import torch

log = logging.getLogger("cutlass_kernel_b200")

_module_gemm = None
_module_opt_only = None

_B200_ARCH_FLAGS = ["-gencode=arch=compute_100a,code=sm_100a"]


def _detect_cuda_home() -> str:
    env = os.environ.get("CUDA_HOME")
    if env and os.path.exists(os.path.join(env, "bin", "nvcc")):
        return env
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(nvcc))
    for cand in ("/usr/local/cuda", "/usr/local/cuda-12.8", "/usr/local/cuda-12"):
        if os.path.exists(os.path.join(cand, "bin", "nvcc")):
            return cand
    raise RuntimeError("Could not locate CUDA toolkit (need 12.8+ for sm_100a).")


def _find_cutlass_root() -> str:
    """Locate a CUTLASS install with sm_100 headers.

    Search order:
      1. $CUTLASS_PATH (explicit override)
      2. The repo's own cutlass/ submodule, if checked out
      3. flashinfer's vendored CUTLASS (CUTLASS 4.2.1, ships with sm_100)
      4. Any cutlass/include/cutlass/cutlass.h discoverable on $PYTHONPATH

    Returns the directory that contains include/ and tools/util/include/.
    """
    candidates = []
    if os.environ.get("CUTLASS_PATH"):
        candidates.append(os.environ["CUTLASS_PATH"])
    # Repo's own cutlass/ — preferred when cloned from upstream so we can
    # pick up post-4.2.1 fixes / scheduler tuning for sm_100.
    here = os.path.dirname(__file__)
    candidates.append(os.path.abspath(os.path.join(here, "..", "..", "cutlass")))
    # flashinfer's vendored CUTLASS 4.2.1 — fallback if the repo's submodule
    # is empty.
    for env_root in (
        "/home/dikshant22176/miniconda3/envs/dikshant/lib/python3.10/"
        "site-packages/flashinfer/data/cutlass",
        "/home/pankaj/miniconda3/envs/fgo_b200/lib/python3.11/"
        "site-packages/flashinfer/data/cutlass",
    ):
        candidates.append(env_root)

    for c in candidates:
        if c and os.path.exists(os.path.join(c, "include", "cutlass", "cutlass.h")):
            return c

    raise RuntimeError(
        "Could not find a CUTLASS install with sm_100 headers. Set "
        "CUTLASS_PATH=/path/to/cutlass to override.")


def _prep_build_env():
    import ninja
    os.environ["PATH"] = ninja.BIN_DIR + ":" + os.environ.get("PATH", "")
    os.environ["CUDA_HOME"] = _detect_cuda_home()
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"


def _get_module_gemm():
    """JIT-compile the CUTLASS Sm100 GEMM wrapper."""
    global _module_gemm
    if _module_gemm is not None:
        return _module_gemm

    _prep_build_env()
    from torch.utils.cpp_extension import load

    here = os.path.dirname(__file__)
    csrc_dir = os.path.join(here, "csrc", "cutlass_b200_gemm")
    cutlass_root = _find_cutlass_root()
    cutlass_inc = os.path.join(cutlass_root, "include")
    cutlass_util_inc = os.path.join(cutlass_root, "tools", "util", "include")

    log.info(f"CUTLASS B200: building from {cutlass_root}")
    _module_gemm = load(
        name="fused_adamw_cutlass_b200",
        sources=[os.path.join(csrc_dir, "fused_adamw_cutlass_b200.cu")],
        extra_include_paths=[cutlass_inc, cutlass_util_inc],
        extra_cuda_cflags=[
            *_B200_ARCH_FLAGS,
            "-std=c++17", "-O3",
            "--use_fast_math",
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
    return _module_gemm


def _get_module_opt_only():
    """JIT-compile the AdamW optimizer-only kernel for sm_100a."""
    global _module_opt_only
    if _module_opt_only is not None:
        return _module_opt_only

    _prep_build_env()
    from torch.utils.cpp_extension import load

    csrc_dir = os.path.join(os.path.dirname(__file__), "csrc", "opt_only")
    _module_opt_only = load(
        name="optimizer_only_adamw_cuda_b200",
        sources=[os.path.join(csrc_dir, "optimizer_only_adamw.cu")],
        extra_cuda_cflags=[
            *_B200_ARCH_FLAGS,
            "-std=c++17", "-O3", "--use_fast_math",
        ],
        verbose=False,
    )
    return _module_opt_only


# ── Tile selection ──────────────────────────────────────────────────────
# B200 has 148 SMs. We pick by *both* shape magnitude and aspect ratio so
# small matmuls (Q/K/V/O at 4096×4096) don't pay tail-wave cost on a big
# tile, while large matmuls (lm_head at 128k×4k) get the max-atom 256×256
# 2SM tile to halve the launch count.

_TILE_FN_MAP = {
    "128x128_1sm": "blackwell_grad_w_bf16_128x128_1sm",
    "128x256_1sm": "blackwell_grad_w_bf16_128x256_1sm",
    "256x128_2sm": "blackwell_grad_w_bf16_256x128_2sm",
    "256x256_2sm": "blackwell_grad_w_bf16_256x256_2sm",
    "256x256_4cl": "blackwell_grad_w_bf16_256x256_4cl",
    # Stream-K variants — split K-dim work across SMs to fill tail waves.
    # Win when tile-count / 148 has bad fractional part.
    "128x128_1sm_streamk": "blackwell_grad_w_bf16_128x128_1sm_streamk",
    "256x256_2sm_streamk": "blackwell_grad_w_bf16_256x256_2sm_streamk",
    # Backwards-compat string keys.
    "128x256": "blackwell_grad_w_bf16_128x256_1sm",
    "256x128": "blackwell_grad_w_bf16_256x128_2sm",
}

_NUM_SMS_B200 = 148


def _waves(M: int, N: int, tile_m: int, tile_n: int) -> float:
    tiles = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    return tiles / _NUM_SMS_B200


def _pick_tile(V: int, H: int) -> str:
    """Pick a tile from the available shapes based on output (M=V, N=H).

    T2 empirical microbench found that on the no-EVT path:

        shape    | 256x128_2sm | 256x256_2sm | winner
        ---------+-------------+-------------+----------------
        qkvo     | 1014 TF/s   |  978 TF/s   | 256x128_2sm  (+3.8%)
        gate_up  | 1107 TF/s   | 1098 TF/s   | 256x128_2sm  (tie)
        down     | 1028 TF/s   | 1030 TF/s   | 256x256_2sm  (tie)
        lmhead   | 1027 TF/s   | 1074 TF/s   | 256x256_2sm  (-4.4%)

    256x256_2sm only wins on very-wide-M shapes (lm_head, M = 32×N) where
    the per-CTA SMEM budget for the lighter no-EVT epilogue can sustain
    enough mainloop stages at 256x256. Everywhere else 256x128_2sm wins
    by ≤4% or ties.

    Decision: dispatch 256x256_2sm only for lm_head-class shapes
    (M ≥ 16·N), 256x128_2sm everywhere else. Keep the streamk / 128x128
    / cluster<4,1,1> variants registered for per-shape ablations.
    """
    M, N = V, H
    if M >= 16 * N:
        return "256x256_2sm"
    return "256x128_2sm"


def fused_grad_adamw_cutlass_b200(
    grad_output: torch.Tensor,   # (BT, V)
    input: torch.Tensor,         # (BT, H)
    weight: torch.Tensor,        # (V, H) — in-place
    m: torch.Tensor,             # (V, H) — in-place
    v: torch.Tensor,             # (V, H) — in-place
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    step: int = 1,
    tile: str | None = None,
):
    """Drop-in replacement for fused_grad_optimizer.kernel.fused_grad_adamw.

    Runs CUTLASS Sm100 bf16 GEMM (grad_W = GO^T @ INP) → bf16 temp →
    optimizer_only_adamw CUDA kernel for the AdamW step.
    """
    assert grad_output.is_contiguous() and input.is_contiguous()

    V = grad_output.shape[1]
    H = input.shape[1]
    if tile is None:
        tile = _pick_tile(V, H)
    if tile not in _TILE_FN_MAP:
        raise ValueError(
            f"Unknown tile '{tile}'. Expected one of {list(_TILE_FN_MAP)}.")

    grad_w = torch.empty((V, H), dtype=torch.bfloat16, device=weight.device)
    mod_gemm = _get_module_gemm()
    getattr(mod_gemm, _TILE_FN_MAP[tile])(grad_output, input, grad_w)

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    mod_opt = _get_module_opt_only()
    mod_opt.optimizer_only_adamw_cuda(
        grad_w, weight, m, v,
        lr, beta1, beta2, eps, weight_decay,
        bc1, bc2,
    )


def patch_dispatch():
    """Install fused_grad_adamw_cutlass_b200 as the backward kernel.

    Returns the previous pair so callers can restore it with
    `restore_dispatch(prev)`. Mirrors kernel_b200_ws.patch_dispatch().
    """
    import fused_grad_optimizer.kernel as _k
    import fused_grad_optimizer.autograd as _a
    prev = (_k.fused_grad_adamw, _a.fused_grad_adamw)
    _k.fused_grad_adamw = fused_grad_adamw_cutlass_b200
    _a.fused_grad_adamw = fused_grad_adamw_cutlass_b200
    return prev


def restore_dispatch(prev):
    import fused_grad_optimizer.kernel as _k
    import fused_grad_optimizer.autograd as _a
    _k.fused_grad_adamw, _a.fused_grad_adamw = prev


def precompile():
    """Force the JIT build now so the first benchmark step doesn't pay it."""
    _get_module_gemm()
    _get_module_opt_only()
