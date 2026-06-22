"""
Sm100 EVT-fused grad_W + AdamW wrapper for B200.

True fused kernel: grad_W is never materialised in HBM. The CUTLASS Sm100
collective epilogue applies the entire AdamW update inline (m', v', W'
written via aux/D stores) while the fp32 grad_W tile is still in registers.

40% reduction in HBM traffic per layer vs the 2-kernel path
(cutlass_kernel_b200.py), at the cost of a much fatter epilogue. Whether
that translates to speedup depends on whether the shape is HBM-bound.
"""

import os
import shutil
import logging
import torch

log = logging.getLogger("cutlass_evt_b200")

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
    log.info(f"CUTLASS B200 EVT: building from {cutlass_root}")
    extra_flags = []
    # ITER8: optional override for Sm100 epilogue dispatch policy (CUTLASS patch in sm100_builder.inl)
    if os.environ.get("EVT_FORCE_DELAY_TMA_STORE"):
        extra_flags.append(f"-DCUTLASS_SM100_FORCE_DELAY_TMA_STORE={int(os.environ['EVT_FORCE_DELAY_TMA_STORE'])}")
    if os.environ.get("EVT_FORCE_REUSE_SMEM") is not None and os.environ.get("EVT_FORCE_REUSE_SMEM") != "":
        extra_flags.append(f"-DCUTLASS_SM100_FORCE_REUSE_SMEM={int(os.environ['EVT_FORCE_REUSE_SMEM'])}")
    # ITER16: scheduler arg overrides via compile-time macros.
    if os.environ.get("EVT_SCHED_SWIZZLE"):
        extra_flags.append(f"-DCUTLASS_EVT_SCHED_SWIZZLE={int(os.environ['EVT_SCHED_SWIZZLE'])}")
    if os.environ.get("EVT_SCHED_RASTER") == "M":
        extra_flags.append("-DCUTLASS_EVT_SCHED_RASTER_M=1")
    if os.environ.get("EVT_SCHED_RASTER") == "N":
        extra_flags.append("-DCUTLASS_EVT_SCHED_RASTER_N=1")
    # The kernel name carries any toggles so torch_extensions caches per-config.
    kernel_name = "fused_adamw_evt_sm100"
    suffix_bits = []
    for f in extra_flags:
        suffix_bits.append(f.replace("-D", "").replace("=", "").replace("CUTLASS_SM100_FORCE_", "f"))
    if suffix_bits:
        kernel_name = "fused_adamw_evt_sm100_" + "_".join(suffix_bits).lower()
    _module = load(
        name=kernel_name,
        sources=[os.path.join(csrc_dir, "fused_adamw_evt_sm100.cu")],
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
            *extra_flags,
        ],
        verbose=True,
    )
    return _module


_TILE_FN_MAP = {
    "128x128_1sm":              "blackwell_evt_adamw_128x128_1sm",
    "128x128_1sm_etile128x64":  "blackwell_evt_adamw_128x128_1sm",  # alias (EpilogueTile<_128,_64> is implicit in 128x128_1sm)
    "256x128_2sm":              "blackwell_evt_adamw_256x128_2sm",
    "256x256_2sm":              "blackwell_evt_adamw_256x256_2sm",
    "256x256_2sm_etile128x64":  "blackwell_evt_adamw_256x256_2sm_etile128x64",
    # ITER5+: epilogue-tile bracketing for amortizing AdamW compute.
    "256x256_2sm_etile128x128": "blackwell_evt_adamw_256x256_2sm_etile128x128",
    "256x256_2sm_etile64x64":   "blackwell_evt_adamw_256x256_2sm_etile64x64",
    "256x256_2sm_etile64x128":  "blackwell_evt_adamw_256x256_2sm_etile64x128",
    "256x128_2sm_etile128x128": "blackwell_evt_adamw_256x128_2sm_etile128x128",
    "256x128_2sm_etile128x64":  "blackwell_evt_adamw_256x128_2sm_etile128x64",
    "256x256_2sm_etile128x64_aux1": "blackwell_evt_adamw_256x256_2sm_etile128x64_aux1",
    "256x128_2sm_aux1":         "blackwell_evt_adamw_256x128_2sm_aux1",
    "256x256_2sm_etile128x64_c2x2": "blackwell_evt_adamw_256x256_2sm_etile128x64_c2x2",
    "256x256_2sm_etile64x64_c2x2":  "blackwell_evt_adamw_256x256_2sm_etile64x64_c2x2",
    "128x128_1sm_streamk":      "blackwell_evt_adamw_128x128_1sm_streamk",
    "256x256_2sm_streamk":      "blackwell_evt_adamw_256x256_2sm_streamk",
}

_NUM_SMS_B200 = 148


def _waves(M: int, N: int, tile_m: int, tile_n: int) -> float:
    tiles = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    return tiles / _NUM_SMS_B200


def _pick_tile(V: int, H: int) -> str:
    """Pick the EVT tile from output shape (M=V, N=H).

    T2 empirical microbench (pre-fix kernel, AuxStages=1 with SMEM-staged
    aux) found 256x128_2sm dominated 256x256_2sm by 15-26% across all
    Llama-3.1-8B shapes — the 256x256 tile lost mainloop stages because
    aux SMEM ate into the auto-carveout budget.

    Post-fix kernel (AuxStages=0, commit 4c36926) eliminates aux SMEM
    entirely → 256x256_2sm now has the same mainloop-stage headroom as
    256x128_2sm, AND its 2× per-atom FLOPs halve launch / dispatch
    overhead for the mlp + lm_head shapes (V or H ≥ 14336). For square
    qkvo at V=H=4096 we keep 256x128_2sm: tile count is small enough
    that wave-fill / fractional-tail effects favour the smaller tile.

    Llama-3.1-8B mapping:
      qkvo    V=4096   H=4096   →  256x128_2sm_etile128x64  (ITER6 keeper)
      gate/up V=14336  H=4096   →  256x256_2sm_etile128x64  (ITER4 keeper)
      down    V=4096   H=14336  →  256x256_2sm_etile128x64
      lm_head V=128256 H=4096   →  256x256_2sm_etile128x64

    Re-run the microbench after every kernel-template change.

    ITER5/6/8/9 results 2026-05-04:
      iter5  EpilogueTile<128,128> on big tile      → -30 ms regression (SMEM wall)
      iter6  EpilogueTile<128,64>  on qkvo (small)  → KEEPER, gap shrinks 9.8→4.8 ms
      iter7  AuxStages=1 with bf16-aux              → still corrupts m,v on tip CUTLASS
      iter8  DelayTmaStore=1 (CUTLASS macro patch)  → no help / slight regress
      iter9  ReuseSmem=0 (CUTLASS macro patch)      → see journey doc
    """
    # Env override for fast iteration without code edits. EVT_TILE_OVERRIDE
    # forces a single tile for ALL shapes; EVT_TILE_BIG / EVT_TILE_SMALL
    # split by V/H>=14336 (mlp/lm_head) vs qkvo.
    override = os.environ.get("EVT_TILE_OVERRIDE")
    if override:
        return override
    big_tile = os.environ.get("EVT_TILE_BIG")
    small_tile = os.environ.get("EVT_TILE_SMALL")
    # ITER14 shape-conditional override: if EVT_SHAPE_COND=1, for mid-BS cells
    # (BT in [4096, 16384]) use the iter13 combo (qkvo 256x128_2sm + mlp c2x2)
    # which won -2.9 ms at BS=2 SEQ=4096. Other cells keep iter2 keeper.
    if os.environ.get("EVT_SHAPE_COND") == "1":
        BT = _BT_HINT.get("BT", 0)
        # mid-BS band: 4k <= BT < 16k captures (BS=2, SEQ=2k..4k) etc.
        if 4096 <= BT < 16384:
            if V >= 14336 or H >= 14336:
                return "256x256_2sm_etile128x64_c2x2"
            return "256x128_2sm_etile128x64"
        # else fall through to default
    if os.environ.get("EVT_SHAPE_COND_V2") == "1":
        # iter15: narrower band — only BT >= 8192 (truly mid+ BS) uses combo.
        # BT=4096 (BS=1 SEQ=4096) is excluded because iter14 showed +0.9 there.
        BT = _BT_HINT.get("BT", 0)
        if BT >= 8192:
            if V >= 14336 or H >= 14336:
                return "256x256_2sm_etile128x64_c2x2"
            return "256x128_2sm_etile128x64"
        # else fall through to default
    if V >= 14336 or H >= 14336:
        return big_tile or "256x256_2sm_etile128x64"
    return small_tile or "256x128_2sm_etile128x64"


# Picker context, set by fused_grad_adamw_evt_b200 each call so _pick_tile
# can be BT-aware without changing its signature.
_BT_HINT = {"BT": 0}


def fused_grad_adamw_evt_b200(
    grad_output, input, weight, m, v,
    lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=1,
    tile=None,
):
    """Drop-in for fused_grad_adamw — but applies AdamW inside the GEMM
    epilogue instead of via a separate kernel."""
    assert grad_output.is_contiguous() and input.is_contiguous()
    V = grad_output.shape[1]
    H = input.shape[1]
    _BT_HINT["BT"] = grad_output.shape[0]
    if tile is None:
        tile = _pick_tile(V, H)

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    mod = _get_module()
    getattr(mod, _TILE_FN_MAP[tile])(
        grad_output, input, weight, m, v,
        lr, beta1, beta2, eps, weight_decay, bc1, bc2,
    )


def patch_dispatch():
    import fused_grad_optimizer.kernel as _k
    import fused_grad_optimizer.autograd as _a
    prev = (_k.fused_grad_adamw, _a.fused_grad_adamw)
    _k.fused_grad_adamw = fused_grad_adamw_evt_b200
    _a.fused_grad_adamw = fused_grad_adamw_evt_b200
    return prev


def restore_dispatch(prev):
    import fused_grad_optimizer.kernel as _k
    import fused_grad_optimizer.autograd as _a
    _k.fused_grad_adamw, _a.fused_grad_adamw = prev


def precompile():
    _get_module()
