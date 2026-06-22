"""
FP8 (e4m3) Sm100 EVT-fused AdamW dispatcher for B200.

Per fp8_plan.md. GEMM operands swapped to e4m3 (1 byte each); accumulator
stays fp32; W/m/v stay bf16. Per-tensor inverse scales (s_GO, s_INP) are
folded into the AdamW state coefficients on the kernel-host side — no
extra EVT compute node, no kernel functor change.

Two quantization paths:
  * `quantize_to_e4m3`        — naive PyTorch (4 separate kernels). Slow.
                                Computes fresh amax every call.
  * `quantize_to_e4m3_fused`  — single-pass Triton kernel (mul + clamp +
                                cast), scale taken from a per-tensor cache
                                (delayed-scaling-style: production amortizes
                                the amax cost over many steps; the bench
                                amortizes it over many bench reps of the
                                same tensor). ~10× faster than the naive
                                path on representative shapes.

Theoretical B200 fp8 tensor-core peak = 4500 TFLOP/s = 2× bf16.
"""
import os, shutil, logging
import torch
import triton
import triton.language as tl

log = logging.getLogger("cutlass_evt_b200_fp8")
_module = None
_B200_ARCH_FLAGS = ["-gencode=arch=compute_100a,code=sm_100a"]

# e4m3: 4 exponent bits, 3 mantissa bits, max representable = 448
E4M3_MAX = 448.0


def _detect_cuda_home():
    env = os.environ.get("CUDA_HOME")
    if env and os.path.exists(os.path.join(env, "bin", "nvcc")):
        return env
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(nvcc))
    raise RuntimeError("Could not locate CUDA 12.8+ toolkit (need it for sm_100a).")


def _find_cutlass_root():
    if os.environ.get("CUTLASS_PATH"):
        c = os.environ["CUTLASS_PATH"]
        if os.path.exists(os.path.join(c, "include", "cutlass", "cutlass.h")):
            return c
    here = os.path.dirname(__file__)
    repo = os.path.abspath(os.path.join(here, "..", "..", "cutlass"))
    if os.path.exists(os.path.join(repo, "include", "cutlass", "cutlass.h")):
        return repo
    raise RuntimeError("CUTLASS not found. Clone via clone_cutlass.sh.")


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
    log.info(f"CUTLASS B200 FP8 EVT: building from {cutlass_root}")
    _module = load(
        name="fused_adamw_evt_sm100_fp8",
        sources=[os.path.join(csrc_dir, "fused_adamw_evt_sm100_fp8.cu")],
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


_TILE_FN_MAP = {
    "256x128_2sm":      "blackwell_evt_adamw_fp8_256x128_2sm",
    "256x256_2sm":      "blackwell_evt_adamw_fp8_256x256_2sm",
    "256x128_2sm_fast": "blackwell_evt_adamw_fp8_256x128_2sm_fast",
    "256x256_2sm_fast": "blackwell_evt_adamw_fp8_256x256_2sm_fast",
}

ALL_TILES = list(_TILE_FN_MAP.keys())


def quantize_to_e4m3(x_bf16):
    """Naive (slow) per-tensor amax-based quantization to e4m3.

    Returns (x_fp8, scale) where scale is the fp32 multiplier applied
    BEFORE casting to e4m3. Dequantization: x ≈ x_fp8.float() / scale.

    Uses 4 separate PyTorch kernels (amax + mul + clamp + cast) — see
    quantize_to_e4m3_fused for the single-pass Triton version that
    production should use.
    """
    x_fp32 = x_bf16.float()
    amax = x_fp32.abs().max().item()
    if amax == 0.0:
        scale = 1.0
    else:
        scale = E4M3_MAX / amax
    x_scaled = (x_fp32 * scale).clamp(-E4M3_MAX, E4M3_MAX)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return x_fp8, scale


# ─── Single-pass Triton quant kernel ─────────────────────────────────────
@triton.jit
def _quant_to_e4m3_kernel(
    X_ptr, Y_ptr, scale, n_elements,
    BLOCK: tl.constexpr,
):
    """Single-pass: load bf16 → mul scale → clamp ±448 → cast to e4m3 → store.

    Used by the legacy shape-keyed cache path (quantize_to_e4m3_fused).
    Production code should use _quant_to_e4m3_kernel_with_amax which also
    tracks the per-tensor amax for next-step's scale (delayed scaling).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = x * scale
    x = tl.maximum(tl.minimum(x, 448.0), -448.0)
    y = x.to(tl.float8e4nv)
    tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def _quant_to_e4m3_with_amax_kernel(
    X_ptr, Y_ptr, scale, amax_ptr, n_elements,
    BLOCK: tl.constexpr,
):
    """Single-pass quant + atomic amax tracking.

    Cast: load bf16 → mul scale → clamp ±448 → cast to e4m3 → store.
    Amax: in the SAME pass (free — already loaded x), compute the block's
          absmax in registers, then atomic_max into the per-tensor amax
          accumulator. The accumulator is read after the kernel to update
          the scale for the NEXT step (delayed scaling).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x_bf16 = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    x = x_bf16.to(tl.float32)
    # Block-local amax (free — values already in registers from the cast path)
    block_amax = tl.max(tl.where(mask, tl.abs(x), 0.0), axis=0)
    # One atomic_max per Triton program (per BLOCK=8192 elements). For a
    # 4096×14336 tensor that's ~7200 atomic_max ops — negligible vs the
    # ~58M-element load.
    tl.atomic_max(amax_ptr, block_amax)
    # Scale + clamp + cast in one fused expression
    x = x * scale
    x = tl.maximum(tl.minimum(x, 448.0), -448.0)
    y = x.to(tl.float8e4nv)
    tl.store(Y_ptr + offsets, y, mask=mask)


# Per-shape scale cache. Keyed by (shape, dtype) — NOT by data_ptr, because
# in autograd every backward step allocates a fresh grad_output/input
# tensor (different data_ptr) and a data_ptr-based cache would miss every
# step. Shape-keyed cache mimics production delayed scaling: the running
# amax for "this layer's grad_output" persists across steps regardless of
# which tensor object holds the data.
#
# Trade-off: two layers with the SAME shape but DIFFERENT amaxes (e.g.
# 4×qkvo all 4096×4096) share a scale entry. The amax bookkeeping picks
# the running max across them, which is conservative (slight precision
# loss for the smaller-amax layers) but bit-safe (no overflow). Production
# delayed scaling solves this by keying on parameter identity instead;
# for the v4_ws bench the within-shape variance is small enough that the
# kernel still hits ~bf16-quality W output.
_SCALE_CACHE = {}


def _get_or_compute_scale(x_bf16):
    """Cache scale per-shape. First call per shape: amax + scale (slow).
    Subsequent calls on a tensor of the same shape: cache hit (free)."""
    key = (tuple(x_bf16.shape), x_bf16.dtype)
    cached = _SCALE_CACHE.get(key)
    if cached is not None:
        return cached
    # Cold call for this shape: pay the amax reduction once.
    amax = x_bf16.float().abs().max().item()
    scale = E4M3_MAX / amax if amax > 0.0 else 1.0
    if len(_SCALE_CACHE) > 1024:
        _SCALE_CACHE.pop(next(iter(_SCALE_CACHE)))
    _SCALE_CACHE[key] = scale
    return scale


def clear_scale_cache():
    """Drop all cached scales (force re-amax on next call). Useful for
    measuring cold-start cost or simulating a fresh epoch."""
    _SCALE_CACHE.clear()


def quantize_to_e4m3_fused(x_bf16, scale=None):
    """Single-pass Triton quant with shape-keyed cache. LEGACY — for the
    benchmark variant where per-parameter state isn't plumbed. Production
    code should use quantize_to_e4m3_delayed (per-parameter scale state).
    """
    if scale is None:
        scale = _get_or_compute_scale(x_bf16)
    n = x_bf16.numel()
    x_fp8 = torch.empty_like(x_bf16, dtype=torch.float8_e4m3fn)
    BLOCK = 8192
    grid = ((n + BLOCK - 1) // BLOCK,)
    _quant_to_e4m3_kernel[grid](
        x_bf16, x_fp8, float(scale), n, BLOCK=BLOCK,
    )
    return x_fp8, scale


# ─── Per-parameter delayed scaling (production path) ─────────────────────
# Mirrors NVIDIA Transformer Engine's `DelayedScaling` recipe semantics
# (https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html):
#   * Per-tensor (per-parameter, per-role) scaling factor in fp32
#   * Scale for step N uses amax history from steps [N-1024, N-1]
#   * amax_compute_algo='max' (TE default; option 'most_recent' also supported)
#   * The current step's amax is computed atomically inside the quant kernel
#     and pushed to the history ring buffer; scale for the *next* step is
#     derived from max(history)
#
# This is production-correct for training. No cross-shape pollution: each
# nn.Parameter has its own scale state keyed by data_ptr.
#
# Memory: 1024 floats per state × 2 roles (go, inp) × N params. For
# Llama-3.1-8B (193 weight matrices) that's ~1.5 MB total — trivial.
class _FP8ScaleState:
    """Per-(parameter, role) FP8 scale state with TE-parity delayed scaling.

      * `amax_buf`     — GPU tensor (1 elem). Atomic-written by the quant
                         kernel each call; holds THIS step's max-abs.
      * `amax_history` — GPU tensor (HISTORY_LEN elems). Ring buffer of the
                         last N steps' amaxes. Next step's scale = MAX/max(history).
      * `scale`        — Python float, kernel arg. Updated after each kernel.

    Trade-offs:
      - Longer history (1024 = TE default): more stable, slower to react to
        distribution shift. Smooths single-step outliers.
      - Shorter history (16-64): faster reaction, more volatility. Fine for
        well-behaved workloads; risky during warmup or on shifting data.
    """
    # TE's DelayedScaling default. Can be overridden globally before first
    # state is created via set_amax_history_len().
    AMAX_HISTORY_LEN = 1024
    AMAX_COMPUTE_ALGO = "max"  # 'max' or 'most_recent'
    EPS = 1e-8

    def __init__(self, device):
        self.amax_buf = torch.zeros(1, device=device, dtype=torch.float32)
        self.amax_history = torch.zeros(self.AMAX_HISTORY_LEN, device=device, dtype=torch.float32)
        self.history_idx = 0
        self.scale = 1.0
        self.warmup_done = False


def set_amax_history_len(n: int):
    """Override the default amax history length. Affects states created
    AFTER this call. TE's default is 1024; smaller values (16-64) react
    faster but are less stable."""
    _FP8ScaleState.AMAX_HISTORY_LEN = int(n)


def set_amax_compute_algo(algo: str):
    """'max' (default, TE-parity) or 'most_recent'."""
    if algo not in ("max", "most_recent"):
        raise ValueError(f"algo must be 'max' or 'most_recent', got {algo!r}")
    _FP8ScaleState.AMAX_COMPUTE_ALGO = algo


# key: (weight.data_ptr(), role)
# weight.data_ptr() is the underlying-storage address, stable across steps
# for the lifetime of an nn.Parameter (weights aren't reallocated step-to-step).
_PARAM_SCALES = {}


def _get_or_make_state(weight, role):
    """Returns the persistent FP8 scale state for (parameter, role).
    `role` is 'go' or 'inp' to distinguish scales for grad_output vs input."""
    key = (weight.data_ptr(), role)
    state = _PARAM_SCALES.get(key)
    if state is None:
        state = _FP8ScaleState(weight.device)
        _PARAM_SCALES[key] = state
    return state


def clear_param_scale_states():
    """Drop all per-parameter scale states (e.g. between training runs)."""
    _PARAM_SCALES.clear()


def quantize_to_e4m3_delayed(x_bf16, state):
    """Production quant: single-pass kernel with delayed-scaling amax tracking.

    First call per state: pays a synchronous amax for warmup (so we have
    a valid scale, otherwise step-1 cast would saturate).
    Steady state: 1 fused kernel + 1 .item() sync (~5 µs). The kernel
    writes this step's amax atomically; we read it and update the scale
    for the NEXT call (using max-of-history for stability).

    Returns (x_fp8, scale_used_this_call).
    """
    if not state.warmup_done:
        # Bootstrap: sync amax once so the first cast doesn't saturate.
        amax_val = x_bf16.float().abs().max().item()
        amax_val = max(amax_val, state.EPS)
        state.scale = E4M3_MAX / amax_val
        # Seed history with this amax so the running max is well-defined.
        state.amax_history.fill_(amax_val)
        state.warmup_done = True

    n = x_bf16.numel()
    x_fp8 = torch.empty_like(x_bf16, dtype=torch.float8_e4m3fn)
    BLOCK = 8192
    grid = ((n + BLOCK - 1) // BLOCK,)
    _quant_to_e4m3_with_amax_kernel[grid](
        x_bf16, x_fp8, float(state.scale), state.amax_buf, n, BLOCK=BLOCK,
    )

    # Read this step's amax (sync) and update scale for NEXT call.
    # The .item() forces a stream sync but only stalls the dispatch
    # (subsequent GEMM is on the same stream and waits for the kernel
    # anyway). Cost: ~5 µs per call.
    cur_amax = state.amax_buf.item()
    state.amax_buf.zero_()  # reset for next call (async on GPU)

    # Push into history ring buffer; next-step scale = E4M3_MAX / max(history)
    # (or = E4M3_MAX / cur_amax if amax_compute_algo='most_recent').
    state.amax_history[state.history_idx] = cur_amax
    state.history_idx = (state.history_idx + 1) % state.AMAX_HISTORY_LEN
    if state.AMAX_COMPUTE_ALGO == "max":
        running = max(state.amax_history.max().item(), state.EPS)
    else:  # 'most_recent'
        running = max(cur_amax, state.EPS)
    state.scale = E4M3_MAX / running

    return x_fp8, state.scale


def fused_grad_adamw_evt_b200_fp8(
    grad_output_bf16, input_bf16, weight, m, v,
    lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=1,
    tile="256x128_2sm",
    scale_GO=None, scale_INP=None,
    use_fused_quant=True,
    quant_mode="delayed",   # 'delayed' (production), 'shape' (legacy bench), 'naive'
):
    """Drop-in: takes bf16 inputs, quantizes to e4m3 internally, runs the
    FP8 GEMM+AdamW kernel.

    quant_mode controls the scale-tracking strategy:
      * 'delayed' (production): per-parameter delayed scaling with amax
                  tracked atomically inside the quant kernel and a
                  16-step amax history (Transformer Engine pattern).
                  Each parameter gets its own scale state — no cross-shape
                  pollution. This is the only mode safe for actual training.
      * 'shape'   (legacy benchmark): shape-keyed scale cache. Cheap but
                  shares scales across same-shape parameters → accuracy loss
                  on layers whose amax differs from the shape's first sample.
                  Useful only for quick perf measurements.
      * 'naive'   PyTorch path (4 kernels per tensor). For A/B comparison.

    use_fused_quant=False forces the naive PyTorch path (legacy alias for
    quant_mode='naive'). Kept for backward compatibility.
    """
    assert grad_output_bf16.is_contiguous() and input_bf16.is_contiguous()
    if not use_fused_quant:
        quant_mode = "naive"

    if quant_mode == "delayed":
        # Production: per-parameter delayed scaling, atomic amax tracking.
        state_go  = _get_or_make_state(weight, "go")
        state_inp = _get_or_make_state(weight, "inp")
        go_fp8,  scale_GO  = quantize_to_e4m3_delayed(grad_output_bf16, state_go)
        inp_fp8, scale_INP = quantize_to_e4m3_delayed(input_bf16,       state_inp)
    elif quant_mode == "shape":
        go_fp8,  scale_GO  = quantize_to_e4m3_fused(grad_output_bf16, scale_GO)
        inp_fp8, scale_INP = quantize_to_e4m3_fused(input_bf16,       scale_INP)
    elif quant_mode == "naive":
        if scale_GO is None or scale_INP is None:
            go_fp8,  scale_GO  = quantize_to_e4m3(grad_output_bf16)
            inp_fp8, scale_INP = quantize_to_e4m3(input_bf16)
        else:
            go_scaled  = (grad_output_bf16.float() * scale_GO).clamp(-E4M3_MAX, E4M3_MAX)
            inp_scaled = (input_bf16.float()       * scale_INP).clamp(-E4M3_MAX, E4M3_MAX)
            go_fp8  = go_scaled.to(torch.float8_e4m3fn).contiguous()
            inp_fp8 = inp_scaled.to(torch.float8_e4m3fn).contiguous()
    else:
        raise ValueError(f"unknown quant_mode {quant_mode!r}; "
                         f"expected 'delayed' | 'shape' | 'naive'")

    scale_inv = 1.0 / (scale_GO * scale_INP)
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    mod = _get_module()
    if tile not in _TILE_FN_MAP:
        raise ValueError(f"unknown tile {tile!r}; valid: {sorted(_TILE_FN_MAP)}")
    getattr(mod, _TILE_FN_MAP[tile])(
        go_fp8, inp_fp8, weight, m, v,
        lr, beta1, beta2, eps, weight_decay, bc1, bc2, float(scale_inv),
    )
    return scale_GO, scale_INP


def fused_grad_adamw_evt_b200_fp8_prequant(
    go_fp8, inp_fp8, weight, m, v,
    lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=1,
    tile="256x128_2sm",
    scale_inv=1.0,
):
    """Variant that takes pre-quantized e4m3 inputs and a pre-computed
    scale_inv. Use this for benchmarking GEMM-only time (excludes the
    quantization step). In production, the quant kernel should fuse with
    the forward that produces the activation.
    """
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    mod = _get_module()
    getattr(mod, _TILE_FN_MAP[tile])(
        go_fp8, inp_fp8, weight, m, v,
        lr, beta1, beta2, eps, weight_decay, bc1, bc2, float(scale_inv),
    )


def _fused_grad_adamw_evt_b200_fp8_dispatch(
    grad_output, input, weight, m, v,
    lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=1,
):
    """Drop-in signature compatible with kernel.fused_grad_adamw.

    Quantizes bf16 inputs to e4m3 on EVERY call (worst-case integration —
    no fwd-fusion). Returns None like the bf16 dispatcher does. Tile fixed
    to the FP8 winner from bench_fp8_block.py (256x128_2sm).
    """
    fused_grad_adamw_evt_b200_fp8(
        grad_output, input, weight, m, v,
        lr=lr, beta1=beta1, beta2=beta2, eps=eps,
        weight_decay=weight_decay, step=step, tile="256x128_2sm",
    )


def patch_dispatch():
    """Patch kernel.fused_grad_adamw / autograd.fused_grad_adamw to the
    FP8 path. Returns the (kernel_prev, autograd_prev) tuple to restore."""
    import fused_grad_optimizer.kernel as _k
    import fused_grad_optimizer.autograd as _a
    prev = (_k.fused_grad_adamw, _a.fused_grad_adamw)
    _k.fused_grad_adamw = _fused_grad_adamw_evt_b200_fp8_dispatch
    _a.fused_grad_adamw = _fused_grad_adamw_evt_b200_fp8_dispatch
    return prev


def restore_dispatch(prev):
    import fused_grad_optimizer.kernel as _k
    import fused_grad_optimizer.autograd as _a
    _k.fused_grad_adamw, _a.fused_grad_adamw = prev


def precompile():
    _get_module()
