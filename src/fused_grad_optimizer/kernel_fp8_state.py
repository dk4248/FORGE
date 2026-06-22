"""
FP8 per-tensor-scaled AdamW optimizer state — fused grad_W + AdamW with m,v in fp8.

Design (see docs/h200_optimization_journey/post_ws_analysis.md for context):

  * ncu on the bf16 WS kernel: long_scoreboard ≈ 45% of stalls across every
    Llama shape. Warps wait on DRAM loads of W, m, v. Bytes/load dominates.
  * Int8 state was tried with *per-64-col* block scales; the per-tile
    requantize epilogue tripped Triton's WS partitioner (silent NaN on 3.4,
    compile fail on 3.6).
  * FP8 per-tensor scale avoids the per-block requantize entirely: one scalar
    per whole tensor, loaded once at kernel start, divided out for dequant,
    multiplied back for quant. No masking. No per-row reductions.

Absmax tracking:
  Earlier version used tl.atomic_max(scalar_ptr, tile_max) inside the tile
  loop. Two problems: (1) ~32k tiles on lm_head hammering one address per
  kernel launch — severe atomic contention, and (2) the MLIR WS partitioner
  doesn't lower "atomic_max on fp32 scalar" inside a warp-specialized loop
  on Triton 3.4 ("PassManager::run failed" at make_ttgir).

  Current approach: per-tile absmax writes to a scratch[num_tiles] fp32
  array (plain TMA-free store, not atomic). A tiny post-pass reduce kernel
  folds the scratch into the next-step scalar absmax. This:
    - Eliminates atomic contention
    - Lets the WS partitioner lower the epilogue (no atomics inside WS)
    - Adds only a memory-bound ~1k-element reduce per parameter per step

Storage layout per-param (see FusedOptimizerState.ensure_buffers):
    m_fp8, v_fp8                (V, H)    fp8_e4m3fn / fp8_e5m2
    m_scale, v_scale            ()        fp32 scalar — this step's divisor
    m_absmax_next, v_absmax_next()        fp32 scalar — next step's seed
    m_absmax_scratch,
    v_absmax_scratch            (NT,)     fp32, per-tile max accumulator
                                          NT = ceil(V/128) * ceil(H/128)
"""

import torch
import triton
import triton.language as tl


# FP8 format finite limits — used to clamp before cast (Triton's .to(float8)
# is nearest-even but doesn't saturate on overflow; we do it explicitly).
FP8_E4M3_MAX = 448.0      # torch.float8_e4m3fn
FP8_E5M2_MAX = 57344.0    # torch.float8_e5m2

# Scratch-buffer sizing. We always allocate with the smallest BLOCK in the
# autotune config space, so every config's actual num_tiles ≤ max_num_tiles.
_SCRATCH_BLOCK_V = 128
_SCRATCH_BLOCK_H = 128


# ---------------------------------------------------------------------------
# Autotune configs — same winners as bf16 WS.
# ---------------------------------------------------------------------------

def _fp8_ws_configs():
    return [
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 32,
                       'GROUP_SIZE_V': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 32,
                       'GROUP_SIZE_V': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64,
                       'GROUP_SIZE_V': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64,
                       'GROUP_SIZE_V': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 128,
                       'GROUP_SIZE_V': 8}, num_warps=4, num_stages=3),
    ]


# ---------------------------------------------------------------------------
# Main fused kernel (WS-capable). Writes per-tile absmax to scratch, no atomics.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=_fp8_ws_configs(),
    key=['BT', 'V', 'H'],
    restore_value=['weight_ptr', 'm_ptr', 'v_ptr',
                   'm_absmax_scratch_ptr', 'v_absmax_scratch_ptr'],
)
@triton.jit
def _fused_grad_adamw_persistent_tma_ws_fp8state(
    grad_output_ptr,                    # (BT, V) bf16
    input_ptr,                          # (BT, H) bf16
    weight_ptr,                         # (V, H) bf16 — in-place
    m_ptr, v_ptr,                       # (V, H) fp8 — in-place
    m_scale_ptr, v_scale_ptr,           # () fp32 scalar — read-only here
    m_absmax_scratch_ptr,               # (NT,) fp32 — one tile-max per entry
    v_absmax_scratch_ptr,               # (NT,) fp32 — one tile-max per entry
    BT, V, H,
    go_stride_bt, go_stride_v,
    in_stride_bt, in_stride_h,
    w_stride_v, w_stride_h,
    m_stride_v, m_stride_h,
    lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2,
    FP8_E4M3_MAX: tl.constexpr,
    FP8_E5M2_MAX: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    GROUP_SIZE_V: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_h = tl.cdiv(H, BLOCK_H)
    num_tiles = num_pid_v * num_pid_h

    # Per-tensor scales: loaded ONCE, broadcast to whole grid.
    m_scale = tl.load(m_scale_ptr).to(tl.float32)
    v_scale = tl.load(v_scale_ptr).to(tl.float32)
    inv_m_scale = 1.0 / m_scale
    inv_v_scale = 1.0 / v_scale

    go_desc = tl.make_tensor_descriptor(
        grad_output_ptr, shape=[BT, V],
        strides=[go_stride_bt, go_stride_v],
        block_shape=[BLOCK_BT, BLOCK_V],
    )
    inp_desc = tl.make_tensor_descriptor(
        input_ptr, shape=[BT, H],
        strides=[in_stride_bt, in_stride_h],
        block_shape=[BLOCK_BT, BLOCK_H],
    )
    w_desc = tl.make_tensor_descriptor(
        weight_ptr, shape=[V, H], strides=[w_stride_v, w_stride_h],
        block_shape=[BLOCK_V, BLOCK_H],
    )
    # NOTE: m, v (fp8) use regular tl.load / tl.store (not TMA). Triton 3.4's
    # WS partitioner cannot route fp8 ops through TMA descriptors inside a
    # warp_specialized loop ("PassManager::run failed" at
    # nvgpu-warp-specialization). Only bf16 go/inp/w go through TMA.

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True):
        num_pid_in_group = GROUP_SIZE_V * num_pid_h
        group_id = tile_id // num_pid_in_group
        first_pid_v = group_id * GROUP_SIZE_V
        group_size = min(num_pid_v - first_pid_v, GROUP_SIZE_V)
        pid_v = first_pid_v + ((tile_id % num_pid_in_group) % group_size)
        pid_h = (tile_id % num_pid_in_group) // group_size

        off_v = pid_v * BLOCK_V
        off_h = pid_h * BLOCK_H

        # --- grad_W accumulation (fp32 registers) ---
        grad_acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
        for bt_start in range(0, BT, BLOCK_BT):
            go  = tl.trans(go_desc.load([bt_start, off_v]))
            inp = inp_desc.load([bt_start, off_h])
            grad_acc = tl.dot(go, inp, acc=grad_acc, out_dtype=tl.float32)

        # --- Load state ---
        w_raw  = w_desc.load([off_v, off_h])
        # Regular (non-TMA) fp8 loads for m, v. No mask — all Llama-3.1 linear
        # shapes (V ∈ {1024, 4096, 14336, 128256}, H ∈ {4096, 14336}) are
        # multiples of BLOCK_V = BLOCK_H = 128, so tiles never need clamping.
        rows = off_v + tl.arange(0, BLOCK_V)[:, None]
        cols = off_h + tl.arange(0, BLOCK_H)[None, :]
        addr_off = rows * m_stride_v + cols * m_stride_h
        m_fp8 = tl.load(m_ptr + addr_off)
        v_fp8 = tl.load(v_ptr + addr_off)

        w      = w_raw.to(tl.float32)
        m_prev = m_fp8.to(tl.float32) * m_scale
        v_prev = v_fp8.to(tl.float32) * v_scale

        # --- AdamW math in fp32 ---
        m      = beta1 * m_prev + (1.0 - beta1) * grad_acc
        v      = beta2 * v_prev + (1.0 - beta2) * grad_acc * grad_acc
        m_hat  = m / bias_correction1
        v_hat  = v / bias_correction2
        w      = w * (1.0 - lr * weight_decay)
        w      = w - lr * m_hat / (tl.sqrt(v_hat) + eps)

        # --- Per-tile absmax (NOT atomic — each tile owns its slot) ---
        m_tile_absmax = tl.max(tl.abs(m))
        v_tile_absmax = tl.max(tl.abs(v))
        tl.store(m_absmax_scratch_ptr + tile_id, m_tile_absmax)
        tl.store(v_absmax_scratch_ptr + tile_id, v_tile_absmax)

        # --- Quantize with THIS step's scale (clamp to fp8 finite range) ---
        m_scaled = m * inv_m_scale
        v_scaled = v * inv_v_scale
        m_scaled = tl.minimum(tl.maximum(m_scaled, -FP8_E4M3_MAX), FP8_E4M3_MAX)
        v_scaled = tl.minimum(tl.maximum(v_scaled, -FP8_E5M2_MAX), FP8_E5M2_MAX)

        # --- Stores ---
        # W goes through TMA (bf16 store works fine under WS).
        w_desc.store([off_v, off_h], w.to(w_raw.dtype))
        # m, v use regular tl.store (same reason as load, above). No mask —
        # shapes are always block-aligned on Llama-3.1.
        tl.store(m_ptr + addr_off, m_scaled.to(tl.float8e4nv))
        tl.store(v_ptr + addr_off, v_scaled.to(tl.float8e5))


# ---------------------------------------------------------------------------
# Non-WS fallback. Same kernel, warp_specialize=False. Kept so we can A/B
# and can fall back if the WS partitioner regresses.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=_fp8_ws_configs(),
    key=['BT', 'V', 'H'],
    restore_value=['weight_ptr', 'm_ptr', 'v_ptr',
                   'm_absmax_scratch_ptr', 'v_absmax_scratch_ptr'],
)
@triton.jit
def _fused_grad_adamw_persistent_tma_fp8state_nows(
    grad_output_ptr, input_ptr, weight_ptr,
    m_ptr, v_ptr,
    m_scale_ptr, v_scale_ptr,
    m_absmax_scratch_ptr, v_absmax_scratch_ptr,
    BT, V, H,
    go_stride_bt, go_stride_v,
    in_stride_bt, in_stride_h,
    w_stride_v, w_stride_h,
    m_stride_v, m_stride_h,
    lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2,
    FP8_E4M3_MAX: tl.constexpr,
    FP8_E5M2_MAX: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    GROUP_SIZE_V: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_h = tl.cdiv(H, BLOCK_H)
    num_tiles = num_pid_v * num_pid_h

    m_scale = tl.load(m_scale_ptr).to(tl.float32)
    v_scale = tl.load(v_scale_ptr).to(tl.float32)
    inv_m_scale = 1.0 / m_scale
    inv_v_scale = 1.0 / v_scale

    go_desc = tl.make_tensor_descriptor(grad_output_ptr, shape=[BT, V],
        strides=[go_stride_bt, go_stride_v], block_shape=[BLOCK_BT, BLOCK_V])
    inp_desc = tl.make_tensor_descriptor(input_ptr, shape=[BT, H],
        strides=[in_stride_bt, in_stride_h], block_shape=[BLOCK_BT, BLOCK_H])
    w_desc = tl.make_tensor_descriptor(weight_ptr, shape=[V, H],
        strides=[w_stride_v, w_stride_h], block_shape=[BLOCK_V, BLOCK_H])
    m_desc = tl.make_tensor_descriptor(m_ptr, shape=[V, H],
        strides=[m_stride_v, m_stride_h], block_shape=[BLOCK_V, BLOCK_H])
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[V, H],
        strides=[m_stride_v, m_stride_h], block_shape=[BLOCK_V, BLOCK_H])

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_V * num_pid_h
        group_id = tile_id // num_pid_in_group
        first_pid_v = group_id * GROUP_SIZE_V
        group_size = min(num_pid_v - first_pid_v, GROUP_SIZE_V)
        pid_v = first_pid_v + ((tile_id % num_pid_in_group) % group_size)
        pid_h = (tile_id % num_pid_in_group) // group_size
        off_v = pid_v * BLOCK_V
        off_h = pid_h * BLOCK_H

        grad_acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
        for bt_start in range(0, BT, BLOCK_BT):
            go  = tl.trans(go_desc.load([bt_start, off_v]))
            inp = inp_desc.load([bt_start, off_h])
            grad_acc = tl.dot(go, inp, acc=grad_acc, out_dtype=tl.float32)

        w_raw  = w_desc.load([off_v, off_h])
        m_fp8  = m_desc.load([off_v, off_h])
        v_fp8  = v_desc.load([off_v, off_h])

        w      = w_raw.to(tl.float32)
        m_prev = m_fp8.to(tl.float32) * m_scale
        v_prev = v_fp8.to(tl.float32) * v_scale

        m      = beta1 * m_prev + (1.0 - beta1) * grad_acc
        v      = beta2 * v_prev + (1.0 - beta2) * grad_acc * grad_acc
        m_hat  = m / bias_correction1
        v_hat  = v / bias_correction2
        w      = w * (1.0 - lr * weight_decay)
        w      = w - lr * m_hat / (tl.sqrt(v_hat) + eps)

        m_tile_absmax = tl.max(tl.abs(m))
        v_tile_absmax = tl.max(tl.abs(v))
        tl.store(m_absmax_scratch_ptr + tile_id, m_tile_absmax)
        tl.store(v_absmax_scratch_ptr + tile_id, v_tile_absmax)

        m_scaled = m * inv_m_scale
        v_scaled = v * inv_v_scale
        m_scaled = tl.minimum(tl.maximum(m_scaled, -FP8_E4M3_MAX), FP8_E4M3_MAX)
        v_scaled = tl.minimum(tl.maximum(v_scaled, -FP8_E5M2_MAX), FP8_E5M2_MAX)

        w_desc.store([off_v, off_h], w.to(w_raw.dtype))
        m_desc.store([off_v, off_h], m_scaled.to(tl.float8e4nv))
        v_desc.store([off_v, off_h], v_scaled.to(tl.float8e5))


# ---------------------------------------------------------------------------
# Tiny post-pass reduce: max over scratch → absmax_next scalar.
# Single block, BLOCK_SIZE=1024 covers up to 1024 tiles per scan; we loop if
# scratch is bigger. Memory-bound, ~µs per call at Llama shapes.
# ---------------------------------------------------------------------------

@triton.jit
def _absmax_scratch_reduce(
    scratch_ptr,                      # (NT,) fp32
    absmax_next_ptr,                  # () fp32 scalar
    NT,                               # number of valid entries in scratch
    BLOCK_SIZE: tl.constexpr,
):
    # Only one program — grid=(1,).
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, NT, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < NT
        chunk = tl.load(scratch_ptr + offs, mask=mask, other=0.0)
        acc = tl.maximum(acc, chunk)
    val = tl.max(acc)
    tl.store(absmax_next_ptr, val)


# ---------------------------------------------------------------------------
# Python-side wrappers
# ---------------------------------------------------------------------------

def max_scratch_tiles(V: int, H: int) -> int:
    """Max number of tiles any autotune config can produce for (V, H)."""
    from triton import cdiv
    return cdiv(V, _SCRATCH_BLOCK_V) * cdiv(H, _SCRATCH_BLOCK_H)


# WS compile status on Triton 3.4 for fp8: BLOCKED. Three variants tested:
#   (a) fp8 TMA load + fp8 TMA store + atomic_max   → PassManager::run failed
#   (b) same but scratch-array instead of atomic     → PassManager::run failed
#   (c) regular tl.load/store for fp8 (no TMA fp8)   → PassManager::run failed
# In all cases NVGPUWarpSpecialization refuses to partition. The non-WS
# path (persistent + TMA for bf16, TMA for fp8 loads/stores) compiles and
# runs bit-clean. Flip to True to retry once Triton teaches the
# partitioner how to route fp8 ops across producer/consumer warps.
_fp8_use_ws = False


def fused_grad_adamw_fp8state(
    grad_output: torch.Tensor,   # (BT, V) bf16
    input: torch.Tensor,         # (BT, H) bf16
    weight: torch.Tensor,        # (V, H) bf16 — in-place
    state,                       # FusedOptimizerState with state_mode="fp8"
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    step: int = 1,
):
    from fused_grad_optimizer.kernel import _ensure_tma_allocator, _get_num_sms

    assert grad_output.is_contiguous() and input.is_contiguous()
    BT, V = grad_output.shape
    H = input.shape[1]
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    _ensure_tma_allocator()
    num_sms = _get_num_sms(weight.device)

    kernel = (_fused_grad_adamw_persistent_tma_ws_fp8state
              if _fp8_use_ws else
              _fused_grad_adamw_persistent_tma_fp8state_nows)

    kernel[(num_sms,)](
        grad_output, input, weight,
        state.m_fp8, state.v_fp8,
        state.m_scale, state.v_scale,
        state.m_absmax_scratch, state.v_absmax_scratch,
        BT, V, H,
        grad_output.stride(0), grad_output.stride(1),
        input.stride(0),       input.stride(1),
        weight.stride(0),      weight.stride(1),
        state.m_fp8.stride(0), state.m_fp8.stride(1),
        lr, beta1, beta2, eps, weight_decay, bc1, bc2,
        FP8_E4M3_MAX=FP8_E4M3_MAX,
        FP8_E5M2_MAX=FP8_E5M2_MAX,
        NUM_SMS=num_sms,
    )

    # Post-pass reduce: fold per-tile scratch into the absmax_next scalars.
    NT = state.m_absmax_scratch.numel()
    BLOCK_SIZE = 1024
    _absmax_scratch_reduce[(1,)](state.m_absmax_scratch, state.m_absmax_next,
                                  NT, BLOCK_SIZE=BLOCK_SIZE)
    _absmax_scratch_reduce[(1,)](state.v_absmax_scratch, state.v_absmax_next,
                                  NT, BLOCK_SIZE=BLOCK_SIZE)
