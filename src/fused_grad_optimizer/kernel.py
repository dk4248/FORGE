"""
Core Triton kernels: fused weight-gradient computation + optimizer update.

These kernels compute grad_W = grad_output.T @ input tile-by-tile and apply
the optimizer (SGD or AdamW) immediately per tile. The full grad_W tensor
is NEVER allocated in HBM.

== Optimizations ==

Level 1: Autotuned tile sizes (64x64 to 256x256) with BLOCK_BT exploration.
         Triton's autotune benchmarks each config for actual (BT, V, H).
Tile swizzle: Grouped tile ordering for L2 cache reuse — tiles that share
              the input tensor execute together.
bf16 tl.dot: Inputs stay in native dtype for tensor cores; accumulator is fp32.

Read amplification for LLaMA 8B lm_head (V=128256, H=4096):
  64x64   → 2004 × 64  = 128,256 tiles
  128x256 → 1003 × 16  =  16,048 tiles  (8x reduction)
  256x256 → 502  × 16  =   8,032 tiles  (16x reduction)
"""

import torch
import triton
import triton.language as tl

# TMA (Tensor Memory Accelerator) requires a scratch memory allocator.
# Set lazily because CUDA may not be initialized at import time.
_tma_allocator_set = False

def _ensure_tma_allocator():
    """Set up TMA scratch memory allocator (once, lazily)."""
    global _tma_allocator_set
    if _tma_allocator_set:
        return
    def _alloc(size, align, stream):
        return torch.empty(size, dtype=torch.uint8, device='cuda').data_ptr()
    triton.set_allocator(_alloc)
    _tma_allocator_set = True


# ---------------------------------------------------------------------------
# Autotune configurations
# ---------------------------------------------------------------------------

def _fused_configs():
    """Autotune configs for the fused grad+optimizer kernels.

    Key tuning axes:
    - BLOCK_V x BLOCK_H: tile size — larger = fewer tiles = less read amplification
    - BLOCK_BT: inner loop chunk — larger = fewer iterations = better MMA utilization
    - GROUP_SIZE_V: L2 reuse group — larger = more tiles share activation data in L2
    - num_stages: pipeline depth — more = better latency hiding (limited by SMEM)
    """
    return [
        # --- BLOCK_BT=32 configs ---
        # Small: 64x64, low register pressure
        triton.Config({'BLOCK_V': 64,  'BLOCK_H': 64,  'BLOCK_BT': 32, 'GROUP_SIZE_V': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_V': 64,  'BLOCK_H': 64,  'BLOCK_BT': 32, 'GROUP_SIZE_V': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_V': 64,  'BLOCK_H': 64,  'BLOCK_BT': 32, 'GROUP_SIZE_V': 8}, num_warps=4, num_stages=3),
        # Medium: 128x128
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 32, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 32, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=3),
        # Large rectangular: 128x256 and 256x128
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 256, 'BLOCK_BT': 32, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 256, 'BLOCK_BT': 32, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_V': 256, 'BLOCK_H': 128, 'BLOCK_BT': 32, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_V': 256, 'BLOCK_H': 128, 'BLOCK_BT': 32, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=3),
        # XL: 256x256 (highest register pressure, may have low occupancy)
        triton.Config({'BLOCK_V': 256, 'BLOCK_H': 256, 'BLOCK_BT': 32, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=2),
        # --- BLOCK_BT=64 configs (fewer loop iterations, better MMA utilization) ---
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 256, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_V': 256, 'BLOCK_H': 128, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=2),
        # --- Deeper pipeline for small tiles (16KB/stage at 64x64, fits 4 stages in SMEM) ---
        triton.Config({'BLOCK_V': 64,  'BLOCK_H': 64,  'BLOCK_BT': 64, 'GROUP_SIZE_V': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_V': 64,  'BLOCK_H': 64,  'BLOCK_BT': 64, 'GROUP_SIZE_V': 8}, num_warps=4, num_stages=4),
        # 128x128 with 3 stages (32KB/stage x 3 = 96KB, fits in SMEM)
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=3),
        # Larger GROUP_SIZE_V=16 for better L2 reuse across tiles
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 256, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 16}, num_warps=8, num_stages=2),
        # Tall-skinny tiles for V >> H layers (lm_head)
        triton.Config({'BLOCK_V': 64,  'BLOCK_H': 256, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_V': 64,  'BLOCK_H': 128, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 8}, num_warps=4, num_stages=3),
        # BLOCK_BT=128 (only 4 inner loop iterations at BT=512)
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 64,  'BLOCK_BT': 128,'GROUP_SIZE_V': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_V': 64,  'BLOCK_H': 64,  'BLOCK_BT': 128,'GROUP_SIZE_V': 8}, num_warps=4, num_stages=2),
        # --- Blackwell-targeted: large tiles + deep pipeline + large L2 groups ---
        # Down_proj (V=4096, H=14336) needs large BLOCK_H to cut tile count.
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 256, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 256, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_V': 256, 'BLOCK_H': 128, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_V': 256, 'BLOCK_H': 256, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_V': 256, 'BLOCK_H': 256, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 8}, num_warps=8, num_stages=3),
        # Gate/up_proj (V=14336, H=4096): wide V → big BLOCK_V helps.
        triton.Config({'BLOCK_V': 256, 'BLOCK_H': 128, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_V': 256, 'BLOCK_H': 64,  'BLOCK_BT': 64, 'GROUP_SIZE_V': 16}, num_warps=8, num_stages=3),
        # 128 MB L2 on Blackwell → use larger GROUP_SIZE_V for more reuse.
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 256, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_V': 256, 'BLOCK_H': 128, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 32}, num_warps=8, num_stages=3),
        # BLOCK_BT=128 with big tiles — only 4 BT iterations at BT=512.
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 128,'GROUP_SIZE_V': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 256, 'BLOCK_BT': 128,'GROUP_SIZE_V': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_V': 256, 'BLOCK_H': 128, 'BLOCK_BT': 128,'GROUP_SIZE_V': 8}, num_warps=8, num_stages=2),
    ]


def _optimizer_only_configs():
    """Autotune configs for the optimizer-only kernel (no matmul)."""
    return [
        triton.Config({'BLOCK_R': 64,  'BLOCK_C': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_R': 128, 'BLOCK_C': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_R': 128, 'BLOCK_C': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_R': 256, 'BLOCK_C': 256}, num_warps=8, num_stages=1),
    ]


# ---------------------------------------------------------------------------
# SGD kernel — simplest case, useful for testing and benchmarking
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=_fused_configs(), key=['BT', 'V', 'H'],
    restore_value=['weight_ptr'],  # in-place: restore between autotune trials
)
@triton.jit
def _fused_grad_sgd_kernel(
    # Pointers
    grad_output_ptr,   # (BT, V)
    input_ptr,         # (BT, H)
    weight_ptr,        # (V, H) — updated in-place
    # Dimensions
    BT, V, H,
    # Strides (element counts, not bytes)
    go_stride_bt, go_stride_v,
    in_stride_bt, in_stride_h,
    w_stride_v,  w_stride_h,
    # Optimizer
    lr, weight_decay,
    # Tile sizes (provided by autotune)
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    GROUP_SIZE_V: tl.constexpr,
):
    """
    Each program instance owns one (BLOCK_V, BLOCK_H) tile of the weight matrix.
    It accumulates the gradient for that tile by looping over the BT dimension,
    then applies SGD: W = W * (1 - lr*wd) - lr * grad.
    """
    # Grouped tile ordering for L2 cache reuse
    pid = tl.program_id(0)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_h = tl.cdiv(H, BLOCK_H)
    num_pid_in_group = GROUP_SIZE_V * num_pid_h
    group_id = pid // num_pid_in_group
    first_pid_v = group_id * GROUP_SIZE_V
    group_size = min(num_pid_v - first_pid_v, GROUP_SIZE_V)
    pid_v = first_pid_v + ((pid % num_pid_in_group) % group_size)
    pid_h = (pid % num_pid_in_group) // group_size

    v_off = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    v_mask = v_off < V
    h_mask = h_off < H

    # Gradient accumulator — lives in registers (fp32 for precision)
    grad_acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)

    # Inner loop: accumulate grad_output[:, v].T @ input[:, h]
    # grad_output is loaded in transposed layout via make_block_ptr order=(0,1)
    # so tl.dot sees (BLOCK_V, BLOCK_BT) @ (BLOCK_BT, BLOCK_H) directly.
    # Inputs stay in native dtype (bf16) for tensor core throughput.
    for bt_start in range(0, BT, BLOCK_BT):
        go_block_ptr = tl.make_block_ptr(
            base=grad_output_ptr,
            shape=(V, BT),
            strides=(go_stride_v, go_stride_bt),
            offsets=(pid_v * BLOCK_V, bt_start),
            block_shape=(BLOCK_V, BLOCK_BT),
            order=(0, 1),
        )
        inp_block_ptr = tl.make_block_ptr(
            base=input_ptr,
            shape=(BT, H),
            strides=(in_stride_bt, in_stride_h),
            offsets=(bt_start, pid_h * BLOCK_H),
            block_shape=(BLOCK_BT, BLOCK_H),
            order=(1, 0),
        )
        go = tl.load(go_block_ptr, boundary_check=(0, 1))
        inp = tl.load(inp_block_ptr, boundary_check=(0, 1))

        # (BLOCK_V, BLOCK_BT) @ (BLOCK_BT, BLOCK_H) -> (BLOCK_V, BLOCK_H)
        grad_acc += tl.dot(go, inp, out_dtype=tl.float32)

    # Load weight tile, apply SGD, store
    # evict_first: single-use per step, don't pollute L2 for activation reuse
    w_ptrs = weight_ptr + v_off[:, None] * w_stride_v + h_off[None, :] * w_stride_h
    w_mask = v_mask[:, None] & h_mask[None, :]
    w = tl.load(w_ptrs, mask=w_mask, other=0.0, eviction_policy="evict_first")

    w_f32 = w.to(tl.float32)
    w_f32 = w_f32 * (1.0 - lr * weight_decay) - lr * grad_acc
    tl.store(w_ptrs, w_f32.to(w.dtype), mask=w_mask, eviction_policy="evict_first")


# ---------------------------------------------------------------------------
# AdamW kernel — the practically important one
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=_fused_configs(), key=['BT', 'V', 'H'],
    restore_value=['weight_ptr', 'm_ptr', 'v_ptr'],  # in-place: restore between autotune trials
)
@triton.jit
def _fused_grad_adamw_kernel(
    # Pointers
    grad_output_ptr,   # (BT, V)
    input_ptr,         # (BT, H)
    weight_ptr,        # (V, H) — updated in-place
    m_ptr,             # (V, H) — first moment, updated in-place
    v_ptr,             # (V, H) — second moment, updated in-place
    # Dimensions
    BT, V, H,
    # Strides
    go_stride_bt, go_stride_v,
    in_stride_bt, in_stride_h,
    w_stride_v,  w_stride_h,
    # Optimizer hyperparams
    lr, beta1, beta2, eps, weight_decay,
    bias_correction1,   # 1 - beta1^step (precomputed in Python)
    bias_correction2,   # 1 - beta2^step
    # Tile sizes (provided by autotune)
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    GROUP_SIZE_V: tl.constexpr,
):
    """
    Each program instance owns one (BLOCK_V, BLOCK_H) tile.
    Steps:
      1. Accumulate gradient tile via matmul loop over BT
      2. Load m, v for this tile
      3. Update m = beta1*m + (1-beta1)*grad
      4. Update v = beta2*v + (1-beta2)*grad^2
      5. AdamW: w = w*(1-lr*wd) - lr * m_hat / (sqrt(v_hat) + eps)
      6. Store w, m, v
    """
    # Grouped tile ordering for L2 cache reuse
    pid = tl.program_id(0)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_h = tl.cdiv(H, BLOCK_H)
    num_pid_in_group = GROUP_SIZE_V * num_pid_h
    group_id = pid // num_pid_in_group
    first_pid_v = group_id * GROUP_SIZE_V
    group_size = min(num_pid_v - first_pid_v, GROUP_SIZE_V)
    pid_v = first_pid_v + ((pid % num_pid_in_group) % group_size)
    pid_h = (pid % num_pid_in_group) // group_size

    v_off = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    v_mask = v_off < V
    h_mask = h_off < H
    tile_mask = v_mask[:, None] & h_mask[None, :]

    # --- Step 1: gradient accumulation in registers ---
    grad_acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)

    for bt_start in range(0, BT, BLOCK_BT):
        go_block_ptr = tl.make_block_ptr(
            base=grad_output_ptr,
            shape=(V, BT),
            strides=(go_stride_v, go_stride_bt),
            offsets=(pid_v * BLOCK_V, bt_start),
            block_shape=(BLOCK_V, BLOCK_BT),
            order=(0, 1),
        )
        inp_block_ptr = tl.make_block_ptr(
            base=input_ptr,
            shape=(BT, H),
            strides=(in_stride_bt, in_stride_h),
            offsets=(bt_start, pid_h * BLOCK_H),
            block_shape=(BLOCK_BT, BLOCK_H),
            order=(1, 0),
        )
        go = tl.load(go_block_ptr, boundary_check=(0, 1))
        inp = tl.load(inp_block_ptr, boundary_check=(0, 1))

        grad_acc += tl.dot(go, inp, out_dtype=tl.float32)

    # --- Step 2-6: load state, update, store ---
    # evict_first: w/m/v are single-use per step, don't pollute L2
    w_offsets = v_off[:, None] * w_stride_v + h_off[None, :] * w_stride_h

    w_raw = tl.load(weight_ptr + w_offsets, mask=tile_mask, other=0.0, eviction_policy="evict_first")
    m_raw = tl.load(m_ptr + w_offsets, mask=tile_mask, other=0.0, eviction_policy="evict_first")
    v_raw = tl.load(v_ptr + w_offsets, mask=tile_mask, other=0.0, eviction_policy="evict_first")

    w = w_raw.to(tl.float32)

    # Moment updates (scalar promotion ensures fp32 arithmetic)
    m = beta1 * m_raw + (1.0 - beta1) * grad_acc
    v = beta2 * v_raw + (1.0 - beta2) * grad_acc * grad_acc

    # Bias-corrected estimates
    m_hat = m / bias_correction1
    v_hat = v / bias_correction2

    # AdamW: decoupled weight decay then Adam step
    w = w * (1.0 - lr * weight_decay)
    w = w - lr * m_hat / (tl.sqrt(v_hat) + eps)

    # Store back — evict_first to avoid polluting cache with single-use data
    tl.store(weight_ptr + w_offsets, w.to(w_raw.dtype), mask=tile_mask, eviction_policy="evict_first")
    tl.store(m_ptr + w_offsets, m.to(m_raw.dtype), mask=tile_mask, eviction_policy="evict_first")
    tl.store(v_ptr + w_offsets, v.to(v_raw.dtype), mask=tile_mask, eviction_policy="evict_first")


# ---------------------------------------------------------------------------
# AdamW kernel with INT8 quantized optimizer states (m, v)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=_fused_configs(), key=['BT', 'V', 'H'],
    restore_value=['weight_ptr', 'm_q_ptr', 'v_q_ptr', 'm_scale_ptr', 'v_scale_ptr'],
)
@triton.jit
def _fused_grad_adamw_int8state_kernel(
    # Pointers
    grad_output_ptr,   # (BT, V)
    input_ptr,         # (BT, H)
    weight_ptr,        # (V, H) — updated in-place
    m_q_ptr,           # (V, H) int8 — first moment, quantized
    v_q_ptr,           # (V, H) int8 — second moment, quantized
    m_scale_ptr,       # (V, H // QBLOCK) fp32 — m absmax scales
    v_scale_ptr,       # (V, H // QBLOCK) fp32 — v absmax scales
    # Dimensions
    BT, V, H,
    # Strides
    go_stride_bt, go_stride_v,
    in_stride_bt, in_stride_h,
    w_stride_v,  w_stride_h,
    mq_stride_v, mq_stride_h,
    ms_stride_v, ms_stride_h,
    # Optimizer hyperparams
    lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2,
    # Tile sizes
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    GROUP_SIZE_V: tl.constexpr,
    QBLOCK: tl.constexpr,
):
    """
    Fused grad+AdamW with int8 block-wise quantized optimizer states.
    Same gradient accumulation as the non-quantized kernel. The difference:
      - m, v are stored as int8 + fp32 absmax scales (per QBLOCK columns)
      - Dequantize m, v on load; requantize on store
      - Weight is still stored/updated in its original dtype (bf16)
    """
    # Grouped tile ordering for L2 cache reuse
    pid = tl.program_id(0)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_h = tl.cdiv(H, BLOCK_H)
    num_pid_in_group = GROUP_SIZE_V * num_pid_h
    group_id = pid // num_pid_in_group
    first_pid_v = group_id * GROUP_SIZE_V
    group_size = min(num_pid_v - first_pid_v, GROUP_SIZE_V)
    pid_v = first_pid_v + ((pid % num_pid_in_group) % group_size)
    pid_h = (pid % num_pid_in_group) // group_size

    v_off = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    v_mask = v_off < V
    h_mask = h_off < H
    tile_mask = v_mask[:, None] & h_mask[None, :]

    # --- Step 1: gradient accumulation in registers ---
    grad_acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)

    for bt_start in range(0, BT, BLOCK_BT):
        go_block_ptr = tl.make_block_ptr(
            base=grad_output_ptr,
            shape=(V, BT),
            strides=(go_stride_v, go_stride_bt),
            offsets=(pid_v * BLOCK_V, bt_start),
            block_shape=(BLOCK_V, BLOCK_BT),
            order=(0, 1),
        )
        inp_block_ptr = tl.make_block_ptr(
            base=input_ptr,
            shape=(BT, H),
            strides=(in_stride_bt, in_stride_h),
            offsets=(bt_start, pid_h * BLOCK_H),
            block_shape=(BLOCK_BT, BLOCK_H),
            order=(1, 0),
        )
        go = tl.load(go_block_ptr, boundary_check=(0, 1))
        inp = tl.load(inp_block_ptr, boundary_check=(0, 1))

        grad_acc += tl.dot(go, inp, out_dtype=tl.float32)

    # --- Step 2: load weight (evict_first: single-use per step) ---
    w_offsets = v_off[:, None] * w_stride_v + h_off[None, :] * w_stride_h
    w_raw = tl.load(weight_ptr + w_offsets, mask=tile_mask, other=0.0, eviction_policy="evict_first")
    w = w_raw.to(tl.float32)

    # --- Step 3: dequantize m and v from int8 (evict_first: single-use) ---
    mq_offsets = v_off[:, None] * mq_stride_v + h_off[None, :] * mq_stride_h
    h_local = tl.arange(0, BLOCK_H)
    BPT: tl.constexpr = BLOCK_H // QBLOCK
    scale_col = pid_h * BPT + h_local // QBLOCK  # per-element scale column

    m_int8 = tl.load(m_q_ptr + mq_offsets, mask=tile_mask, other=0, eviction_policy="evict_first")
    v_int8 = tl.load(v_q_ptr + mq_offsets, mask=tile_mask, other=0, eviction_policy="evict_first")

    sc_ptrs_m = m_scale_ptr + v_off[:, None] * ms_stride_v + scale_col[None, :] * ms_stride_h
    sc_ptrs_v = v_scale_ptr + v_off[:, None] * ms_stride_v + scale_col[None, :] * ms_stride_h
    m_scales = tl.load(sc_ptrs_m, mask=tile_mask, other=1.0, eviction_policy="evict_first")
    v_scales = tl.load(sc_ptrs_v, mask=tile_mask, other=1.0, eviction_policy="evict_first")

    m = m_int8.to(tl.float32) * m_scales
    # v (second moment) is always non-negative; int8 round-trip can map small
    # values to 0, which makes sqrt(v_hat)≈0 and blows up the AdamW step.
    # Floor at 1e-8 so the denominator stays sane after quantization.
    v = tl.maximum(v_int8.to(tl.float32) * v_scales, 1e-8)

    # --- Step 4: AdamW update (identical math) ---
    m = beta1 * m + (1.0 - beta1) * grad_acc
    v = beta2 * v + (1.0 - beta2) * grad_acc * grad_acc

    m_hat = m / bias_correction1
    v_hat = v / bias_correction2

    w = w * (1.0 - lr * weight_decay)
    w = w - lr * m_hat / (tl.sqrt(v_hat) + eps)

    # --- Step 5: store weight (evict_first: don't pollute cache) ---
    tl.store(weight_ptr + w_offsets, w.to(w_raw.dtype), mask=tile_mask, eviction_policy="evict_first")

    # --- Step 6: requantize and store m ---
    # Optimized: tl.extra.cuda.libdevice.rint for 1-op rounding (vs 5 ops manual)
    m_abs = tl.abs(m)
    for qb in tl.static_range(BLOCK_H // QBLOCK):
        qb_mask = (h_local >= qb * QBLOCK) & (h_local < (qb + 1) * QBLOCK)
        m_abs_qb = tl.where(qb_mask[None, :], m_abs, 0.0)
        qb_absmax = tl.max(m_abs_qb, axis=1)
        qb_scale = qb_absmax / 127.0 + 1e-12

        sc = pid_h * BPT + qb
        tl.store(m_scale_ptr + v_off * ms_stride_v + sc * ms_stride_h,
                 qb_scale, mask=v_mask)

        m_div = tl.where(qb_mask[None, :], m / qb_scale[:, None], 0.0)
        m_q_val = tl.extra.cuda.libdevice.rint(m_div).to(tl.int32)
        m_q_val = tl.minimum(tl.maximum(m_q_val, -127), 127)
        tl.store(m_q_ptr + mq_offsets, m_q_val.to(tl.int8),
                 mask=tile_mask & qb_mask[None, :])

    # --- Step 7: requantize and store v ---
    v_abs = tl.abs(v)
    for qb in tl.static_range(BLOCK_H // QBLOCK):
        qb_mask = (h_local >= qb * QBLOCK) & (h_local < (qb + 1) * QBLOCK)
        v_abs_qb = tl.where(qb_mask[None, :], v_abs, 0.0)
        qb_absmax = tl.max(v_abs_qb, axis=1)
        qb_scale = qb_absmax / 127.0 + 1e-12

        sc = pid_h * BPT + qb
        tl.store(v_scale_ptr + v_off * ms_stride_v + sc * ms_stride_h,
                 qb_scale, mask=v_mask)

        v_div = tl.where(qb_mask[None, :], v / qb_scale[:, None], 0.0)
        v_q_val = tl.extra.cuda.libdevice.rint(v_div).to(tl.int32)
        v_q_val = tl.minimum(tl.maximum(v_q_val, -127), 127)
        tl.store(v_q_ptr + mq_offsets, v_q_val.to(tl.int8),
                 mask=tile_mask & qb_mask[None, :])


# ---------------------------------------------------------------------------
# Optimizer-only AdamW kernel with INT8 quantized states (for accumulation)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=_optimizer_only_configs(), key=['rows', 'cols'],
    restore_value=['weight_ptr', 'm_q_ptr', 'v_q_ptr', 'm_scale_ptr', 'v_scale_ptr'],
)
@triton.jit
def _optimizer_only_adamw_int8state_kernel(
    grad_ptr, weight_ptr,
    m_q_ptr, v_q_ptr, m_scale_ptr, v_scale_ptr,
    rows, cols,
    g_stride_r, g_stride_c,
    w_stride_r, w_stride_c,
    mq_stride_r, mq_stride_c,
    ms_stride_r, ms_stride_c,
    lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
    QBLOCK: tl.constexpr,
):
    """Apply AdamW to a pre-computed gradient with int8 quantized m/v states."""
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    r_off = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    r_mask = r_off < rows
    mask = r_mask[:, None] & (c_off[None, :] < cols)

    offsets = r_off[:, None] * w_stride_r + c_off[None, :] * w_stride_c
    g_offsets = r_off[:, None] * g_stride_r + c_off[None, :] * g_stride_c
    mq_offsets = r_off[:, None] * mq_stride_r + c_off[None, :] * mq_stride_c

    g = tl.load(grad_ptr + g_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)

    # Dequantize m, v
    c_local = tl.arange(0, BLOCK_C)
    BPC: tl.constexpr = BLOCK_C // QBLOCK
    scale_col = pid_c * BPC + c_local // QBLOCK

    m_int8 = tl.load(m_q_ptr + mq_offsets, mask=mask, other=0)
    v_int8 = tl.load(v_q_ptr + mq_offsets, mask=mask, other=0)

    sc_ptrs_m = m_scale_ptr + r_off[:, None] * ms_stride_r + scale_col[None, :] * ms_stride_c
    sc_ptrs_v = v_scale_ptr + r_off[:, None] * ms_stride_r + scale_col[None, :] * ms_stride_c
    m_scales = tl.load(sc_ptrs_m, mask=mask, other=1.0)
    v_scales = tl.load(sc_ptrs_v, mask=mask, other=1.0)

    m = m_int8.to(tl.float32) * m_scales
    v = tl.maximum(v_int8.to(tl.float32) * v_scales, 1e-8)  # floor: prevent div-by-zero from quantized zeros

    # AdamW update
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g

    m_hat = m / bias_correction1
    v_hat = v / bias_correction2

    w_f32 = w.to(tl.float32)
    w_f32 = w_f32 * (1.0 - lr * weight_decay)
    w_f32 = w_f32 - lr * m_hat / (tl.sqrt(v_hat) + eps)

    tl.store(weight_ptr + offsets, w_f32.to(w.dtype), mask=mask)

    # Requantize m (optimized: tl.extra.cuda.libdevice.rint for 1-op rounding)
    m_abs = tl.abs(m)
    for qb in tl.static_range(BLOCK_C // QBLOCK):
        qb_mask = (c_local >= qb * QBLOCK) & (c_local < (qb + 1) * QBLOCK)
        m_abs_qb = tl.where(qb_mask[None, :], m_abs, 0.0)
        qb_absmax = tl.max(m_abs_qb, axis=1)
        qb_scale = qb_absmax / 127.0 + 1e-12

        sc = pid_c * BPC + qb
        tl.store(m_scale_ptr + r_off * ms_stride_r + sc * ms_stride_c,
                 qb_scale, mask=r_mask)

        m_div = tl.where(qb_mask[None, :], m / qb_scale[:, None], 0.0)
        m_q_val = tl.extra.cuda.libdevice.rint(m_div).to(tl.int32)
        m_q_val = tl.minimum(tl.maximum(m_q_val, -127), 127)
        tl.store(m_q_ptr + mq_offsets, m_q_val.to(tl.int8),
                 mask=mask & qb_mask[None, :])

    # Requantize v (optimized: tl.extra.cuda.libdevice.rint for 1-op rounding)
    v_abs = tl.abs(v)
    for qb in tl.static_range(BLOCK_C // QBLOCK):
        qb_mask = (c_local >= qb * QBLOCK) & (c_local < (qb + 1) * QBLOCK)
        v_abs_qb = tl.where(qb_mask[None, :], v_abs, 0.0)
        qb_absmax = tl.max(v_abs_qb, axis=1)
        qb_scale = qb_absmax / 127.0 + 1e-12

        sc = pid_c * BPC + qb
        tl.store(v_scale_ptr + r_off * ms_stride_r + sc * ms_stride_c,
                 qb_scale, mask=r_mask)

        v_div = tl.where(qb_mask[None, :], v / qb_scale[:, None], 0.0)
        v_q_val = tl.extra.cuda.libdevice.rint(v_div).to(tl.int32)
        v_q_val = tl.minimum(tl.maximum(v_q_val, -127), 127)
        tl.store(v_q_ptr + mq_offsets, v_q_val.to(tl.int8),
                 mask=mask & qb_mask[None, :])


# ---------------------------------------------------------------------------
# Optimizer-only AdamW kernel (for pre-computed gradients / accumulation)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=_optimizer_only_configs(), key=['rows', 'cols'],
    restore_value=['weight_ptr', 'm_ptr', 'v_ptr'],
)
@triton.jit
def _optimizer_only_adamw_kernel(
    grad_ptr, weight_ptr, m_ptr, v_ptr,
    rows, cols,
    g_stride_r, g_stride_c,
    w_stride_r, w_stride_c,
    lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Apply AdamW to a pre-computed gradient tensor. No matmul fusion."""
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    r_off = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (r_off[:, None] < rows) & (c_off[None, :] < cols)

    offsets = r_off[:, None] * w_stride_r + c_off[None, :] * w_stride_c
    g_offsets = r_off[:, None] * g_stride_r + c_off[None, :] * g_stride_c

    g = tl.load(grad_ptr + g_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    m_orig = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    v_orig = tl.load(v_ptr + offsets, mask=mask, other=0.0)

    m = beta1 * m_orig + (1.0 - beta1) * g
    v = beta2 * v_orig + (1.0 - beta2) * g * g

    m_hat = m / bias_correction1
    v_hat = v / bias_correction2

    w_f32 = w.to(tl.float32)
    w_f32 = w_f32 * (1.0 - lr * weight_decay)
    w_f32 = w_f32 - lr * m_hat / (tl.sqrt(v_hat) + eps)

    tl.store(weight_ptr + offsets, w_f32.to(w.dtype), mask=mask)
    tl.store(m_ptr + offsets, m.to(m_orig.dtype), mask=mask)
    tl.store(v_ptr + offsets, v.to(v_orig.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Persistent kernels — one CTA per SM, looping over tiles
# ---------------------------------------------------------------------------
# These avoid wave quantization and improve L2 reuse across tiles.
# Fixed tile sizes (no autotune) since autotune + persistent is too slow.

@triton.jit
def _fused_grad_adamw_persistent(
    # Pointers
    grad_output_ptr, input_ptr, weight_ptr, m_ptr, v_ptr,
    # Dimensions
    BT, V, H,
    # Strides
    go_stride_bt, go_stride_v,
    in_stride_bt, in_stride_h,
    w_stride_v, w_stride_h,
    # Optimizer hyperparams
    lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2,
    # Constexprs
    NUM_SMS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    GROUP_SIZE_V: tl.constexpr,
):
    """Persistent fused grad+AdamW: one CTA per SM, looping over weight tiles."""
    start_pid = tl.program_id(0)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_h = tl.cdiv(H, BLOCK_H)
    num_tiles = num_pid_v * num_pid_h

    # flatten=False so num_stages only pipelines the INNER BT loop
    # (prefetch next grad_output/input chunk while computing current tl.dot).
    # With flatten=True, Triton would also pipeline across tiles, requiring
    # SMEM buffers for W/m/v (~96 KB per stage) — exceeding the 99 KB limit.
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        # Grouped tile decomposition for L2 cache reuse
        num_pid_in_group = GROUP_SIZE_V * num_pid_h
        group_id = tile_id // num_pid_in_group
        first_pid_v = group_id * GROUP_SIZE_V
        group_size = min(num_pid_v - first_pid_v, GROUP_SIZE_V)
        pid_v = first_pid_v + ((tile_id % num_pid_in_group) % group_size)
        pid_h = (tile_id % num_pid_in_group) // group_size

        v_off = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        v_mask = v_off < V
        h_mask = h_off < H
        tile_mask = v_mask[:, None] & h_mask[None, :]

        # --- Gradient accumulation ---
        grad_acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
        for bt_start in range(0, BT, BLOCK_BT):
            go_block_ptr = tl.make_block_ptr(
                base=grad_output_ptr, shape=(V, BT),
                strides=(go_stride_v, go_stride_bt),
                offsets=(pid_v * BLOCK_V, bt_start),
                block_shape=(BLOCK_V, BLOCK_BT), order=(0, 1),
            )
            inp_block_ptr = tl.make_block_ptr(
                base=input_ptr, shape=(BT, H),
                strides=(in_stride_bt, in_stride_h),
                offsets=(bt_start, pid_h * BLOCK_H),
                block_shape=(BLOCK_BT, BLOCK_H), order=(1, 0),
            )
            go = tl.load(go_block_ptr, boundary_check=(0, 1))
            inp = tl.load(inp_block_ptr, boundary_check=(0, 1))
            grad_acc += tl.dot(go, inp, out_dtype=tl.float32)

        # --- Load state, update, store ---
        # evict_first: W/m/v are single-use per step — don't pollute L2 cache.
        # This is CRITICAL: without evict_first, W/m/v data fills L2 and evicts
        # the grad_output/input tiles that GROUP_SIZE_V needs for reuse across
        # V-tiles. This was confirmed to cause a 12ms regression.
        w_offsets = v_off[:, None] * w_stride_v + h_off[None, :] * w_stride_h
        w_raw = tl.load(weight_ptr + w_offsets, mask=tile_mask, other=0.0, eviction_policy="evict_first")
        m_raw = tl.load(m_ptr + w_offsets, mask=tile_mask, other=0.0, eviction_policy="evict_first")
        v_raw = tl.load(v_ptr + w_offsets, mask=tile_mask, other=0.0, eviction_policy="evict_first")

        w = w_raw.to(tl.float32)
        m = beta1 * m_raw + (1.0 - beta1) * grad_acc
        v = beta2 * v_raw + (1.0 - beta2) * grad_acc * grad_acc
        m_hat = m / bias_correction1
        v_hat = v / bias_correction2
        w = w * (1.0 - lr * weight_decay)
        w = w - lr * m_hat / (tl.sqrt(v_hat) + eps)

        tl.store(weight_ptr + w_offsets, w.to(w_raw.dtype), mask=tile_mask, eviction_policy="evict_first")
        tl.store(m_ptr + w_offsets, m.to(m_raw.dtype), mask=tile_mask, eviction_policy="evict_first")
        tl.store(v_ptr + w_offsets, v.to(v_raw.dtype), mask=tile_mask, eviction_policy="evict_first")


# ---------------------------------------------------------------------------
# TMA persistent kernel — uses hardware copy engine for ALL loads/stores
# ---------------------------------------------------------------------------
# On Blackwell (sm_120), TMA (Tensor Memory Accelerator) replaces per-thread
# loads with a single hardware instruction that copies entire 2D tiles.
#
# Key differences from the regular persistent kernel:
#   1. Descriptors created ONCE, reused for every tile (less instruction overhead)
#   2. W/m/v use TMA instead of per-thread pointer arithmetic (main win)
#   3. grad_output loaded as (BLOCK_BT, BLOCK_V) then transposed for matmul
#   4. No masks needed — TMA handles boundaries via padding_option='zero'

@triton.jit
def _fused_grad_adamw_persistent_tma(
    # Pointers
    grad_output_ptr, input_ptr, weight_ptr, m_ptr, v_ptr,
    # Dimensions
    BT, V, H,
    # Strides (element counts)
    go_stride_bt, go_stride_v,
    in_stride_bt, in_stride_h,
    w_stride_v, w_stride_h,
    # Optimizer hyperparams
    lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2,
    # Constexprs
    NUM_SMS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    GROUP_SIZE_V: tl.constexpr,
):
    """Persistent fused grad+AdamW using TMA for all memory operations."""
    start_pid = tl.program_id(0)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_h = tl.cdiv(H, BLOCK_H)
    num_tiles = num_pid_v * num_pid_h

    # ── Create TMA descriptors (once per kernel launch) ──
    #
    # Each descriptor captures: base pointer, tensor shape, strides, tile shape.
    # The hardware copy engine uses these to compute addresses and transfer
    # entire 2D tiles in one instruction.
    #
    # grad_output is (BT, V) in memory, contiguous (last dim stride=1).
    # We load (BLOCK_BT, BLOCK_V) tiles and transpose to (BLOCK_V, BLOCK_BT)
    # because the matmul needs: (BLOCK_V, BLOCK_BT) @ (BLOCK_BT, BLOCK_H).
    go_desc = tl.make_tensor_descriptor(
        grad_output_ptr,
        shape=[BT, V],
        strides=[go_stride_bt, go_stride_v],   # [V, 1] for contiguous
        block_shape=[BLOCK_BT, BLOCK_V],
    )

    # input is (BT, H) contiguous — loads directly as (BLOCK_BT, BLOCK_H)
    inp_desc = tl.make_tensor_descriptor(
        input_ptr,
        shape=[BT, H],
        strides=[in_stride_bt, in_stride_h],   # [H, 1] for contiguous
        block_shape=[BLOCK_BT, BLOCK_H],
    )

    # W, m, v are all (V, H) contiguous — same shape/strides, different base ptrs
    w_desc = tl.make_tensor_descriptor(
        weight_ptr, shape=[V, H], strides=[w_stride_v, w_stride_h],
        block_shape=[BLOCK_V, BLOCK_H],
    )
    m_desc = tl.make_tensor_descriptor(
        m_ptr, shape=[V, H], strides=[w_stride_v, w_stride_h],
        block_shape=[BLOCK_V, BLOCK_H],
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[V, H], strides=[w_stride_v, w_stride_h],
        block_shape=[BLOCK_V, BLOCK_H],
    )

    # ── Persistent tile loop ──
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        # Grouped tile ordering for L2 cache reuse
        num_pid_in_group = GROUP_SIZE_V * num_pid_h
        group_id = tile_id // num_pid_in_group
        first_pid_v = group_id * GROUP_SIZE_V
        group_size = min(num_pid_v - first_pid_v, GROUP_SIZE_V)
        pid_v = first_pid_v + ((tile_id % num_pid_in_group) % group_size)
        pid_h = (tile_id % num_pid_in_group) // group_size

        off_v = pid_v * BLOCK_V
        off_h = pid_h * BLOCK_H

        # --- Gradient accumulation via TMA ---
        # grad_output tile: load as (BLOCK_BT, BLOCK_V) then transpose to
        #   (BLOCK_V, BLOCK_BT) for the matmul. Transposition in Triton is a
        #   register-level permute — essentially free.
        # input tile: (BLOCK_BT, BLOCK_H) — correct shape, no transpose.
        grad_acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
        for bt_start in range(0, BT, BLOCK_BT):
            go  = tl.trans(go_desc.load([bt_start, off_v]))
            inp = inp_desc.load([bt_start, off_h])
            # (BLOCK_V, BLOCK_BT) @ (BLOCK_BT, BLOCK_H) → (BLOCK_V, BLOCK_H)
            grad_acc = tl.dot(go, inp, acc=grad_acc, out_dtype=tl.float32)

        # --- TMA load W, m, v (one instruction each, not per-thread loads) ---
        w_raw = w_desc.load([off_v, off_h])
        m_raw = m_desc.load([off_v, off_h])
        v_raw = v_desc.load([off_v, off_h])

        # --- AdamW update (identical math, all in fp32 registers) ---
        w = w_raw.to(tl.float32)
        m = beta1 * m_raw + (1.0 - beta1) * grad_acc
        v = beta2 * v_raw + (1.0 - beta2) * grad_acc * grad_acc
        m_hat = m / bias_correction1
        v_hat = v / bias_correction2
        w = w * (1.0 - lr * weight_decay)
        w = w - lr * m_hat / (tl.sqrt(v_hat) + eps)

        # --- TMA store updated W, m, v ---
        w_desc.store([off_v, off_h], w.to(w_raw.dtype))
        m_desc.store([off_v, off_h], m.to(m_raw.dtype))
        v_desc.store([off_v, off_h], v.to(v_raw.dtype))


# ---------------------------------------------------------------------------
# TMA persistent + auto-warp-specialization + @triton.autotune (Hopper)
# ---------------------------------------------------------------------------
# Same body as _fused_grad_adamw_persistent_tma, but:
#   1. Outer tile loop has `warp_specialize=True` so Triton's auto-WS pass
#      partitions warps into producer (TMA-issue) and consumer (WGMMA +
#      epilogue) roles.
#   2. @triton.autotune picks BLOCK_BT / GROUP_SIZE_V / num_stages per
#      (BT, V, H). BLOCK_V × BLOCK_H is locked at 128×128 — num_warps=4
#      (required by Hopper WS) caps the fp32 accumulator at 128×128
#      (32768 fp32 / 128 threads = 256 regs/thread, right at the 255 limit);
#      larger tiles register-spill.
#
# Hopper gates:
#   * Must launch at num_warps=4 — Triton's Hopper lowering silently drops WS
#     at num_warps=8 (upstream test_warp_specialization.py:435).
#   * All loads must be TMA (we already do this).
#   * Needs Triton>=3.4.
#
# Bit-exact vs the non-WS TMA kernel (validated in
# benchmarks/benchmark_ws_vs_tma.py).  Measured on H200 / Llama 8B /
# SEQ=4096: bwd 349 ms (no WS) → 327 ms (WS fixed tile) → 301 ms (WS +
# autotune, this path), i.e. -14% over plain TMA and -9% over fixed-tile WS.
# Beats the CUTLASS EVT-Sm90 path (332 ms) and is the only path that brings
# total step time below the cuBLAS+fused-AdamW baseline on H200.

def _ws_tma_configs():
    """Autotune configs for the TMA+WS persistent kernel.

    Measured winners on H200 / Llama 8B / SEQ=4096:
        lm_head  (V=128256)   BT= 64  GV= 8  stages=4
        gate/up  (V= 14336)   BT=128  GV= 8  stages=3
        down_proj(H= 14336)   BT= 64  GV= 8  stages=4
        q/o_proj (V=  4096)   BT=128  GV= 8  stages=3
        k/v_proj (V=  1024)   BT=128  GV= 8  stages=3
    """
    return [
        # BLOCK_BT=32 (fine reduction)
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 32,  'GROUP_SIZE_V': 8 },
                      num_warps=4, num_stages=3),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 32,  'GROUP_SIZE_V': 8 },
                      num_warps=4, num_stages=4),
        # BLOCK_BT=64 (default-ish)
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64,  'GROUP_SIZE_V': 8 },
                      num_warps=4, num_stages=2),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64,  'GROUP_SIZE_V': 8 },
                      num_warps=4, num_stages=3),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64,  'GROUP_SIZE_V': 8 },
                      num_warps=4, num_stages=4),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64,  'GROUP_SIZE_V': 16},
                      num_warps=4, num_stages=3),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64,  'GROUP_SIZE_V': 32},
                      num_warps=4, num_stages=3),
        # BLOCK_BT=128 (fewer inner iters, good on mid-sized layers)
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 128, 'GROUP_SIZE_V': 8 },
                      num_warps=4, num_stages=2),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 128, 'GROUP_SIZE_V': 8 },
                      num_warps=4, num_stages=3),
    ]
    # Note: deeper num_stages (5, 6) and BLOCK_BT=256 were tested via
    # benchmarks/ab_ws_configs.py — autotune rejected all of them (no new
    # winners on any Llama-3.1-8B shape). Reason: `num_stages` controls
    # software pipelining in plain Triton, but warp_specialize=True replaces
    # that with a hardware producer/consumer pipeline whose depth is fixed
    # by the WS infrastructure. Adding stages here is mostly a no-op. The
    # long_scoreboard stall (≈45% of stall samples) is not a `num_stages`
    # knob — it needs either fewer bytes per load (int8 state) or a
    # structural change to the WS pipeline.


@triton.autotune(
    configs=_ws_tma_configs(),
    key=['BT', 'V', 'H'],
    restore_value=['weight_ptr', 'm_ptr', 'v_ptr'],  # in-place
)
@triton.jit
def _fused_grad_adamw_persistent_tma_ws(
    grad_output_ptr, input_ptr, weight_ptr, m_ptr, v_ptr,
    BT, V, H,
    go_stride_bt, go_stride_v,
    in_stride_bt, in_stride_h,
    w_stride_v, w_stride_h,
    lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2,
    NUM_SMS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    GROUP_SIZE_V: tl.constexpr,
):
    """Persistent TMA fused grad+AdamW with auto-warp-specialization."""
    start_pid = tl.program_id(0)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_h = tl.cdiv(H, BLOCK_H)
    num_tiles = num_pid_v * num_pid_h

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
    m_desc = tl.make_tensor_descriptor(
        m_ptr, shape=[V, H], strides=[w_stride_v, w_stride_h],
        block_shape=[BLOCK_V, BLOCK_H],
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[V, H], strides=[w_stride_v, w_stride_h],
        block_shape=[BLOCK_V, BLOCK_H],
    )

    # warp_specialize=True lets the compiler split warps into producer
    # (TMA-issue) and consumer (WGMMA + epilogue) roles automatically.
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True):
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

        w_raw = w_desc.load([off_v, off_h])
        m_raw = m_desc.load([off_v, off_h])
        v_raw = v_desc.load([off_v, off_h])

        w = w_raw.to(tl.float32)
        m = beta1 * m_raw + (1.0 - beta1) * grad_acc
        v = beta2 * v_raw + (1.0 - beta2) * grad_acc * grad_acc
        m_hat = m / bias_correction1
        v_hat = v / bias_correction2
        w = w * (1.0 - lr * weight_decay)
        w = w - lr * m_hat / (tl.sqrt(v_hat) + eps)

        w_desc.store([off_v, off_h], w.to(w_raw.dtype))
        m_desc.store([off_v, off_h], m.to(m_raw.dtype))
        v_desc.store([off_v, off_h], v.to(v_raw.dtype))


# ---------------------------------------------------------------------------
# Persistent + TMA + WS kernel with INT8 quantized optimizer state (m, v)
# ---------------------------------------------------------------------------
# ⚠ NOT WIRED IN THE DISPATCHER. Kept here as documented future work.
#
# Reason: Triton's warp_specialize=True pass mis-partitions this kernel's
# int8 requantize epilogue across producer/consumer warps. Tested Triton
# 3.4 and 3.6 — both produce silent NaN on a handful of tiles (5–9 tiles
# out of ~1000, always 2 rows × 128 cols per affected tile). With WS=False
# the kernel is correct but ~30% SLOWER than the plain non-persistent int8
# kernel (persistent launch without WS serializes what was parallel).
#
# Tried patterns that did NOT work:
#   - masked-loop requantize + WS=True on Triton 3.4 → NaN
#   - masked-loop requantize + WS=True on Triton 3.6 → NaN (different tiles)
#   - 3D-reshape requantize + WS=True on Triton 3.6 → compile error
#     ("NVGPUWarpSpecialization: unsupported op type" — WS partitioner
#      can't lower tl.reshape(2D↔3D) in an epilogue)
#
# When Triton fixes WS codegen for this pattern, re-wire the dispatcher:
# search for "NOT WIRED" in fused_grad_adamw_int8state below.
#
# Same WS pipeline as _fused_grad_adamw_persistent_tma_ws, but m and v are
# stored as int8 + fp32 per-QBLOCK-col absmax scales (matching
# _fused_grad_adamw_int8state_kernel). Scales are tiny (BLOCK_V * BPT * 4B
# per tile) so they bypass TMA; big tensors go through TMA descriptors.
@triton.autotune(
    configs=_ws_tma_configs(),
    key=['BT', 'V', 'H'],
    restore_value=['weight_ptr', 'm_q_ptr', 'v_q_ptr',
                   'm_scale_ptr', 'v_scale_ptr'],
)
@triton.jit
def _fused_grad_adamw_persistent_tma_ws_int8state(
    grad_output_ptr, input_ptr, weight_ptr,
    m_q_ptr, v_q_ptr, m_scale_ptr, v_scale_ptr,
    BT, V, H,
    go_stride_bt, go_stride_v,
    in_stride_bt, in_stride_h,
    w_stride_v, w_stride_h,
    mq_stride_v, mq_stride_h,
    ms_stride_v, ms_stride_h,
    lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2,
    NUM_SMS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    GROUP_SIZE_V: tl.constexpr,
    QBLOCK: tl.constexpr,
):
    """Persistent TMA WS fused grad+AdamW with int8 quantized m/v state."""
    start_pid = tl.program_id(0)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_h = tl.cdiv(H, BLOCK_H)
    num_tiles = num_pid_v * num_pid_h
    BPT: tl.constexpr = BLOCK_H // QBLOCK

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
    m_q_desc = tl.make_tensor_descriptor(
        m_q_ptr, shape=[V, H], strides=[mq_stride_v, mq_stride_h],
        block_shape=[BLOCK_V, BLOCK_H],
    )
    v_q_desc = tl.make_tensor_descriptor(
        v_q_ptr, shape=[V, H], strides=[mq_stride_v, mq_stride_h],
        block_shape=[BLOCK_V, BLOCK_H],
    )

    # WS is on (per Triton ≥3.6 which fixes the "2 NaN rows per tile"
    # partition-assignment bug we saw on 3.4). On older Triton this
    # regresses to scattered NaNs — upgrade before re-enabling if needed.
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, warp_specialize=True):
        num_pid_in_group = GROUP_SIZE_V * num_pid_h
        group_id = tile_id // num_pid_in_group
        first_pid_v = group_id * GROUP_SIZE_V
        group_size = min(num_pid_v - first_pid_v, GROUP_SIZE_V)
        pid_v = first_pid_v + ((tile_id % num_pid_in_group) % group_size)
        pid_h = (tile_id % num_pid_in_group) // group_size

        off_v = pid_v * BLOCK_V
        off_h = pid_h * BLOCK_H

        # grad_W accumulation (fp32 registers)
        grad_acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
        for bt_start in range(0, BT, BLOCK_BT):
            go  = tl.trans(go_desc.load([bt_start, off_v]))
            inp = inp_desc.load([bt_start, off_h])
            grad_acc = tl.dot(go, inp, acc=grad_acc, out_dtype=tl.float32)

        # Load weight + int8 m/v via TMA; scales via pointer arithmetic.
        w_raw  = w_desc.load([off_v, off_h])
        m_int8 = m_q_desc.load([off_v, off_h])
        v_int8 = v_q_desc.load([off_v, off_h])

        v_off = off_v + tl.arange(0, BLOCK_V)
        h_local = tl.arange(0, BLOCK_H)
        # Each element in [0, BLOCK_H) maps to scale column h_local // QBLOCK
        # within this tile's BPT-wide scale window.
        scale_col = pid_h * BPT + h_local // QBLOCK
        v_mask = v_off < V
        h_mask = (off_h + h_local) < H
        tile_mask = v_mask[:, None] & h_mask[None, :]
        scale_offs = v_off[:, None] * ms_stride_v + scale_col[None, :] * ms_stride_h
        m_scales = tl.load(m_scale_ptr + scale_offs, mask=tile_mask,
                           other=1.0, eviction_policy="evict_first")
        v_scales = tl.load(v_scale_ptr + scale_offs, mask=tile_mask,
                           other=1.0, eviction_policy="evict_first")

        # Dequantize. v gets a 1e-8 floor so sqrt(v_hat) can't underflow
        # after an int8 round-trip mapped a tiny v to 0.
        m = m_int8.to(tl.float32) * m_scales
        v = tl.maximum(v_int8.to(tl.float32) * v_scales, 1e-8)

        # AdamW update (same math as bf16 kernel)
        w = w_raw.to(tl.float32)
        m = beta1 * m + (1.0 - beta1) * grad_acc
        v = beta2 * v + (1.0 - beta2) * grad_acc * grad_acc
        m_hat = m / bias_correction1
        v_hat = v / bias_correction2
        w = w * (1.0 - lr * weight_decay)
        w = w - lr * m_hat / (tl.sqrt(v_hat) + eps)

        # Store weight via TMA
        w_desc.store([off_v, off_h], w.to(w_raw.dtype))

        # Requantize m/v per QBLOCK columns. NOTE: the cleaner 3D-reshape
        # pattern hits `NVGPUWarpSpecialization: unsupported op type` on
        # Triton 3.6 — the WS partitioner can't lower tl.reshape(2D↔3D)
        # in an epilogue. The static-loop + tl.where pattern stays inside
        # shape classes the partitioner understands.
        m_abs = tl.abs(m)
        v_abs = tl.abs(v)
        m_q_out = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.int8)
        v_q_out = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.int8)

        for qb in tl.static_range(BPT):
            qb_mask = (h_local >= qb * QBLOCK) & (h_local < (qb + 1) * QBLOCK)

            m_abs_qb = tl.where(qb_mask[None, :], m_abs, 0.0)
            m_absmax = tl.max(m_abs_qb, axis=1)
            m_scale_qb = m_absmax / 127.0 + 1e-12
            m_div = tl.where(qb_mask[None, :], m / m_scale_qb[:, None], 0.0)
            m_q_val = tl.extra.cuda.libdevice.rint(m_div).to(tl.int32)
            m_q_val = tl.minimum(tl.maximum(m_q_val, -127), 127).to(tl.int8)
            m_q_out = tl.where(qb_mask[None, :], m_q_val, m_q_out)

            v_abs_qb = tl.where(qb_mask[None, :], v_abs, 0.0)
            v_absmax = tl.max(v_abs_qb, axis=1)
            v_scale_qb = v_absmax / 127.0 + 1e-12
            v_div = tl.where(qb_mask[None, :], v / v_scale_qb[:, None], 0.0)
            v_q_val = tl.extra.cuda.libdevice.rint(v_div).to(tl.int32)
            v_q_val = tl.minimum(tl.maximum(v_q_val, -127), 127).to(tl.int8)
            v_q_out = tl.where(qb_mask[None, :], v_q_val, v_q_out)

            sc = pid_h * BPT + qb
            tl.store(m_scale_ptr + v_off * ms_stride_v + sc * ms_stride_h,
                     m_scale_qb, mask=v_mask)
            tl.store(v_scale_ptr + v_off * ms_stride_v + sc * ms_stride_h,
                     v_scale_qb, mask=v_mask)

        m_q_desc.store([off_v, off_h], m_q_out)
        v_q_desc.store([off_v, off_h], v_q_out)


@triton.jit
def _fused_grad_sgd_persistent(
    grad_output_ptr, input_ptr, weight_ptr,
    BT, V, H,
    go_stride_bt, go_stride_v,
    in_stride_bt, in_stride_h,
    w_stride_v, w_stride_h,
    lr, weight_decay,
    NUM_SMS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    GROUP_SIZE_V: tl.constexpr,
):
    """Persistent fused grad+SGD: one CTA per SM, looping over weight tiles."""
    start_pid = tl.program_id(0)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_h = tl.cdiv(H, BLOCK_H)
    num_tiles = num_pid_v * num_pid_h

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_V * num_pid_h
        group_id = tile_id // num_pid_in_group
        first_pid_v = group_id * GROUP_SIZE_V
        group_size = min(num_pid_v - first_pid_v, GROUP_SIZE_V)
        pid_v = first_pid_v + ((tile_id % num_pid_in_group) % group_size)
        pid_h = (tile_id % num_pid_in_group) // group_size

        v_off = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        v_mask = v_off < V
        h_mask = h_off < H

        grad_acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
        for bt_start in range(0, BT, BLOCK_BT):
            go_block_ptr = tl.make_block_ptr(
                base=grad_output_ptr, shape=(V, BT),
                strides=(go_stride_v, go_stride_bt),
                offsets=(pid_v * BLOCK_V, bt_start),
                block_shape=(BLOCK_V, BLOCK_BT), order=(0, 1),
            )
            inp_block_ptr = tl.make_block_ptr(
                base=input_ptr, shape=(BT, H),
                strides=(in_stride_bt, in_stride_h),
                offsets=(bt_start, pid_h * BLOCK_H),
                block_shape=(BLOCK_BT, BLOCK_H), order=(1, 0),
            )
            go = tl.load(go_block_ptr, boundary_check=(0, 1))
            inp = tl.load(inp_block_ptr, boundary_check=(0, 1))
            grad_acc += tl.dot(go, inp, out_dtype=tl.float32)

        w_ptrs = weight_ptr + v_off[:, None] * w_stride_v + h_off[None, :] * w_stride_h
        w_mask = v_mask[:, None] & h_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0, eviction_policy="evict_first")
        w_f32 = w.to(tl.float32)
        w_f32 = w_f32 * (1.0 - lr * weight_decay) - lr * grad_acc
        tl.store(w_ptrs, w_f32.to(w.dtype), mask=w_mask, eviction_policy="evict_first")


# Cache SM count per device to avoid repeated queries
_num_sms_cache: dict[int, int] = {}

def _get_num_sms(device) -> int:
    """Get SM count for a device, cached."""
    idx = device.index if device.index is not None else 0
    if idx not in _num_sms_cache:
        _num_sms_cache[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _num_sms_cache[idx]

# Minimum tiles to justify persistent kernel overhead (below this, non-persistent is fine)
_PERSISTENT_TILE_THRESHOLD = 256

# Benchmarking toggle: set to "persistent", "autotuned", or "auto" (default).
# "auto" uses persistent for large layers, autotuned for small.
# The benchmark script can override this to compare the two paths directly.
_kernel_mode = "auto"

# Pipeline depth for the persistent kernel's inner BT loop.
# 2 = safe default (64 KB SMEM). 3 = deeper pipeline (96 KB, close to 99 KB limit).
_persistent_num_stages = 2

# Pre-load W/m/v before the matmul loop to overlap state loads with compute.
# True = issue loads early (may cause register spilling if too tight).
# False = load after matmul (original order, safe).
_preload_state = False

# Use TMA (Tensor Memory Accelerator) for loads/stores in the persistent kernel.
# TMA uses a hardware copy engine instead of per-thread loads — fewer instructions,
# perfect coalescing, and better async overlap. Requires sm_90+ (Hopper/Blackwell)
# and Triton>=3.4 for tl.make_tensor_descriptor.
#
# Measured on H200 / Llama 8B / SEQ=4096: TMA cuts persistent bwd from
# ~382 ms to ~351 ms (8% faster) and is bit-exact vs the block_ptr path.
# Default kept at False because Triton 3.2.0 (the version torch 2.6 pins)
# doesn't have tl.make_tensor_descriptor. Flip True only if you've upgraded
# Triton to 3.4+ (see pyproject.toml note). The benchmark forces it True
# regardless, gated on the runtime Triton version.
_use_tma = False

# Layer on Triton's auto-warp-specialization on top of the TMA persistent
# kernel, with per-shape @triton.autotune. Requires Triton>=3.4 and
# num_warps=4 (Hopper lowering drops WS at num_warps=8). Bit-exact vs the
# non-WS TMA path.
#
# Measured on H200 / Llama 8B / SEQ=4096:
#   TMA persistent (no WS)          : bwd ≈ 349 ms  total ≈ 499 ms
#   TMA persistent + WS (autotuned) : bwd ≈ 301 ms  total ≈ 438 ms  ← best
#   EVT-Sm90 128x256 (CUTLASS)      : bwd ≈ 333 ms  total ≈ 482 ms
# Only path that brings total step time below baseline on H200.
#
# Only takes effect when `_use_tma=True`. Ignored otherwise.
_use_tma_ws = False


# Opt-in Stream-K path: replace the fused WS kernel entirely with
#
#     grad_w = grad_output.t() @ input        # cuBLAS bf16 GEMM
#     optimizer_only_adamw(grad_w, W, m, v)   # existing Triton opt-only
#
# Rationale — measured on Llama-3.1-8B / BT=4096 / H200 / FA3 + Liger:
#     Triton + TMA + WS          bwd ≈ 246.4 ms (σ 0.1)
#     cuBLAS + opt-only          bwd ≈ 238.0 ms (σ 5.0, 3 runs)
#     delta                       ≈ −8.5 ms (−3.4%) mean, Stream-K wins every run
# The fusion win (avoiding grad_W HBM round-trip, ~0.5–1 ms total) is
# smaller than cuBLAS's GEMM-scheduling edge over our persistent-WS path
# (~9 ms across all fused linears). Same pattern as CUTLASS 3.x EVT
# losing to the 2-kernel fallback (see docs/h200_optimization_journey/
# README.md §6).
#
# Default OFF because the delta is close to noise (σ of Stream-K ≈ 5 ms)
# and shape-dependent. Flip to True on workloads where matmul efficiency
# matters more than HBM round-trip (e.g., BT >> H).
_use_streamk = False


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def fused_grad_sgd(
    grad_output: torch.Tensor,   # (BT, V)
    input: torch.Tensor,         # (BT, H)
    weight: torch.Tensor,        # (V, H) — modified in-place
    lr: float = 1e-3,
    weight_decay: float = 0.0,
):
    """Fused weight gradient + SGD update. grad_W is never allocated."""
    assert grad_output.is_contiguous() and input.is_contiguous()
    BT, V = grad_output.shape
    H = input.shape[1]
    grid = lambda meta: (triton.cdiv(V, meta['BLOCK_V']) * triton.cdiv(H, meta['BLOCK_H']),)

    _fused_grad_sgd_kernel[grid](
        grad_output, input, weight,
        BT, V, H,
        grad_output.stride(0), grad_output.stride(1),
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        lr, weight_decay,
    )


def fused_grad_adamw(
    grad_output: torch.Tensor,   # (BT, V)
    input: torch.Tensor,         # (BT, H)
    weight: torch.Tensor,        # (V, H) — modified in-place
    m: torch.Tensor,             # (V, H) — modified in-place
    v: torch.Tensor,             # (V, H) — modified in-place
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    step: int = 1,
):
    """Fused weight gradient + AdamW update. grad_W is never allocated.

    For large layers (total tiles > _PERSISTENT_TILE_THRESHOLD) uses the
    persistent kernel: exactly NUM_SMS CTAs are launched, each looping over
    all weight tiles internally. This eliminates wave-quantization overhead
    and keeps the input/grad_output tensors warm in L2 across the whole sweep.
    For small layers the standard autotuned kernel is faster (lower launch cost).
    """
    assert grad_output.is_contiguous() and input.is_contiguous()
    BT, V = grad_output.shape
    H = input.shape[1]

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    # Opt-in Stream-K path: cuBLAS grad_W + Triton opt-only AdamW.
    # See comment on `_use_streamk` above for rationale / measurements.
    if _use_streamk:
        grad_w = grad_output.t() @ input  # (V, H) bf16 via cuBLAS
        optimizer_only_adamw(
            grad_w, weight, m, v,
            lr=lr, beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=weight_decay, step=step,
        )
        return

    # Choose tile size for persistent kernel. Use 128x128 with BLOCK_BT=64
    # and GROUP_SIZE_V=32 — balances register pressure and L2 reuse on Blackwell.
    # 128x128 gives 1002 tiles for lm_head vs 8032 for 64x64, meaning each SM
    # loops ~5 tiles instead of ~43, keeping more of the BT inner loop hot.
    PBLOCKE_V   = 128
    PBLOCK_H    = 128
    PBLOCK_BT   = 64
    PGROUP_SIZE = 32

    num_tiles = triton.cdiv(V, PBLOCKE_V) * triton.cdiv(H, PBLOCK_H)

    # Decide which kernel to use based on _kernel_mode toggle
    use_persistent = (
        _kernel_mode == "persistent"
        or (_kernel_mode == "auto" and num_tiles >= _PERSISTENT_TILE_THRESHOLD)
    )

    if use_persistent:
        num_sms = _get_num_sms(weight.device)
        if _use_tma and _use_tma_ws:
            # TMA + auto-warp-specialization + autotune.
            # Tile sizes, num_warps (must be 4), and num_stages all come from
            # @triton.autotune on the kernel; we only pass NUM_SMS as a
            # compile-time constant because it's a device property.
            _ensure_tma_allocator()
            _fused_grad_adamw_persistent_tma_ws[(num_sms,)](
                grad_output, input, weight, m, v,
                BT, V, H,
                grad_output.stride(0), grad_output.stride(1),
                input.stride(0), input.stride(1),
                weight.stride(0), weight.stride(1),
                lr, beta1, beta2, eps, weight_decay, bc1, bc2,
                NUM_SMS=num_sms,
            )
        elif _use_tma:
            # TMA kernel: hardware copy engine for all loads/stores
            _ensure_tma_allocator()
            _fused_grad_adamw_persistent_tma[(num_sms,)](
                grad_output, input, weight, m, v,
                BT, V, H,
                grad_output.stride(0), grad_output.stride(1),
                input.stride(0), input.stride(1),
                weight.stride(0), weight.stride(1),
                lr, beta1, beta2, eps, weight_decay, bc1, bc2,
                NUM_SMS=num_sms,
                BLOCK_V=PBLOCKE_V, BLOCK_H=PBLOCK_H, BLOCK_BT=PBLOCK_BT,
                GROUP_SIZE_V=PGROUP_SIZE,
                num_warps=8,
                num_stages=_persistent_num_stages,
            )
        else:
            # Regular persistent kernel: block_ptr for all loads/stores
            _fused_grad_adamw_persistent[(num_sms,)](
                grad_output, input, weight, m, v,
                BT, V, H,
                grad_output.stride(0), grad_output.stride(1),
                input.stride(0), input.stride(1),
                weight.stride(0), weight.stride(1),
                lr, beta1, beta2, eps, weight_decay, bc1, bc2,
                NUM_SMS=num_sms,
                BLOCK_V=PBLOCKE_V, BLOCK_H=PBLOCK_H, BLOCK_BT=PBLOCK_BT,
                GROUP_SIZE_V=PGROUP_SIZE,
                num_warps=8,
                num_stages=_persistent_num_stages,
            )
    else:
        grid = lambda meta: (triton.cdiv(V, meta['BLOCK_V']) * triton.cdiv(H, meta['BLOCK_H']),)
        _fused_grad_adamw_kernel[grid](
            grad_output, input, weight, m, v,
            BT, V, H,
            grad_output.stride(0), grad_output.stride(1),
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            lr, beta1, beta2, eps, weight_decay, bc1, bc2,
        )


def optimizer_only_adamw(
    grad: torch.Tensor,          # (V, H) — pre-computed gradient
    weight: torch.Tensor,        # (V, H) — modified in-place
    m: torch.Tensor,             # (V, H) — modified in-place
    v: torch.Tensor,             # (V, H) — modified in-place
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    step: int = 1,
):
    """AdamW on a pre-computed gradient (for gradient accumulation fallback)."""
    assert grad.is_contiguous()
    rows, cols = weight.shape
    grid = lambda meta: (triton.cdiv(rows, meta['BLOCK_R']), triton.cdiv(cols, meta['BLOCK_C']))

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    _optimizer_only_adamw_kernel[grid](
        grad, weight, m, v,
        rows, cols,
        grad.stride(0), grad.stride(1),
        weight.stride(0), weight.stride(1),
        lr, beta1, beta2, eps, weight_decay, bc1, bc2,
    )


def fused_grad_adamw_int8state(
    grad_output: torch.Tensor,   # (BT, V)
    input: torch.Tensor,         # (BT, H)
    weight: torch.Tensor,        # (V, H) — modified in-place
    m_q: torch.Tensor,           # (V, H) int8 — modified in-place
    v_q: torch.Tensor,           # (V, H) int8 — modified in-place
    m_scale: torch.Tensor,       # (V, H // qblock) fp32 — modified in-place
    v_scale: torch.Tensor,       # (V, H // qblock) fp32 — modified in-place
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    step: int = 1,
    qblock: int = 64,
):
    """Fused weight gradient + AdamW with int8 quantized m/v. grad_W never allocated.

    Dispatches to the persistent + TMA + WS int8 kernel when the _use_tma
    and _use_tma_ws toggles are on (same selection logic as the bf16 path
    in fused_grad_adamw). Otherwise falls back to the plain non-persistent
    int8 kernel.
    """
    assert grad_output.is_contiguous() and input.is_contiguous()
    BT, V = grad_output.shape
    H = input.shape[1]
    assert H % qblock == 0, f"H ({H}) must be divisible by qblock ({qblock})"

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    # NOTE: the persistent-TMA-WS int8 kernel
    # (_fused_grad_adamw_persistent_tma_ws_int8state) exists in this file
    # but is NOT dispatched here. Tested on Triton 3.4 and 3.6 — both
    # produce silent NaN on a handful of tiles under WS=True (Triton WS
    # partitioner can't correctly lower the int8 requantize epilogue).
    # With WS=False the kernel is correct but loses to the non-persistent
    # grid launch. See ncu data + notes in the kernel's docstring.
    # Always use the non-persistent int8 kernel here until Triton's WS
    # codegen catches up.
    grid = lambda meta: (triton.cdiv(V, meta['BLOCK_V']) * triton.cdiv(H, meta['BLOCK_H']),)
    _fused_grad_adamw_int8state_kernel[grid](
        grad_output, input, weight,
        m_q, v_q, m_scale, v_scale,
        BT, V, H,
        grad_output.stride(0), grad_output.stride(1),
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        m_q.stride(0), m_q.stride(1),
        m_scale.stride(0), m_scale.stride(1),
        lr, beta1, beta2, eps, weight_decay, bc1, bc2,
        QBLOCK=qblock,
    )


def optimizer_only_adamw_int8state(
    grad: torch.Tensor,          # (V, H) — pre-computed gradient
    weight: torch.Tensor,        # (V, H) — modified in-place
    m_q: torch.Tensor,           # (V, H) int8
    v_q: torch.Tensor,           # (V, H) int8
    m_scale: torch.Tensor,       # (V, H // qblock) fp32
    v_scale: torch.Tensor,       # (V, H // qblock) fp32
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    step: int = 1,
    qblock: int = 64,
):
    """AdamW on a pre-computed gradient with int8 quantized m/v states."""
    assert grad.is_contiguous()
    rows, cols = weight.shape
    assert cols % qblock == 0, f"cols ({cols}) must be divisible by qblock ({qblock})"
    grid = lambda meta: (triton.cdiv(rows, meta['BLOCK_R']), triton.cdiv(cols, meta['BLOCK_C']))

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    _optimizer_only_adamw_int8state_kernel[grid](
        grad, weight, m_q, v_q, m_scale, v_scale,
        rows, cols,
        grad.stride(0), grad.stride(1),
        weight.stride(0), weight.stride(1),
        m_q.stride(0), m_q.stride(1),
        m_scale.stride(0), m_scale.stride(1),
        lr, beta1, beta2, eps, weight_decay, bc1, bc2,
        QBLOCK=qblock,
    )
