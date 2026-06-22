"""
v7: FP8 fused grad+AdamW kernel — FlashAttention-3-style low-precision MMA.

Motivation (from FA-3 §3.3):
  Hopper FP8 WGMMA delivers 2× the FLOPs/s of bf16/fp16 WGMMA. The backward
  pass of a Llama-3.1-8B layer is ~70% grad-MMA, so halving the MMA cost is
  where the biggest remaining win comes from on the Triton side.

What this kernel does:
  1. Cast bf16 `grad_output` and `input` to FP8 E4M3 with per-tile fp32
     scales computed inline (block-quantization, FA-3 §3.3). Scale = absmax
     / E4M3_MAX (448.0) so the cast saturates just below the format max.
  2. Run the K-reduction MMA with both operands FP8 (`tl.dot` picks up
     fp8e4nv automatically) and an fp32 accumulator — same as FA-3 FP8.
  3. Multiply the fp32 accumulator by `go_scale * inp_scale` per output
     tile to undo the block quantization.
  4. Run the AdamW epilogue identical to v1 WS on the rescaled gradient.
     W / m / v stay bf16 throughout.

Numerical notes:
  * Per-BT-slab quantization (one scalar per (BT, V) or (BT, H) tile per
    kernel invocation, not per inner K-chunk) is coarser than FA-3's
    per-query-block but is what's computable without a second kernel pass.
  * The cast GO and IN buffers are throwaway scratch, allocated once per
    call. Peak memory is +bf16_input_bytes / 2 (FP8 is half of bf16).
  * `disallow_acc_multi_buffer=True` + `disable_licm=True` on the outer
    persistent loop, matching the v3 path that survived Triton 3.6.

Limitations / caveats:
  * FP8 gradient MMA IS numerically riskier than FP8 forward (no softmax
    renorm to mask error). Expect correctness drift on long training runs —
    this file is for BENCHMARK only until a convergence test validates it.
  * `warp_specialize=True` with FP8 inputs has historically been flakier
    than bf16; we keep it ON here because the 3D-TMA/permute experiment
    was the specific pattern that broke, not FP8 WGMMA itself.

See benchmarks/benchmark_llama_selective_h200_v6_fp8.py for the A/B harness.
"""

import torch
import triton
import triton.language as tl

from fused_grad_optimizer.kernel import _ensure_tma_allocator, _get_num_sms

# E4M3 max representable magnitude (absmax / E4M3_MAX = scale).
_E4M3_MAX = 448.0


def _fp8_configs():
    """Autotune configs for the FP8 v7 kernel.

    BLOCK_BT is locked to _FP8_SLAB (=64) so the scale indexing
    `slab_id += 1` per inner-loop iteration stays in 1:1 correspondence
    with the per-slab scales emitted by the pre-quantization kernel.
    """
    return [
        triton.Config(
            {'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64, 'GROUP_SIZE_V': 8},
            num_warps=4, num_stages=ns,
        )
        for ns in (3, 4, 5)
    ]


# ---------------------------------------------------------------------------
# Pre-quantization kernel: bf16 tensor → FP8 E4M3 + per-row-slab fp32 scale.
# ---------------------------------------------------------------------------
# A (ROWS, COLS) bf16 tensor is split row-wise into tiles of SLAB rows each.
# Each program handles one slab: two inner passes over BLOCK_C-wide column
# chunks — first computes absmax, second writes FP8. Keeps tile numel small
# regardless of COLS (lm_head has V=128256, which exceeded Triton's 1M limit
# when we tried one-program-per-slab with BLOCK_C=next_pow2(COLS)).
@triton.jit
def _quantize_rowslab_fp8(
    src_ptr, dst_ptr, scale_ptr,
    ROWS, COLS,
    src_stride_r, src_stride_c,
    dst_stride_r, dst_stride_c,
    SLAB: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """dst[slab*SLAB:(slab+1)*SLAB, :] = quantize_e4m3(src slab, absmax/448).

    One program per slab; inner loop chunks columns so SLAB*BLOCK_C stays
    well under Triton's 1M-element tile limit.
    """
    pid_slab = tl.program_id(0)
    row0     = pid_slab * SLAB
    off_r    = row0 + tl.arange(0, SLAB)
    row_mask = off_r < ROWS

    # Pass 1: compute absmax across all columns.
    am = tl.zeros((), dtype=tl.float32)
    for c0 in range(0, COLS, BLOCK_C):
        off_c = c0 + tl.arange(0, BLOCK_C)
        mask  = row_mask[:, None] & (off_c[None, :] < COLS)
        ptrs  = src_ptr + off_r[:, None] * src_stride_r + off_c[None, :] * src_stride_c
        x     = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        am    = tl.maximum(am, tl.max(tl.abs(x)))

    scale = tl.maximum(am, 1e-12) / 448.0
    tl.store(scale_ptr + pid_slab, scale)

    # Pass 2: write quantized FP8.
    inv_scale = 1.0 / scale
    for c0 in range(0, COLS, BLOCK_C):
        off_c = c0 + tl.arange(0, BLOCK_C)
        mask  = row_mask[:, None] & (off_c[None, :] < COLS)
        sptrs = src_ptr + off_r[:, None] * src_stride_r + off_c[None, :] * src_stride_c
        x     = tl.load(sptrs, mask=mask, other=0.0).to(tl.float32)
        q     = (x * inv_scale).to(tl.float8e4nv)
        dptrs = dst_ptr + off_r[:, None] * dst_stride_r + off_c[None, :] * dst_stride_c
        tl.store(dptrs, q, mask=mask)


# ---------------------------------------------------------------------------
# Main fused grad+AdamW kernel with FP8 K-reduction.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=_fp8_configs(),
    key=['BT', 'V', 'H'],
    restore_value=['weight_ptr', 'm_ptr', 'v_ptr'],
)
@triton.jit
def _fused_grad_adamw_fp8_v7(
    grad_output_fp8_ptr, input_fp8_ptr,
    go_scale_ptr, inp_scale_ptr,          # (n_slabs,) fp32 per-slab scales
    weight_ptr, m_ptr, v_ptr,
    BT, V, H, n_slabs,
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
    """Persistent TMA WS kernel: FP8 MMA for grad_W, bf16 AdamW epilogue.

    grad_output_fp8 and input_fp8 are pre-quantized (shape identical to the
    bf16 originals, dtype float8_e4m3). The per-slab scales let us undo the
    block quantization with a single scalar multiply per inner K-chunk.
    """
    start_pid = tl.program_id(0)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_h = tl.cdiv(H, BLOCK_H)
    num_tiles = num_pid_v * num_pid_h

    # FP8 descriptors — same (BT, V) / (BT, H) shapes as bf16 but element
    # size is 1 byte, so BLOCK_BT can be larger for the same smem budget.
    go_desc = tl.make_tensor_descriptor(
        grad_output_fp8_ptr, shape=[BT, V],
        strides=[go_stride_bt, go_stride_v],
        block_shape=[BLOCK_BT, BLOCK_V],
    )
    inp_desc = tl.make_tensor_descriptor(
        input_fp8_ptr, shape=[BT, H],
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

    # NOTE: warp_specialize=True on FP8 + 2 TMA loads fails in
    # TritonNvidiaGPUTMALoweringPass on Triton 3.6 (reproducer captured in
    # benchmarks/logs/v6_fp8_first_attempt_ws.log). FP8 WGMMA has a k-major
    # constraint on both operands (FA-3 §3.3) that doesn't play nicely with
    # our tl.trans(go) layout under WS partitioning. Keep WS off for FP8.
    for tile_id in tl.range(
            start_pid, num_tiles, NUM_SMS,
            disallow_acc_multi_buffer=True,
            disable_licm=True,
    ):
        num_pid_in_group = GROUP_SIZE_V * num_pid_h
        group_id = tile_id // num_pid_in_group
        first_pid_v = group_id * GROUP_SIZE_V
        group_size = min(num_pid_v - first_pid_v, GROUP_SIZE_V)
        pid_v = first_pid_v + ((tile_id % num_pid_in_group) % group_size)
        pid_h = (tile_id % num_pid_in_group) // group_size

        off_v = pid_v * BLOCK_V
        off_h = pid_h * BLOCK_H

        # --- K-reduction in FP8. BLOCK_BT must equal the slab size used by
        # the pre-quantization kernel so scales align 1:1 with K-chunks. ---
        grad_acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
        slab_id = 0
        for bt_start in range(0, BT, BLOCK_BT):
            go_fp8  = tl.trans(go_desc.load([bt_start, off_v]))
            inp_fp8 = inp_desc.load([bt_start, off_h])
            # FP8 × FP8 → fp32 accumulator. tl.dot auto-selects FP8 WGMMA.
            chunk = tl.dot(go_fp8, inp_fp8, out_dtype=tl.float32)
            # Rescale this K-chunk by (go_scale * inp_scale). Scales are
            # one-per-slab scalars loaded with a single fp32 load each.
            go_s  = tl.load(go_scale_ptr  + slab_id)
            inp_s = tl.load(inp_scale_ptr + slab_id)
            grad_acc += chunk * (go_s * inp_s)
            slab_id += 1

        # --- AdamW epilogue, identical to v1 WS ---
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
# Python wrapper
# ---------------------------------------------------------------------------

# Reusable scratch — allocated lazily on first call per (shape, device) key.
_fp8_scratch: dict[tuple, dict] = {}


def _get_fp8_scratch(BT: int, V: int, H: int, device: torch.device,
                     n_slabs: int) -> dict:
    """Lazy-alloc scratch buffers for go_fp8, inp_fp8, and their scales."""
    key = (BT, V, H, device.index)
    if key not in _fp8_scratch:
        _fp8_scratch[key] = {
            'go_fp8':    torch.empty(BT, V, device=device, dtype=torch.float8_e4m3fn),
            'inp_fp8':   torch.empty(BT, H, device=device, dtype=torch.float8_e4m3fn),
            'go_scale':  torch.empty(n_slabs, device=device, dtype=torch.float32),
            'inp_scale': torch.empty(n_slabs, device=device, dtype=torch.float32),
        }
    return _fp8_scratch[key]


def _quantize_fp8(src: torch.Tensor, dst: torch.Tensor,
                  scale: torch.Tensor, slab_size: int) -> None:
    """Row-slab quantize bf16 src → FP8 E4M3 dst + fp32 per-slab scale.

    slab_size = how many bf16 rows share one scale. Must evenly divide
    src.shape[0]. Inner column chunk of 256 keeps SLAB*BLOCK_C under the
    Triton 1M-element tile limit even for lm_head (V=128256).
    """
    ROWS, COLS = src.shape
    assert ROWS % slab_size == 0
    n_slabs = ROWS // slab_size
    BLOCK_C = 256
    _quantize_rowslab_fp8[(n_slabs,)](
        src, dst, scale,
        ROWS, COLS,
        src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1),
        SLAB=slab_size, BLOCK_C=BLOCK_C,
    )


# Fixed slab size = BT tiling used by the main kernel's inner K-loop.
# Must match BLOCK_BT picked by autotune. We hard-pick 64 here (the v1/v3
# autotune winner on most Llama shapes); the main kernel's autotune is
# constrained in _fp8_configs() so BLOCK_BT ∈ {64, 128}.
_FP8_SLAB = 64


def fused_grad_adamw_fp8_v7(
    grad_output: torch.Tensor,   # (BT, V) bf16
    input: torch.Tensor,         # (BT, H) bf16
    weight: torch.Tensor,        # (V, H) bf16 — in-place
    m: torch.Tensor,             # (V, H) bf16 — in-place
    v: torch.Tensor,             # (V, H) bf16 — in-place
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    step: int = 1,
):
    """Drop-in replacement for kernel.fused_grad_adamw using FP8 MMA.

    Pre-quantizes GO and IN to FP8 E4M3 with per-slab fp32 scales, then
    runs the persistent TMA WS kernel with an FP8 K-reduction. Peak
    throughput target ~2× the bf16 MMA path on Hopper.
    """
    assert grad_output.is_contiguous() and input.is_contiguous()
    assert grad_output.dtype == torch.bfloat16
    assert input.dtype == torch.bfloat16
    BT, V = grad_output.shape
    H = input.shape[1]
    # BT must be a multiple of the slab size so pre-quant aligns with the
    # main kernel's inner K-loop.
    if BT % _FP8_SLAB != 0:
        # Fall back to v1 path if we can't tile cleanly.
        from fused_grad_optimizer.kernel import fused_grad_adamw as _v1
        return _v1(grad_output, input, weight, m, v, lr=lr, beta1=beta1,
                   beta2=beta2, eps=eps, weight_decay=weight_decay, step=step)

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    _ensure_tma_allocator()
    num_sms = _get_num_sms(weight.device)

    n_slabs = BT // _FP8_SLAB
    scratch = _get_fp8_scratch(BT, V, H, weight.device, n_slabs)

    # Pre-quantize GO and IN to FP8.
    _quantize_fp8(grad_output, scratch['go_fp8'],  scratch['go_scale'],  _FP8_SLAB)
    _quantize_fp8(input,       scratch['inp_fp8'], scratch['inp_scale'], _FP8_SLAB)

    _fused_grad_adamw_fp8_v7[(num_sms,)](
        scratch['go_fp8'], scratch['inp_fp8'],
        scratch['go_scale'], scratch['inp_scale'],
        weight, m, v,
        BT, V, H, n_slabs,
        scratch['go_fp8'].stride(0), scratch['go_fp8'].stride(1),
        scratch['inp_fp8'].stride(0), scratch['inp_fp8'].stride(1),
        weight.stride(0), weight.stride(1),
        lr, beta1, beta2, eps, weight_decay, bc1, bc2,
        NUM_SMS=num_sms,
    )


def patch_dispatch():
    import fused_grad_optimizer.kernel as _k
    import fused_grad_optimizer.autograd as _a
    originals = (_k.fused_grad_adamw, _a.fused_grad_adamw)
    _k.fused_grad_adamw = fused_grad_adamw_fp8_v7
    _a.fused_grad_adamw = fused_grad_adamw_fp8_v7
    return originals


def restore_dispatch(originals):
    import fused_grad_optimizer.kernel as _k
    import fused_grad_optimizer.autograd as _a
    _k.fused_grad_adamw = originals[0]
    _a.fused_grad_adamw = originals[1]
