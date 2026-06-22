"""
v2 of the TMA + warp-specialized persistent fused grad+AdamW kernel (Hopper).

Derived from kernel._fused_grad_adamw_persistent_tma_ws, with changes that
only use primitives already present in Triton 3.4:

1.  **Structural 2-way epilogue subtiling on the BLOCK_H axis.**

    The reshape/permute/tl.split pattern from
    python/tutorials/09-persistent-matmul.py:560 (upstream Triton) does NOT
    survive Hopper's NVGPUWarpSpecialization pass on Triton 3.4 / 3.6 — it
    aborts with `note: Pipeline failed while executing
    [NVGPUWarpSpecialization]: reproducer generated...`. This matches the
    existing kernel.py comment on the int8 WS hybrid
    ("3D-reshape requantize + WS=True on Triton 3.6 → compile error").

    Instead, we subtile structurally: run two half-width MMAs (128x64 each)
    sharing one TMA load of grad_output, then run the AdamW epilogue twice
    on (BLOCK_V, BLOCK_H/2) accumulators. Under warp_specialize=True the
    second half's load/store pair can overlap with the first half's stores.

    The M axis is not subtiled — BLOCK_V stays 128 because the Hopper
    WS+num_warps=4 register budget is already at the ceiling (128x128 fp32
    ≈ 128 regs/thread).

2.  **`disallow_acc_multi_buffer=True` on the outer persistent tl.range.**

    Tells Triton not to double-buffer the fp32 accumulator across outer
    iterations, freeing smem for a deeper inner-K pipeline (num_stages=4-5
    instead of 3).

Math is byte-for-byte identical to the v1 WS kernel (same fp32 fma order,
same tl.sqrt, same division) once you flatten the two halves — each half
does the same AdamW update on the same (V, H/2) slice of state. Bit-exact
equivalence is checked by benchmarks/benchmark_llama_selective_h200_v5_ws_v2.py
before the wall-clock run.

Hopper WS gates enforced here (same as v1):
  * num_warps=4 (Hopper lowering drops WS at num_warps=8 —
    upstream test_warp_specialization.py:314)
  * all loads via TMA
  * Triton >= 3.4 for tl.make_tensor_descriptor + warp_specialize
"""

import torch
import triton
import triton.language as tl

from fused_grad_optimizer.kernel import _ensure_tma_allocator, _get_num_sms


def _ws_tma_v2_configs():
    """Autotune configs for the subtiled TMA+WS persistent kernel.

    BLOCK_V / BLOCK_H stay at 128 (Hopper WS + num_warps=4 register ceiling
    on a 128x128 fp32 accumulator). The subtile split is controlled by
    SUBTILE_H. Config set is pruned to the axes that the v1 WS autotune
    ever picked as winners on Llama-3.1-8B — BT ∈ {64, 128}, GROUP_SIZE_V=8,
    num_stages ∈ {3, 4, 5}. SUBTILE_H ∈ {1, 2} is included so autotune can
    fall back to the unsubtiled path on shapes where it's slower.
    """
    cfgs = []
    for subtile in (2, 1):
        for BT in (64, 128):
            for ns in (3, 4, 5):
                cfgs.append(triton.Config(
                    {'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': BT,
                     'GROUP_SIZE_V': 8, 'SUBTILE_H': subtile},
                    num_warps=4, num_stages=ns,
                ))
    return cfgs


@triton.autotune(
    configs=_ws_tma_v2_configs(),
    key=['BT', 'V', 'H'],
    restore_value=['weight_ptr', 'm_ptr', 'v_ptr'],
)
@triton.jit
def _fused_grad_adamw_persistent_tma_ws_v2(
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
    SUBTILE_H: tl.constexpr,
):
    """Persistent TMA fused grad+AdamW, WS + structural H-axis subtiling.

    SUBTILE_H == 1 : identical to the v1 WS kernel (full BLOCK_H epilogue).
    SUBTILE_H == 2 : two half-BLOCK_H MMAs that share the go load, each
                     followed by its own AdamW epilogue.
    """
    start_pid = tl.program_id(0)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_h = tl.cdiv(H, BLOCK_H)
    num_tiles = num_pid_v * num_pid_h

    BLOCK_H_OUT: tl.constexpr = BLOCK_H // SUBTILE_H

    # Full-width grad_output descriptor. We load one (BLOCK_BT, BLOCK_V) GO
    # tile per K-step and transpose; both subtiles consume the same GO, so
    # it's loaded once per bt iteration regardless of SUBTILE_H.
    go_desc = tl.make_tensor_descriptor(
        grad_output_ptr, shape=[BT, V],
        strides=[go_stride_bt, go_stride_v],
        block_shape=[BLOCK_BT, BLOCK_V],
    )
    # Half-width input descriptor — one TMA load per half per bt iteration.
    inp_desc = tl.make_tensor_descriptor(
        input_ptr, shape=[BT, H],
        strides=[in_stride_bt, in_stride_h],
        block_shape=[BLOCK_BT, BLOCK_H_OUT],
    )
    # Half-width state descriptors for the epilogue.
    w_desc = tl.make_tensor_descriptor(
        weight_ptr, shape=[V, H], strides=[w_stride_v, w_stride_h],
        block_shape=[BLOCK_V, BLOCK_H_OUT],
    )
    m_desc = tl.make_tensor_descriptor(
        m_ptr, shape=[V, H], strides=[w_stride_v, w_stride_h],
        block_shape=[BLOCK_V, BLOCK_H_OUT],
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[V, H], strides=[w_stride_v, w_stride_h],
        block_shape=[BLOCK_V, BLOCK_H_OUT],
    )

    for tile_id in tl.range(
            start_pid, num_tiles, NUM_SMS,
            warp_specialize=True,
            disallow_acc_multi_buffer=True,
    ):
        num_pid_in_group = GROUP_SIZE_V * num_pid_h
        group_id = tile_id // num_pid_in_group
        first_pid_v = group_id * GROUP_SIZE_V
        group_size = min(num_pid_v - first_pid_v, GROUP_SIZE_V)
        pid_v = first_pid_v + ((tile_id % num_pid_in_group) % group_size)
        pid_h = (tile_id % num_pid_in_group) // group_size

        off_v = pid_v * BLOCK_V
        off_h = pid_h * BLOCK_H

        if SUBTILE_H == 2:
            # --- K-reduction: two half-width MMAs sharing the GO load ---
            acc0 = tl.zeros((BLOCK_V, BLOCK_H // 2), dtype=tl.float32)
            acc1 = tl.zeros((BLOCK_V, BLOCK_H // 2), dtype=tl.float32)
            off_h1 = off_h + BLOCK_H // 2
            for bt_start in range(0, BT, BLOCK_BT):
                go   = tl.trans(go_desc.load([bt_start, off_v]))
                inp0 = inp_desc.load([bt_start, off_h])
                inp1 = inp_desc.load([bt_start, off_h1])
                acc0 = tl.dot(go, inp0, acc=acc0, out_dtype=tl.float32)
                acc1 = tl.dot(go, inp1, acc=acc1, out_dtype=tl.float32)

            # --- Subtile 0 epilogue ---
            w0_raw = w_desc.load([off_v, off_h])
            m0_raw = m_desc.load([off_v, off_h])
            v0_raw = v_desc.load([off_v, off_h])
            w0 = w0_raw.to(tl.float32)
            m0 = beta1 * m0_raw + (1.0 - beta1) * acc0
            v0 = beta2 * v0_raw + (1.0 - beta2) * acc0 * acc0
            m0_hat = m0 / bias_correction1
            v0_hat = v0 / bias_correction2
            w0 = w0 * (1.0 - lr * weight_decay)
            w0 = w0 - lr * m0_hat / (tl.sqrt(v0_hat) + eps)
            w_desc.store([off_v, off_h], w0.to(w0_raw.dtype))
            m_desc.store([off_v, off_h], m0.to(m0_raw.dtype))
            v_desc.store([off_v, off_h], v0.to(v0_raw.dtype))

            # --- Subtile 1 epilogue. Under warp_specialize=True the loads
            # below can overlap with the stores above. ---
            w1_raw = w_desc.load([off_v, off_h1])
            m1_raw = m_desc.load([off_v, off_h1])
            v1_raw = v_desc.load([off_v, off_h1])
            w1 = w1_raw.to(tl.float32)
            m1 = beta1 * m1_raw + (1.0 - beta1) * acc1
            v1 = beta2 * v1_raw + (1.0 - beta2) * acc1 * acc1
            m1_hat = m1 / bias_correction1
            v1_hat = v1 / bias_correction2
            w1 = w1 * (1.0 - lr * weight_decay)
            w1 = w1 - lr * m1_hat / (tl.sqrt(v1_hat) + eps)
            w_desc.store([off_v, off_h1], w1.to(w1_raw.dtype))
            m_desc.store([off_v, off_h1], m1.to(m1_raw.dtype))
            v_desc.store([off_v, off_h1], v1.to(v1_raw.dtype))
        else:
            # SUBTILE_H == 1: identical to the v1 WS kernel body.
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
# Python wrapper
# ---------------------------------------------------------------------------

def fused_grad_adamw_ws_v2(
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
):
    """Drop-in replacement for kernel.fused_grad_adamw that routes to the
    v2 persistent TMA+WS kernel (structural epilogue subtiling + deeper
    inner pipeline).

    Same arg contract as the v1 wrapper, same in-place semantics on W/m/v.
    """
    assert grad_output.is_contiguous() and input.is_contiguous()
    BT, V = grad_output.shape
    H = input.shape[1]
    assert weight.shape == (V, H)
    assert m.shape == (V, H)
    assert v.shape == (V, H)

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    _ensure_tma_allocator()
    num_sms = _get_num_sms(weight.device)

    _fused_grad_adamw_persistent_tma_ws_v2[(num_sms,)](
        grad_output, input, weight, m, v,
        BT, V, H,
        grad_output.stride(0), grad_output.stride(1),
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        lr, beta1, beta2, eps, weight_decay, bc1, bc2,
        NUM_SMS=num_sms,
    )


def patch_dispatch():
    """Patch fused_grad_optimizer.kernel + .autograd so FusedLinear's
    backward calls the v2 kernel. Mirrors the _patch/_restore pattern used
    in benchmark_llama_selective_*.
    """
    import fused_grad_optimizer.kernel as _k
    import fused_grad_optimizer.autograd as _a
    originals = (_k.fused_grad_adamw, _a.fused_grad_adamw)
    _k.fused_grad_adamw = fused_grad_adamw_ws_v2
    _a.fused_grad_adamw = fused_grad_adamw_ws_v2
    return originals


def restore_dispatch(originals):
    import fused_grad_optimizer.kernel as _k
    import fused_grad_optimizer.autograd as _a
    _k.fused_grad_adamw = originals[0]
    _a.fused_grad_adamw = originals[1]
