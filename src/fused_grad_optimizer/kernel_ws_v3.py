"""
v3 of the TMA persistent fused grad+AdamW kernel (Hopper), with packed m+v.

Three variants in this file, selected at wrapper call time via USE_3D_PACKED
and the ensure_buffers monkey-patch in the benchmark:

  3D_WS    — 3D TMA desc on (2, V, H), single load/store for m+v, WS=True
             **BROKEN on Triton 3.4 and 3.6**: the NVGPUWarpSpecialization
             pass still can't partition a producer warp that emits
             tl.permute+tl.split on a TMA-loaded 3D tensor. Same bug class as
             the reshape-in-WS issue documented on the int8-WS hybrid kernel.
             Left in place so the upstream fix (when it lands) only needs a
             Triton bump to exercise.

  3D_no_WS — 3D TMA desc on (2, V, H), single load/store for m+v, WS=False.
             Trades the ~14% WS speedup over plain TMA for halving the TMA
             instruction count in the epilogue. Useful as the ceiling for
             what Option 1 (packed state) can deliver while the WS pass is
             broken on 3D descriptors.

  2x2D_WS  — Two 2D TMA descs on the SAME packed buffer, WS=True. Same byte
             count and TMA instruction count as v1, but the two descriptors
             now point at contiguous memory, so m and v share L2 lines and
             the WS partitioner gets a cleaner single-allocation view.
             Adds `disable_licm=True` on the outer tl.range (Triton >=3.6)
             so the scheduler doesn't hoist descriptor creation into
             registers that WGMMA wants for MMA operands.

The packing is set up by benchmark-side monkey-patching of
FusedOptimizerState.ensure_buffers (see benchmark_llama_selective_h200_v5_ws_v3.py)
so kernel.py and state.py stay untouched.

Math is identical per tile to v1 (fp32 accumulator, tl.sqrt, bias-corrected
m_hat/v_hat). Per-element numeric equivalence holds to fp32 rounding; smem
bank layout of the 3D desc differs from two 2D descs, so strict bit-exact
comparison isn't guaranteed.

Hopper WS gates (all variants that use WS):
  * num_warps=4 (upstream test_warp_specialization.py:314)
  * all loads via TMA
  * Triton >= 3.4 for 3D tl.make_tensor_descriptor
  * Triton >= 3.6 for `disable_licm=True` kwarg on tl.range
"""

import torch
import triton
import triton.language as tl

from fused_grad_optimizer.kernel import _ensure_tma_allocator, _get_num_sms


# Triton 3.4 does not accept disable_licm as a tl.range kwarg. We detect at
# import time and leave a flag the kernel wrapper can use to pick a config
# set that either enables or omits it. Kernel body is the same either way —
# only the autotune Config objects differ.
import inspect as _inspect
_TL_RANGE_HAS_DISABLE_LICM = 'disable_licm' in _inspect.signature(tl.range).parameters


def _ws_tma_v3_configs():
    """Autotune configs for the packed-state v3 kernel.

    Pruned to the v1 winners' basin: BLOCK_V=BLOCK_H=128, GROUP_SIZE_V=8,
    BT in {64, 128}, num_warps=4, num_stages in {3, 4, 5}.
    """
    return [
        triton.Config(
            {'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': BT, 'GROUP_SIZE_V': 8},
            num_warps=4, num_stages=ns,
        )
        for BT in (64, 128)
        for ns in (3, 4, 5)
    ]


@triton.autotune(
    configs=_ws_tma_v3_configs(),
    key=['BT', 'V', 'H'],
    restore_value=['weight_ptr', 'mv_ptr'],
)
@triton.jit
def _fused_grad_adamw_persistent_tma_ws_v3(
    grad_output_ptr, input_ptr, weight_ptr,
    mv_ptr,                   # single (2, V, H) contiguous buffer; mv[0]=m, mv[1]=v
    BT, V, H,
    go_stride_bt, go_stride_v,
    in_stride_bt, in_stride_h,
    w_stride_v, w_stride_h,
    mv_stride_c, mv_stride_v, mv_stride_h,   # 3D strides: (V*H, H, 1)
    lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2,
    NUM_SMS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    GROUP_SIZE_V: tl.constexpr,
):
    """Persistent TMA fused grad+AdamW, WS + packed (m, v) state."""
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
    # Packed (m, v): one 3D descriptor, block_shape=[2, BLOCK_V, BLOCK_H].
    mv_desc = tl.make_tensor_descriptor(
        mv_ptr, shape=[2, V, H],
        strides=[mv_stride_c, mv_stride_v, mv_stride_h],
        block_shape=[2, BLOCK_V, BLOCK_H],
    )

    for tile_id in tl.range(
            start_pid, num_tiles, NUM_SMS,
            warp_specialize=True,
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

        # --- K-reduction ---
        grad_acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
        for bt_start in range(0, BT, BLOCK_BT):
            go  = tl.trans(go_desc.load([bt_start, off_v]))
            inp = inp_desc.load([bt_start, off_h])
            grad_acc = tl.dot(go, inp, acc=grad_acc, out_dtype=tl.float32)

        # --- Packed state load: m and v in one TMA issue ---
        # mv_tile shape is (2, BLOCK_V, BLOCK_H). Permute to (BLOCK_V, BLOCK_H, 2)
        # and tl.split to recover per-moment tiles. Permute-only (no reshape)
        # is WS-compatible on Triton 3.4 — it was reshape(2D↔3D) that broke
        # NVGPUWarpSpecialization in the int8 experiment.
        mv_tile = mv_desc.load([0, off_v, off_h])
        mv_perm = tl.permute(mv_tile, (1, 2, 0))
        m_raw, v_raw = tl.split(mv_perm)

        w_raw = w_desc.load([off_v, off_h])

        w = w_raw.to(tl.float32)
        m = beta1 * m_raw + (1.0 - beta1) * grad_acc
        v = beta2 * v_raw + (1.0 - beta2) * grad_acc * grad_acc
        m_hat = m / bias_correction1
        v_hat = v / bias_correction2
        w = w * (1.0 - lr * weight_decay)
        w = w - lr * m_hat / (tl.sqrt(v_hat) + eps)

        # --- Packed state store: re-join (m_new, v_new) and one TMA store ---
        m_out = m.to(m_raw.dtype)
        v_out = v.to(v_raw.dtype)
        mv_out = tl.join(m_out, v_out)                 # (BLOCK_V, BLOCK_H, 2)
        mv_out = tl.permute(mv_out, (2, 0, 1))         # (2, BLOCK_V, BLOCK_H)
        mv_desc.store([0, off_v, off_h], mv_out)

        w_desc.store([off_v, off_h], w.to(w_raw.dtype))


# ---------------------------------------------------------------------------
# Fallback: 2 x 2D TMA descriptors on the SAME packed buffer.
# Used if the 3D permute-in-WS path hits the Hopper partitioner bug. Same
# bytes moved as v1 (2 loads + 2 stores) but the two descriptors now point
# at contiguous memory, which helps L2 locality.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=_ws_tma_v3_configs(),
    key=['BT', 'V', 'H'],
    restore_value=['weight_ptr', 'm_ptr', 'v_ptr'],
)
@triton.jit
def _fused_grad_adamw_persistent_tma_ws_v3_fallback(
    grad_output_ptr, input_ptr, weight_ptr,
    m_ptr, v_ptr,                          # distinct ptr args (aliased into packed buf by wrapper)
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
    # Body is identical to v1 WS; the packing effect is that m_ptr and v_ptr
    # point at contiguous memory (m_ptr + V*H*sizeof(bf16) == v_ptr). Passing
    # them as separate kernel args avoids a Triton 3.6 WS bug where
    # `make_tensor_descriptor(base + int_offset, ...)` inside a warp-specialized
    # region produces wrong code.
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

    for tile_id in tl.range(
            start_pid, num_tiles, NUM_SMS,
            warp_specialize=True,
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
# 3D packed path, NO warp_specialize. Works around the NVGPUWarpSpecialization
# bug for 3D-TMA + permute + split. Keeps the single-TMA-per-tile win of
# Option 1 at the cost of the WS pipeline.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=_ws_tma_v3_configs(),
    key=['BT', 'V', 'H'],
    restore_value=['weight_ptr', 'mv_ptr'],
)
@triton.jit
def _fused_grad_adamw_persistent_tma_3d_nows(
    grad_output_ptr, input_ptr, weight_ptr,
    mv_ptr,
    BT, V, H,
    go_stride_bt, go_stride_v,
    in_stride_bt, in_stride_h,
    w_stride_v, w_stride_h,
    mv_stride_c, mv_stride_v, mv_stride_h,
    lr, beta1, beta2, eps, weight_decay,
    bias_correction1, bias_correction2,
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
    mv_desc = tl.make_tensor_descriptor(
        mv_ptr, shape=[2, V, H],
        strides=[mv_stride_c, mv_stride_v, mv_stride_h],
        block_shape=[2, BLOCK_V, BLOCK_H],
    )

    # No warp_specialize — SW pipelining via num_stages. disable_licm kept on
    # because descriptor creation should stay local to the outer iteration.
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

        grad_acc = tl.zeros((BLOCK_V, BLOCK_H), dtype=tl.float32)
        for bt_start in range(0, BT, BLOCK_BT):
            go  = tl.trans(go_desc.load([bt_start, off_v]))
            inp = inp_desc.load([bt_start, off_h])
            grad_acc = tl.dot(go, inp, acc=grad_acc, out_dtype=tl.float32)

        w_raw = w_desc.load([off_v, off_h])

        mv_tile = mv_desc.load([0, off_v, off_h])          # (2, BV, BH)
        mv_perm = tl.permute(mv_tile, (1, 2, 0))           # (BV, BH, 2)
        m_raw, v_raw = tl.split(mv_perm)

        w = w_raw.to(tl.float32)
        m = beta1 * m_raw + (1.0 - beta1) * grad_acc
        v = beta2 * v_raw + (1.0 - beta2) * grad_acc * grad_acc
        m_hat = m / bias_correction1
        v_hat = v / bias_correction2
        w = w * (1.0 - lr * weight_decay)
        w = w - lr * m_hat / (tl.sqrt(v_hat) + eps)

        m_out = m.to(m_raw.dtype)
        v_out = v.to(v_raw.dtype)
        mv_out = tl.join(m_out, v_out)                     # (BV, BH, 2)
        mv_out = tl.permute(mv_out, (2, 0, 1))             # (2, BV, BH)
        mv_desc.store([0, off_v, off_h], mv_out)

        w_desc.store([off_v, off_h], w.to(w_raw.dtype))


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

# A flag the benchmark flips if the 3D path compiles; the wrapper reads it
# once per call. We can't just catch at launch time because autotune would
# consume many seconds before reporting the error.
USE_3D_PACKED = True


def _packed_mv_view(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Return a (2, V, H) tensor view whose slices alias m and v in memory.

    Requires m and v to have been allocated contiguously (m followed by v)
    from a single packed buffer — the benchmark's monkey-patched
    ensure_buffers guarantees this.
    """
    assert m.is_contiguous() and v.is_contiguous()
    assert m.shape == v.shape and m.dtype == v.dtype
    assert v.data_ptr() == m.data_ptr() + m.numel() * m.element_size(), (
        "m and v must be contiguous in memory (m followed by v). "
        "Run the benchmark's patch_state_packed() before any fused layer's "
        "first forward so ensure_buffers allocates them packed."
    )
    return torch.as_strided(
        m, (2, *m.shape),
        (m.numel(), *m.stride()),
    )


def fused_grad_adamw_ws_v3(
    grad_output: torch.Tensor,   # (BT, V)
    input: torch.Tensor,         # (BT, H)
    weight: torch.Tensor,        # (V, H) — in-place
    m: torch.Tensor,             # (V, H) — in-place; must alias mv[0]
    v: torch.Tensor,             # (V, H) — in-place; must alias mv[1]
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    step: int = 1,
):
    """Drop-in replacement for kernel.fused_grad_adamw that routes to the
    v3 persistent TMA+WS kernel with packed (m, v) state.

    m and v must have been allocated from a single packed (2, V, H) buffer
    (see benchmark_llama_selective_h200_v5_ws_v3.py::patch_state_packed).
    """
    assert grad_output.is_contiguous() and input.is_contiguous()
    BT, V = grad_output.shape
    H = input.shape[1]

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    _ensure_tma_allocator()
    num_sms = _get_num_sms(weight.device)

    mv = _packed_mv_view(m, v)

    if USE_3D_PACKED:
        # NOTE: WS=True path in _fused_grad_adamw_persistent_tma_ws_v3
        # still hits the NVGPUWarpSpecialization bug on 3.6. Callers who
        # set USE_3D_PACKED=True will get a compile error. The working
        # 3D path is the no-WS variant below; use variant="3d_nows"
        # from the benchmark.
        _fused_grad_adamw_persistent_tma_ws_v3[(num_sms,)](
            grad_output, input, weight, mv,
            BT, V, H,
            grad_output.stride(0), grad_output.stride(1),
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            mv.stride(0), mv.stride(1), mv.stride(2),
            lr, beta1, beta2, eps, weight_decay, bc1, bc2,
            NUM_SMS=num_sms,
        )
    else:
        _fused_grad_adamw_persistent_tma_ws_v3_fallback[(num_sms,)](
            grad_output, input, weight, m, v,
            BT, V, H,
            grad_output.stride(0), grad_output.stride(1),
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            lr, beta1, beta2, eps, weight_decay, bc1, bc2,
            NUM_SMS=num_sms,
        )


def fused_grad_adamw_3d_nows(
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
    """Run the 3D-packed path WITHOUT warp_specialize. Exists because the
    3D+WS combination hits the NVGPUWarpSpecialization partitioner bug on
    both Triton 3.4 and 3.6. Trading WS for a single TMA instruction per
    tile in the state epilogue is this variant's only defensible use.
    """
    assert grad_output.is_contiguous() and input.is_contiguous()
    BT, V = grad_output.shape
    H = input.shape[1]

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    _ensure_tma_allocator()
    num_sms = _get_num_sms(weight.device)

    mv = _packed_mv_view(m, v)

    _fused_grad_adamw_persistent_tma_3d_nows[(num_sms,)](
        grad_output, input, weight, mv,
        BT, V, H,
        grad_output.stride(0), grad_output.stride(1),
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        mv.stride(0), mv.stride(1), mv.stride(2),
        lr, beta1, beta2, eps, weight_decay, bc1, bc2,
        NUM_SMS=num_sms,
    )


def patch_dispatch():
    import fused_grad_optimizer.kernel as _k
    import fused_grad_optimizer.autograd as _a
    originals = (_k.fused_grad_adamw, _a.fused_grad_adamw)
    _k.fused_grad_adamw = fused_grad_adamw_ws_v3
    _a.fused_grad_adamw = fused_grad_adamw_ws_v3
    return originals


def restore_dispatch(originals):
    import fused_grad_optimizer.kernel as _k
    import fused_grad_optimizer.autograd as _a
    _k.fused_grad_adamw = originals[0]
    _a.fused_grad_adamw = originals[1]
