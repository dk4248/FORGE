"""
B200-specific TMA + warp-specialized persistent fused grad+AdamW kernel
(Blackwell, sm_100a) for Triton 3.6.0.

Why a separate file
-------------------
The kernel.py WS path (`_fused_grad_adamw_persistent_tma_ws`) crashes
Triton 3.6.0's MLIR pipeline on sm_100 with:

    PartitionScheduling.cpp:592:
        assignMissingPartitions(...)(ttng::TMEMAllocOp):
        Assertion `mmaPartitionId && "mma must have a partition"' failed.

This is triton-lang/triton#8932 (open, slated for 3.7). The H200 path works
because Hopper WGMMA keeps the fp32 accumulator in registers; Blackwell's
tcgen05.mma stores it in Tensor Memory (TMEM), and the 3.6.0
`TritonGPUAutomaticWarpSpecialization` / `TritonGPUPartitionScheduling`
pipeline fails to associate the `ttng::TMEMAllocOp` with its MMA op under
the default Hopper-ish config that kernel.py uses.

Workarounds stacked here (each is independently documented in triton#8932,
triton#8260, and upstream WS PRs #5622 / #6288 / #8534):

  1. `num_warps=8` — per #8260, the auto-WS pass is only reliable with 8
     warps on Blackwell. Hopper WS needs 4; Blackwell WS needs 8. This is
     why we can't reuse kernel.py's `_ws_tma_configs` here.

  2. `num_stages=2` — deeper pipelines (3+) on this kernel on sm_100 trigger
     the TMEM-alloc code path that lacks a partition id. We keep 3 as a
     secondary config for autotune to try if 2 recompiles cleanly.

  3. `input_precision="ieee"` on `tl.dot` — the default `"tf32"` on an
     otherwise-bf16 dot emits a secondary tcgen05.mma that the WS scheduler
     fails to annotate. Setting `"ieee"` keeps one clean bf16 mma.

  4. `disallow_acc_multi_buffer=True` on the outer `tl.range` — prevents WS
     from double-buffering the fp32 accumulator across iterations. With one
     TMEMAlloc per MMA the partitioner's invariant holds.

  5. `disable_licm=True` on the outer `tl.range` — keeps TMA-descriptor
     creation local to the iter, preventing the scheduler from hoisting
     descriptors into registers that WS wants for MMA operands.

If WS still asserts at compile time, `fused_grad_adamw_b200_ws` auto-falls
back to the TMA-no-WS kernel in the same module, so the benchmark completes
instead of dying in the autotune warmup. The fallback is cached module-level
so we pay the compile-failure cost at most once.

Retire this file when triton>=3.7 is available and kernel.py's WS path
compiles cleanly on sm_100a.
"""

import logging
import torch
import triton
import triton.language as tl

from fused_grad_optimizer.kernel import _ensure_tma_allocator, _get_num_sms

log = logging.getLogger("kernel_b200_ws")

# Set to True after a compile fails, so subsequent calls skip WS.
_FALLBACK_TO_NOWS = False


def _b200_ws_configs():
    """B200 WS autotune configs. See module docstring for the num_warps=8 /
    num_stages=2 rationale.
    """
    return [
        # Primary: num_warps=8, num_stages=2 — the combo #8260 recommends.
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64,
                       'GROUP_SIZE_V': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 128,
                       'GROUP_SIZE_V': 8}, num_warps=8, num_stages=2),
        # Secondary: try num_stages=3 in case the partitioner tolerates it
        # for some shapes. Autotune will drop it if it fails to compile.
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64,
                       'GROUP_SIZE_V': 8}, num_warps=8, num_stages=3),
    ]


def _b200_nows_configs():
    """Configs for the no-WS fallback kernel (if WS compile asserts).

    NOTE: we tried matching v1/v2's full `_fused_configs()` (~30 configs)
    here to make memory comparison apples-to-apples. Two findings:
      1. The +1 GB peak on v3 vs v1/v2 is NOT from these configs — it's
         from the WS compile attempt itself pinning state in Triton's
         autotune cache before failing with #8932. Matching configs
         didn't change the +1 GB peak.
      2. Matching configs made v3 ~10% SLOWER because autotune picked a
         different "best" config from the bigger set than v1/v2's
         persistent_tma kernel does for the same shape.
    Keep the small num_warps=8/num_stages=3 set — it's tuned for the
    fallback path's specific behavior even if it adds ~1 GB.
    """
    return [
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64,
                       'GROUP_SIZE_V': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64,
                       'GROUP_SIZE_V': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 128,
                       'GROUP_SIZE_V': 8}, num_warps=8, num_stages=3),
    ]


# -----------------------------------------------------------------------------
# WS kernel (primary path)
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=_b200_ws_configs(),
    key=['BT', 'V', 'H'],
    restore_value=['weight_ptr', 'm_ptr', 'v_ptr'],
)
@triton.jit
def _fused_grad_adamw_persistent_tma_ws_b200(
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
    """Persistent TMA fused grad+AdamW with WS on Blackwell (sm_100a)."""
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
            disallow_acc_multi_buffer=True,  # one TMEMAlloc per MMA on sm_100
            disable_licm=True,               # keep descriptor creation local
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
            # input_precision="ieee" keeps one clean bf16 mma; default "tf32"
            # on sm_100 emits a secondary tcgen05.mma that confuses WS.
            grad_acc = tl.dot(go, inp, acc=grad_acc, out_dtype=tl.float32,
                              input_precision="ieee")

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


# -----------------------------------------------------------------------------
# No-WS fallback kernel (used if WS still asserts at compile time).
# Same body, warp_specialize=False, num_warps=8, deeper pipeline stages.
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=_b200_nows_configs(),
    key=['BT', 'V', 'H'],
    restore_value=['weight_ptr', 'm_ptr', 'v_ptr'],
)
@triton.jit
def _fused_grad_adamw_persistent_tma_nows_b200(
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
            grad_acc = tl.dot(go, inp, acc=grad_acc, out_dtype=tl.float32,
                              input_precision="ieee")

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


def _launch(kernel, grad_output, input, weight, m, v,
            lr, beta1, beta2, eps, weight_decay, bc1, bc2):
    BT, V = grad_output.shape
    H = input.shape[1]
    _ensure_tma_allocator()
    num_sms = _get_num_sms(weight.device)
    kernel[(num_sms,)](
        grad_output, input, weight, m, v,
        BT, V, H,
        grad_output.stride(0), grad_output.stride(1),
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        lr, beta1, beta2, eps, weight_decay, bc1, bc2,
        NUM_SMS=num_sms,
    )


def fused_grad_adamw_b200_ws(
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
    """Drop-in replacement for fused_grad_optimizer.kernel.fused_grad_adamw.

    Tries the WS kernel first; if it asserts in Triton's MLIR pipeline
    (triton#8932), caches that failure and uses the no-WS kernel for the
    rest of the process lifetime.
    """
    global _FALLBACK_TO_NOWS
    assert grad_output.is_contiguous() and input.is_contiguous()
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    if not _FALLBACK_TO_NOWS:
        try:
            _launch(_fused_grad_adamw_persistent_tma_ws_b200,
                    grad_output, input, weight, m, v,
                    lr, beta1, beta2, eps, weight_decay, bc1, bc2)
            return
        except (RuntimeError, AssertionError) as e:
            # Autotune / MLIR compile failure → fall back for this process.
            msg = str(e)
            if ("PassManager::run failed" in msg
                or "mma must have a partition" in msg
                or "WarpSpecialization" in msg):
                log.warning("B200 WS kernel failed to compile (triton#8932); "
                            "falling back to TMA-no-WS for this process.")
                _FALLBACK_TO_NOWS = True
            else:
                raise

    _launch(_fused_grad_adamw_persistent_tma_nows_b200,
            grad_output, input, weight, m, v,
            lr, beta1, beta2, eps, weight_decay, bc1, bc2)


def patch_dispatch():
    """Install fused_grad_adamw_b200_ws as the backward kernel.

    Returns the previous pair so callers can restore it with
    `restore_dispatch(prev)`.
    """
    import fused_grad_optimizer.kernel as _k
    import fused_grad_optimizer.autograd as _a
    prev = (_k.fused_grad_adamw, _a.fused_grad_adamw)
    _k.fused_grad_adamw = fused_grad_adamw_b200_ws
    _a.fused_grad_adamw = fused_grad_adamw_b200_ws
    return prev


def restore_dispatch(prev):
    import fused_grad_optimizer.kernel as _k
    import fused_grad_optimizer.autograd as _a
    _k.fused_grad_adamw, _a.fused_grad_adamw = prev
