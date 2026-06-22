"""
Triton autotune re-tuning for H200 (sm_90a).

Call tune_for_h200() ONCE before any training step runs.  It:
  1) wipes the Triton on-disk cache (~/.triton) so stale sm_120-era
     pre-compiled kernels don't get hit;
  2) replaces the autotune config list on the existing
     fused_grad_optimizer.kernel._fused_grad_adamw_kernel (and the
     companion SGD / int8 / persistent kernels) with H200-tuned configs.

Why re-tune on H200
-------------------
The configs in kernel.py were picked for sm_120 (188 SMs, 128 MB L2, no
TMA, no WGMMA).  H200 is different in three ways that matter for config
choice:

  * 132 SMs (fewer).  Prefer enough tiles to cover all SMs but not so many
    that we pay launch+schedule cost without saturating bandwidth.
  * 60 MB L2 (less than half of Blackwell workstation).  GROUP_SIZE_V>8
    on H200 tends to cause L2 thrash, not reuse -- so we mostly stay at 8.
  * HBM3e at ~4.8 TB/s.  Deeper pipelines (num_stages=4) hide the longer
    HBM latency better than num_stages=2 that works on GDDR7.
  * Wider SM (228 KB SMEM, same as Blackwell) + larger register file still
    afford 128x128 and 256x128 at num_stages=4.

The configs below were biased toward num_stages>=3 and BLOCK_BT=64 with a
sprinkling of BLOCK_BT=128 for the large-K case (SEQ_LEN=4096 now).
"""

import os
import shutil
import triton


def _h200_fused_configs():
    """H200-tuned autotune configs for fused grad_W+AdamW kernel.

    Pruned to the three tiles that ever win on Llama 8B at BT=4096:

        * 128x128, BT=64, warps=8, stages=4   — wins lm_head in some runs
        * 128x128, BT=64, warps=8, stages=5   — wins k/v and gate/up_proj
        * 128x128, BT=64, warps=8, stages=6   — wins down_proj and q/o_proj

    These three stages values are within Triton autotune's timing noise for
    most shapes — pick per-shape flips run-to-run. Total time is stable at
    ~359 ms bwd regardless. Below this list is the config-exploration
    landscape we considered (all pruned):

      Dead on this kernel (never won across runs):
        - BT=128 variants      (larger K-chunk didn't help)
        - BLOCK_V=256 tiles    (register spill / occupancy loss)
        - BLOCK_H=256 tiles    (same)
        - BLOCK_V=64  tiles    (too-small grid, tile overhead dominates)
        - GROUP_SIZE_V != 8    (didn't help L2 reuse here)

    Further Triton-side wins require structural changes (TMA path enabled,
    different dispatch strategy) — autotune can't find them.
    """
    return [
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64,  'GROUP_SIZE_V': 8},
                      num_warps=8, num_stages=4),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64,  'GROUP_SIZE_V': 8},
                      num_warps=8, num_stages=5),
        triton.Config({'BLOCK_V': 128, 'BLOCK_H': 128, 'BLOCK_BT': 64,  'GROUP_SIZE_V': 8},
                      num_warps=8, num_stages=6),
    ]


def _h200_optimizer_only_configs():
    """Optimizer-only (memory-bound) is architecture-insensitive, keep small."""
    return [
        triton.Config({'BLOCK_R': 64,  'BLOCK_C': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_R': 128, 'BLOCK_C': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_R': 128, 'BLOCK_C': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_R': 256, 'BLOCK_C': 128}, num_warps=8, num_stages=1),
    ]


def _wipe_triton_cache():
    """Remove the Triton on-disk cache so no sm_120-era artefacts are reused."""
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".triton", "cache"),
        os.path.join(home, ".triton"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
            print(f"[kernel_h200_tune] wiped {p}")


def _replace_configs_on(kernel_obj, new_configs):
    """Reach inside a triton.autotune()-decorated kernel and swap its configs.

    triton.autotune returns an Autotuner object with a `.configs` attribute
    (a list of triton.Config).  We overwrite it in-place so subsequent
    launches re-autotune against the new config list.  We also clear the
    memoised best-config cache so the first call triggers a fresh search.
    """
    if not hasattr(kernel_obj, "configs"):
        return False
    kernel_obj.configs = list(new_configs)
    # The Autotuner caches best config per (key, config-hash) tuple.  Clear it.
    for attr in ("cache", "configs_timings"):
        if hasattr(kernel_obj, attr):
            try:
                getattr(kernel_obj, attr).clear()
            except Exception:
                pass
    return True


def tune_for_h200(verbose: bool = True):
    """Install H200-specific autotune configs on all fused kernels.

    Call this ONCE, before any training step runs.  Idempotent.
    """
    _wipe_triton_cache()

    from fused_grad_optimizer import kernel as _k

    # The fused grad+optimizer kernels that use @triton.autotune
    fused_cfgs = _h200_fused_configs()
    opt_cfgs = _h200_optimizer_only_configs()

    targets_fused = [
        getattr(_k, name, None) for name in (
            "_fused_grad_adamw_kernel",
            "_fused_grad_sgd_kernel",
            "_fused_grad_adamw_int8state_kernel",
        )
    ]
    targets_opt = [
        getattr(_k, name, None) for name in (
            "_optimizer_only_adamw_kernel",
            "_optimizer_only_sgd_kernel",
        )
    ]

    patched = 0
    for t in targets_fused:
        if t is not None and _replace_configs_on(t, fused_cfgs):
            patched += 1
    for t in targets_opt:
        if t is not None and _replace_configs_on(t, opt_cfgs):
            patched += 1

    if verbose:
        print(f"[kernel_h200_tune] patched {patched} autotune targets "
              f"with H200 configs ({len(fused_cfgs)} fused, "
              f"{len(opt_cfgs)} opt-only)")
    return patched
