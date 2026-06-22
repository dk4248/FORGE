"""
L2 Cache Pinning — Level 2 memory hierarchy optimization.

Provides Python API to pin CUDA tensors in L2 cache so that the fused
grad+optimizer kernel's repeated reads hit L2 (~200 cycles) instead of
HBM (~400 cycles).

The CUDA extension is JIT-compiled on first import.
"""

import os
import logging
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

logger = logging.getLogger(__name__)

# JIT-compile the CUDA extension
_CSRC_DIR = Path(__file__).parent / "csrc"
_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="l2_cache_ext",
            sources=[str(_CSRC_DIR / "l2_cache.cu")],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _ext


class L2CacheManager:
    """
    Manages L2 cache pinning for the fused kernel.

    Usage:
        l2 = L2CacheManager()
        l2.pin(input_tensor)          # hint: keep in L2
        launch_fused_kernel(...)
        l2.unpin()                    # release L2 hints

    Or as a context manager:
        with l2.pinned(input_tensor):
            launch_fused_kernel(...)
    """

    def __init__(self):
        self._ext = _get_ext()
        self._total_l2, self._max_persisting = self._ext.get_l2_info()
        self._is_pinned = False

        logger.info(
            f"L2 cache: {self._total_l2 / 1024**2:.0f} MB total, "
            f"{self._max_persisting / 1024**2:.0f} MB max persisting"
        )

    @property
    def total_l2_bytes(self) -> int:
        return self._total_l2

    @property
    def max_persisting_bytes(self) -> int:
        return self._max_persisting

    def pin(self, *tensors: torch.Tensor, hit_ratio: float = 1.0):
        """
        Pin one or more tensors in L2 cache.

        If multiple tensors are provided, they are pinned sequentially.
        The last pin call's window is what the hardware uses (only one
        window per stream is active). For multiple tensors, we pin the
        largest one that fits, or adjust hit_ratio.
        """
        if not tensors:
            return

        # Pick the best tensor to pin (largest that fits in persisting L2)
        # If only one tensor, pin it directly
        if len(tensors) == 1:
            t = tensors[0]
            if t.nbytes <= self._max_persisting:
                self._ext.pin_l2(t, hit_ratio)
            else:
                # Tensor larger than persisting budget — adjust hit_ratio
                effective_ratio = min(hit_ratio, self._max_persisting / t.nbytes)
                self._ext.pin_l2(t, effective_ratio)
            self._is_pinned = True
            return

        # Multiple tensors: pin the one with highest reuse that fits
        # For our kernel, input (BT, H) is smaller and re-read by every tile
        # Pin the smallest tensor that fits entirely, then the larger one
        sorted_tensors = sorted(tensors, key=lambda t: t.nbytes)
        for t in sorted_tensors:
            if t.nbytes <= self._max_persisting:
                self._ext.pin_l2(t, hit_ratio)
                self._is_pinned = True
                return

        # Nothing fits entirely — pin the smallest with adjusted ratio
        t = sorted_tensors[0]
        effective_ratio = min(hit_ratio, self._max_persisting / t.nbytes)
        self._ext.pin_l2(t, effective_ratio)
        self._is_pinned = True

    def unpin(self):
        """Reset L2 persistence hints."""
        if self._is_pinned:
            self._ext.unpin_l2()
            self._is_pinned = False

    def pinned(self, *tensors: torch.Tensor, hit_ratio: float = 1.0):
        """Context manager: pin on entry, unpin on exit."""
        return _L2PinContext(self, tensors, hit_ratio)

    def info_str(self) -> str:
        """Human-readable L2 info string."""
        return (
            f"L2 total={self._total_l2 / 1024**2:.0f}MB, "
            f"persisting_max={self._max_persisting / 1024**2:.0f}MB"
        )


class _L2PinContext:
    def __init__(self, manager, tensors, hit_ratio):
        self._manager = manager
        self._tensors = tensors
        self._hit_ratio = hit_ratio

    def __enter__(self):
        self._manager.pin(*self._tensors, hit_ratio=self._hit_ratio)
        return self._manager

    def __exit__(self, *exc):
        self._manager.unpin()
        return False
