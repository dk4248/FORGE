# Contributing to FORGE

Thanks for your interest in improving FORGE! Contributions of all kinds are welcome —
bug reports, new optimizer kernels, additional GPU backends, docs, and benchmarks.

## Getting started

```bash
git clone https://github.com/dk4248/FORGE.git
cd FORGE
pip install -e ".[test]"
pytest tests/ -v        # requires a CUDA GPU
```

## Guidelines

- **Correctness first.** Any new fused kernel must match a reference PyTorch
  implementation of the same optimizer within floating-point round-off. Add a
  test in `tests/` (see `tests/test_correctness.py` for the pattern).
- **Keep the public API stable.** The exported surface is `FusedLinear`,
  `FusedOptimizerManager`, and `OptimizerConfig` (`src/fused_grad_optimizer/__init__.py`).
- **Element-wise rule.** FORGE fuses optimizers whose update factors element-wise.
  Cross-element preconditioners (Muon, Shampoo) are out of scope for the fused path.
- **Hardware notes.** State which GPU / CUDA / Triton versions you tested on in your PR.

## Good first contributions

- A new element-wise optimizer kernel (NAdam, AdaGrad variants, …).
- An AMD/ROCm or Apple-Metal backend (currently unvalidated).
- Distributed (FSDP / DDP) integration — gradients are consumed in place, so
  the all-reduce needs rethinking.

## Reporting bugs

Open an issue with: GPU model, CUDA/PyTorch/Triton versions, a minimal repro, and
the full traceback.
