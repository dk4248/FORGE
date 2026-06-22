"""Minimal runnable FORGE example.

Trains a tiny MLP on a synthetic regression task, with its linear layers stepped
by FORGE (the AdamW update is fused into the backward pass, tile-by-tile, so the
weight gradients are never materialized in HBM).

Run on a CUDA GPU:
    python examples/quickstart.py
"""
import torch
import torch.nn as nn

from fused_grad_optimizer import FusedLinear, FusedOptimizerManager


def main():
    assert torch.cuda.is_available(), "FORGE kernels require a CUDA GPU."
    device = "cuda"
    torch.manual_seed(0)

    # A small MLP.
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.GELU(),
        nn.Linear(1024, 1024),
        nn.GELU(),
        nn.Linear(1024, 512),
    ).to(device)

    # Swap every nn.Linear for a FusedLinear (AdamW fused into backward).
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            model[i] = FusedLinear.from_linear(layer, optimizer_type="adamw")
    model.to(device)

    # The manager broadcasts the LR/step to the fused layers each step; a standard
    # optimizer handles any non-fused params (none here, but this is the pattern).
    manager = FusedOptimizerManager(model)
    non_fused = manager.get_non_fused_params()
    optimizer = torch.optim.AdamW(non_fused, lr=1e-3) if non_fused else None

    # Synthetic target.
    X = torch.randn(64, 512, device=device)
    target = torch.randn(64, 512, device=device)

    for step in range(200):
        manager.pre_step(lr=1e-3)
        out = model(X)
        loss = ((out - target) ** 2).mean()
        loss.backward()          # <-- FORGE applies AdamW to each linear here
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad()
        if step % 40 == 0:
            print(f"step {step:3d}  loss {loss.item():.5f}")

    print(f"final loss {loss.item():.5f}  (it should have dropped steadily)")


if __name__ == "__main__":
    main()
