"""
Fused Gradient + Optimizer: tile-wise weight update without materializing grad_weight.

The core idea: instead of computing grad_W = grad_output.T @ input as a full tensor
and then running optimizer.step(), we process the weight matrix tile-by-tile in Triton.
For each tile, the gradient is accumulated in on-chip memory (registers/SMEM) and
immediately consumed by the optimizer. The full grad_W tensor never exists in HBM.

This trades zero gradient storage for read amplification -- activations are re-read
from HBM once per weight tile. Larger tiles (via SMEM, L2, SM clusters) reduce this
cost. That tradeoff is the key optimization axis.
"""

from fused_grad_optimizer.module import FusedLinear
from fused_grad_optimizer.module import FusedOptimizerManager
from fused_grad_optimizer.state import OptimizerConfig

__all__ = ["FusedLinear", "FusedOptimizerManager", "OptimizerConfig"]
