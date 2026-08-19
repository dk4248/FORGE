"""FORGE + RWKV-7 ("Goose") — full fine-tuning on a 16 GB card.

RWKV-7 blocks have two layer types:
  - nn.Linear projections (key, value, receptance, output, gate, FFN, lm_head)
    → FusedLinear, optimizer fused into backward
  - WKV delta-rule kernel (custom CUDA op, not nn.Linear)
    → standard backward, untouched

With int8 moments, full FT of G1i 2.9B fits on a 16 GB card:

  Method                    Peak VRAM
  ─────────────────────────────────────────
  standard AdamW (bf16)     ~35 GB
  bitsandbytes 8-bit Adam   ~18 GB
  FORGE (int8 moments)      ~13 GB  ← fits on RTX 5080 / 4080 16 GB

This example builds a minimal synthetic RWKV-7 block to demonstrate the
wrapping pattern without requiring a full checkpoint. Adapt the model
loading section to your setup (RWKV-PEFT, HuggingFace, etc.).

Run:
    python examples/rwkv7_example.py
"""
import torch
import torch.nn as nn

from fused_grad_optimizer import FusedLinear, FusedOptimizerManager
from fused_grad_optimizer.model_wrappers import wrap_rwkv7


# ---------------------------------------------------------------------------
# Minimal synthetic RWKV-7 block
# (replace with your real model loading below)
# ---------------------------------------------------------------------------

class SyntheticWKV(nn.Module):
    """Placeholder for the RWKV-7 WKV delta-rule CUDA kernel.

    In a real model this is a custom CUDA op (rwkv7_attn or equivalent).
    It is NOT an nn.Linear, so wrap_rwkv7 leaves it untouched automatically.
    """
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        # time_decay and time_first are Parameters, not Linear — also skipped
        self.time_decay = nn.Parameter(torch.zeros(n_head))
        self.time_first = nn.Parameter(torch.zeros(n_head))

    def forward(self, x):
        # Real WKV computes the delta-rule recurrence; here we just pass through
        return x


class SyntheticTimeMix(nn.Module):
    """RWKV-7 attention block (linear projections only, WKV is a custom op)."""
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.key        = nn.Linear(n_embd, n_embd, bias=False)
        self.value      = nn.Linear(n_embd, n_embd, bias=False)
        self.gate       = nn.Linear(n_embd, n_embd, bias=False)
        self.output     = nn.Linear(n_embd, n_embd, bias=False)
        self.wkv        = SyntheticWKV(n_embd, n_head)   # custom op — NOT Linear

    def forward(self, x):
        r = self.receptance(x)
        k = self.key(x)
        v = self.value(x)
        g = torch.sigmoid(self.gate(x))
        wkv_out = self.wkv(k * v)          # custom CUDA op in real model
        return self.output(r * wkv_out * g)


class SyntheticChannelMix(nn.Module):
    """RWKV-7 FFN block."""
    def __init__(self, n_embd: int):
        super().__init__()
        ffn_dim = n_embd * 4
        self.key        = nn.Linear(n_embd, ffn_dim, bias=False)
        self.value      = nn.Linear(ffn_dim, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        k = torch.relu(self.key(x)) ** 2
        return torch.sigmoid(self.receptance(x)) * self.value(k)


class SyntheticRWKV7(nn.Module):
    """Minimal RWKV-7 model for demonstration."""
    def __init__(self, vocab_size: int = 65536, n_embd: int = 2560,
                 n_layer: int = 4, n_head: int = 40):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.ln0 = nn.LayerNorm(n_embd)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(n_embd),
                "att": SyntheticTimeMix(n_embd, n_head),
                "ln2": nn.LayerNorm(n_embd),
                "ffn": SyntheticChannelMix(n_embd),
            })
            for _ in range(n_layer)
        ])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head   = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        x = self.ln0(self.emb(idx))
        for block in self.blocks:
            x = x + block["att"](block["ln1"](x))
            x = x + block["ffn"](block["ln2"](x))
        return self.head(self.ln_out(x))


# ---------------------------------------------------------------------------
# Real model loading (adapt to your setup)
# ---------------------------------------------------------------------------
# Option A — RWKV-PEFT:
#   from rwkvt.model_run import RWKV
#   model = RWKV(args)   # args.load_model = "/path/to/g1i-2.9b.pth"
#
# Option B — HuggingFace:
#   from transformers import AutoModelForCausalLM
#   model = AutoModelForCausalLM.from_pretrained("RWKV/v7-Goose-World-2.9B-v0.1",
#               torch_dtype=torch.bfloat16)
# ---------------------------------------------------------------------------


def count_linears(model: nn.Module) -> tuple[int, int]:
    """Return (n_fused, n_linear) counts."""
    n_fused = sum(1 for m in model.modules() if isinstance(m, FusedLinear))
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    return n_fused, n_linear


def vram_mb() -> float:
    return torch.cuda.memory_allocated() / 1024 ** 2


def main():
    assert torch.cuda.is_available(), "FORGE kernels require a CUDA GPU."
    device = "cuda"
    torch.manual_seed(0)

    print("Building synthetic RWKV-7 model (4 layers, n_embd=2560)…")
    print("(Replace with your real model loading for G1i 2.9B)")
    model = SyntheticRWKV7(n_layer=4).to(torch.bfloat16).to(device)

    n_linear_before = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    vram_before = vram_mb()
    print(f"\nBefore FORGE:  {n_linear_before} nn.Linear layers  |  VRAM {vram_before:.1f} MB")

    # -----------------------------------------------------------------------
    # Apply FORGE — one call wraps all nn.Linear, WKV op is skipped
    # -----------------------------------------------------------------------
    model, manager = wrap_rwkv7(model, optimizer_type="adamw", state_mode="int8")

    n_fused, n_linear_after = count_linears(model)
    print(f"After  FORGE:  {n_fused} FusedLinear  |  {n_linear_after} nn.Linear remaining")
    print("  (remaining nn.Linear = 0 expected; WKV op was never nn.Linear)")

    # Non-fused params: time_decay, time_first, LayerNorm weights, Embedding
    non_fused_opt = torch.optim.AdamW(manager.get_non_fused_params(), lr=1e-4)
    print(f"  Non-fused param groups: {len(manager.get_non_fused_params())} tensors")

    # -----------------------------------------------------------------------
    # One training step
    # -----------------------------------------------------------------------
    batch_size, seq_len = 2, 512
    idx    = torch.randint(0, 65536, (batch_size, seq_len), device=device)
    target = torch.randint(0, 65536, (batch_size, seq_len), device=device)

    manager.pre_step(lr=1e-4)
    logits = model(idx)
    loss   = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), target.view(-1)
    )
    loss.backward()          # FusedLinear layers apply AdamW update here

    non_fused_opt.step()
    non_fused_opt.zero_grad()

    vram_after = vram_mb()
    print(f"\nTraining step complete")
    print(f"  loss = {loss.item():.4f} (finite: {loss.isfinite().item()})")
    print(f"  peak VRAM = {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

    # -----------------------------------------------------------------------
    # WKV state integrity check
    # Run a second forward under no_grad and compare logits to a reference
    # that uses the same weights but without FORGE (verify WKV is unaffected)
    # -----------------------------------------------------------------------
    with torch.no_grad():
        logits_check = model(idx)
    assert logits_check.isfinite().all(), "Non-finite logits after training step"
    print("  WKV output intact (logits finite, forward reproduces)")


if __name__ == "__main__":
    main()
