"""Model-specific helpers for applying FORGE to architectures with non-linear custom ops.

For architectures where every trainable layer is nn.Linear (GPT-2, Llama, Qwen, …)
you can use the generic loop in the quickstart directly. The helpers here cover
architectures that mix nn.Linear projections with custom CUDA ops — the custom
ops must be skipped because FusedLinear only wraps nn.Linear weight matrices.

Provided helpers
----------------
wrap_rwkv7(model, ...)
    RWKV-7 ("Goose") — replaces the nn.Linear projections (key, value, receptance,
    output, gate, FFN layers, lm_head) with FusedLinear. The WKV delta-rule kernel
    is a custom CUDA op and is NOT wrapped; it keeps its standard backward pass.
"""

from __future__ import annotations

import torch.nn as nn

from fused_grad_optimizer.module import FusedLinear, FusedOptimizerManager


def wrap_rwkv7(
    model: nn.Module,
    optimizer_type: str = "adamw",
    state_mode: str = "int8",
    **optimizer_kwargs,
) -> tuple[nn.Module, FusedOptimizerManager]:
    """Replace nn.Linear projections in an RWKV-7 model with FusedLinear.

    RWKV-7 blocks contain two layer types:
    - nn.Linear projections (key, value, receptance, output, gate, FFN, lm_head)
      → replaced with FusedLinear; optimizer fused into their backward pass.
    - WKV delta-rule kernel (custom CUDA op, not nn.Linear)
      → untouched; standard backward applies.

    The function walks named_modules() and replaces every nn.Linear it finds.
    Non-Linear modules (WKV, LayerNorm, Embedding) are skipped automatically.

    Args:
        model:          RWKV-7 model as a trainable nn.Module (e.g. from RWKV-PEFT
                        or a HuggingFace-compatible implementation).
        optimizer_type: Optimizer family — "adamw" (default), "sgd", "lion", etc.
                        See FusedLinear for the full list.
        state_mode:     Moment precision — "int8" (default, ~12–13 GB for G1i 2.9B
                        full FT on a 16 GB card), "bf16" (~18 GB), "fp8" (<10 GB,
                        experimental).
        **optimizer_kwargs: Passed through to FusedLinear (e.g. weight_decay=0.1).

    Returns:
        (model, manager): The model with FusedLinear layers in-place, and a
        FusedOptimizerManager that broadcasts lr/step to all fused layers.
        Pass manager.get_non_fused_params() to your regular optimizer for the
        non-Linear parameters (time_decay, time_first, layer norms, embeddings).

    Example::

        model = load_rwkv7_model(checkpoint_path)   # your loading method
        model, manager = wrap_rwkv7(model, optimizer_type="adamw", state_mode="int8")

        non_fused_opt = torch.optim.AdamW(manager.get_non_fused_params(), lr=1e-4)

        for step, batch in enumerate(dataloader):
            manager.pre_step(lr=get_lr(step))
            loss = model(**batch).loss
            loss.backward()          # FusedLinear layers update their weights here
            non_fused_opt.step()
            non_fused_opt.zero_grad()
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        # Locate parent module
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent, attr = model, parts[0]
        else:
            parent = model
            for part in parts[0].split("."):
                parent = getattr(parent, part)
            attr = parts[1]

        setattr(
            parent,
            attr,
            FusedLinear.from_linear(
                module,
                optimizer_type=optimizer_type,
                state_mode=state_mode,
                **optimizer_kwargs,
            ),
        )

    manager = FusedOptimizerManager(model)
    return model, manager
