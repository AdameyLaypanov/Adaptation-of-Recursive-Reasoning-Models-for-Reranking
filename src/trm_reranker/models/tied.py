"""Weight-tied deep transformer (ALBERT arm, experiment E3).

Cross-layer parameter sharing without the TRM machinery: no two latent states,
no input re-injection — a block stack of ``num_layers`` unique layers applied
``num_repeats`` times. Together with E5 this decomposes what exactly in TRM
contributes on top of plain tying.

Effective depth = ``num_layers * num_repeats``; unique body params match a
``num_layers``-deep vanilla model.
"""

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from .vanilla import VanillaRerankerCarry, VanillaRerankerConfig, VanillaRerankerInner


@dataclass
class TiedRerankerConfig(VanillaRerankerConfig):
    num_repeats: int = 1


class TiedRerankerInner(VanillaRerankerInner):
    def __init__(self, config: TiedRerankerConfig) -> None:
        super().__init__(config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        from .blocks import build_additive_attention_mask

        attention_mask = build_additive_attention_mask(batch["attention_mask"], self.forward_dtype)
        hidden_states = self._input_embeddings(batch["input_ids"], batch["token_type_ids"])
        for _ in range(self.config.num_repeats):
            for layer in self.layers:
                hidden_states = layer(hidden_states=hidden_states, cos_sin=cos_sin, attention_mask=attention_mask)
        cls_state = hidden_states[:, 0]
        scores = self.score_head(cls_state).squeeze(-1)
        return {"scores": scores}


class TiedReranker(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TiedRerankerConfig(**config_dict)
        self.inner = TiedRerankerInner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> VanillaRerankerCarry:
        batch_size = batch["input_ids"].shape[0]
        return VanillaRerankerCarry(
            halted=torch.ones((batch_size,), dtype=torch.bool, device=batch["input_ids"].device),
        )

    def forward(self, carry: VanillaRerankerCarry, batch: Dict[str, torch.Tensor]):
        del carry
        outputs = self.inner(batch)
        batch_size = batch["input_ids"].shape[0]
        new_carry = VanillaRerankerCarry(
            halted=torch.ones((batch_size,), dtype=torch.bool, device=batch["input_ids"].device),
        )
        return new_carry, outputs


__all__ = ["TiedReranker", "TiedRerankerConfig", "TiedRerankerInner"]
