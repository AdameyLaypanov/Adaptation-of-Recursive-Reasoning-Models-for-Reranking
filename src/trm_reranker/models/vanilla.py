"""Non-recursive transformer cross-encoder baselines.

Same blocks and embedding scheme as the TRM arm (K1 alignment); the only
difference is the application scheme — a plain stack of independent layers.
``num_layers`` selects the arm: shallow param-matched (e.g. 2) or deep
FLOP/depth-matched (e.g. TRM effective depth).
"""

import math
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from .blocks import (
    CastedEmbedding,
    CastedLinear,
    RotaryEmbedding,
    TransformerBlock,
    build_additive_attention_mask,
)


@dataclass
class VanillaRerankerCarry:
    halted: torch.Tensor


@dataclass
class VanillaRerankerConfig:
    batch_size: int
    seq_len: int
    vocab_size: int
    num_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int = 1
    forward_dtype: str = "float32"
    num_segment_types: int = 3


class VanillaRerankerInner(nn.Module):
    def __init__(self, config: VanillaRerankerConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.segment_emb = CastedEmbedding(self.config.num_segment_types, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.score_head = CastedLinear(self.config.hidden_size, 1, bias=True)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_heads,
                    expansion=self.config.expansion,
                    rms_norm_eps=self.config.rms_norm_eps,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

    def _input_embeddings(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(input_ids.to(torch.int64))
        embedding = embedding + self.segment_emb(token_type_ids.to(torch.int64))
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight[: input_ids.shape[1]].to(self.forward_dtype))
        return self.embed_scale * embedding

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        attention_mask = build_additive_attention_mask(batch["attention_mask"], self.forward_dtype)
        hidden_states = self._input_embeddings(batch["input_ids"], batch["token_type_ids"])
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, cos_sin=cos_sin, attention_mask=attention_mask)
        cls_state = hidden_states[:, 0]
        scores = self.score_head(cls_state).squeeze(-1)
        return {"scores": scores}


class VanillaReranker(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = VanillaRerankerConfig(**config_dict)
        self.inner = VanillaRerankerInner(self.config)

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


__all__ = ["VanillaReranker", "VanillaRerankerConfig", "VanillaRerankerInner", "VanillaRerankerCarry"]
