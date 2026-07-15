"""BERT-based variants.

- ``BertTRMReranker``: pretrained encoder feeding the TRM reasoning loop
  (legacy `bert_encoder.ipynb`). ``freeze_encoder`` controls whether the
  encoder is trained. The recursion/ACT machinery is inherited from the TRM
  variant; only the input-embedding path differs.
- ``BertScoringReranker``: pretrained encoder + LayerNorm + dropout + linear
  scoring head (legacy `bert_encoder_only_ablation.ipynb`). With
  ``freeze_encoder=True`` this is the linear-probe competitor (E12a); with
  ``freeze_encoder=False`` it is the fully fine-tuned BERT cross-encoder
  baseline required by E1.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel

from .blocks import CastedEmbedding, CastedLinear, RotaryEmbedding
from .trm import RecursiveACTReranker, RecursiveInnerBase, TRMReasoningModule, TRMRerankerConfig, _build_block


@dataclass
class BertTRMRerankerConfig(TRMRerankerConfig):
    encoder_name: str | None = None
    encoder_local_path: str | None = None
    freeze_encoder: bool = True
    add_trm_segment_embeddings: bool = False


def _load_encoder(encoder_name: str | None, encoder_local_path: str | None) -> nn.Module:
    if encoder_name is None and encoder_local_path is None:
        raise ValueError("BERT variants require encoder_name or encoder_local_path")
    encoder_name_or_path = encoder_local_path or encoder_name
    encoder_kwargs = {"local_files_only": True} if encoder_local_path else {}
    encoder = AutoModel.from_pretrained(encoder_name_or_path, **encoder_kwargs)
    if encoder_name is not None:
        encoder.config._name_or_path = encoder_name
    return encoder


class BertTRMRerankerInner(RecursiveInnerBase):
    def __init__(self, config: BertTRMRerankerConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.encoder = _load_encoder(self.config.encoder_name, self.config.encoder_local_path)
        if self.config.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        encoder_hidden_size = int(self.encoder.config.hidden_size)
        if encoder_hidden_size != self.config.hidden_size:
            self.encoder_proj = CastedLinear(encoder_hidden_size, self.config.hidden_size, bias=False)
        else:
            self.encoder_proj = nn.Identity()
        self.encoder_norm = nn.LayerNorm(self.config.hidden_size)

        self.segment_emb = CastedEmbedding(
            self.config.num_segment_types, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.score_head = CastedLinear(self.config.hidden_size, 1, bias=True)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
            )

        self.L_level = TRMReasoningModule(layers=[_build_block(self.config) for _ in range(self.config.L_layers)])
        self._register_recursive_state()

    def _compute_input_embeddings(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        bert_token_type_ids = batch.get("bert_token_type_ids")
        if bert_token_type_ids is None:
            bert_token_type_ids = (token_type_ids == 2).to(torch.long)
        encoder_kwargs = {
            "input_ids": input_ids.to(torch.long),
            "attention_mask": batch["attention_mask"].to(torch.long),
            "token_type_ids": bert_token_type_ids.to(torch.long),
        }
        if self.config.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                encoder_states = self.encoder(**encoder_kwargs).last_hidden_state
        else:
            encoder_states = self.encoder(**encoder_kwargs).last_hidden_state
        embedding = self.encoder_proj(encoder_states.to(self.forward_dtype))
        if self.config.add_trm_segment_embeddings:
            embedding = embedding + self.segment_emb(token_type_ids.to(torch.long))
        embedding = self.encoder_norm(embedding.float()).to(self.forward_dtype)
        return embedding


class BertTRMReranker(RecursiveACTReranker):
    config_cls = BertTRMRerankerConfig
    inner_cls = BertTRMRerankerInner


class BertScoringReranker(nn.Module):
    """Pretrained encoder + linear scoring head.

    freeze_encoder=True  -> linear probe (E12a competitor)
    freeze_encoder=False -> fully fine-tuned BERT cross-encoder baseline (E1)
    """

    def __init__(
        self,
        encoder_name: str | None = None,
        encoder_local_path: str | None = None,
        freeze_encoder: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = _load_encoder(encoder_name, encoder_local_path)
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

        hidden_size = int(self.encoder.config.hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.score_head = nn.Linear(hidden_size, 1)

    def initial_carry(self, batch: dict[str, torch.Tensor]):
        return None

    def _bert_token_type_ids(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "bert_token_type_ids" in batch:
            return batch["bert_token_type_ids"].to(torch.long)
        if "token_type_ids" in batch:
            token_type_ids = batch["token_type_ids"].to(torch.long)
            return (token_type_ids == 2).to(torch.long)
        return torch.zeros_like(batch["input_ids"], dtype=torch.long)

    def forward(self, carry, batch: dict[str, torch.Tensor]):
        encoder_kwargs = {
            "input_ids": batch["input_ids"].to(torch.long),
            "attention_mask": batch["attention_mask"].to(torch.long),
            "token_type_ids": self._bert_token_type_ids(batch),
        }
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                encoder_outputs = self.encoder(**encoder_kwargs)
        else:
            encoder_outputs = self.encoder(**encoder_kwargs)

        cls_state = encoder_outputs.last_hidden_state[:, 0]
        cls_state = self.norm(cls_state)
        cls_state = self.dropout(cls_state)
        scores = self.score_head(cls_state).squeeze(-1)
        return None, {"scores": scores}


__all__ = ["BertScoringReranker", "BertTRMReranker", "BertTRMRerankerConfig", "BertTRMRerankerInner"]
