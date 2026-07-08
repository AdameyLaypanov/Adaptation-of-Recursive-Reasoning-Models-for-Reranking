"""BERT-based arms.

- ``BertTRMReranker``: pretrained encoder feeding the TRM reasoning loop
  (legacy `bert_encoder.ipynb`). ``freeze_encoder`` controls whether the
  encoder is trained.
- ``BertScoringReranker``: pretrained encoder + LayerNorm + dropout + linear
  scoring head (legacy `bert_encoder_only_ablation.ipynb`). With
  ``freeze_encoder=True`` this is the linear-probe competitor (E12a); with
  ``freeze_encoder=False`` it is the fully fine-tuned BERT cross-encoder
  baseline required by E1.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional

import torch
from torch import nn
from transformers import AutoModel

from .blocks import CastedEmbedding, CastedLinear, RotaryEmbedding, build_additive_attention_mask, trunc_normal_init_
from .trm import TRMCarry, TRMInnerCarry, TRMReasoningModule, TRMRerankerConfig, _build_block


@dataclass
class BertTRMRerankerConfig(TRMRerankerConfig):
    encoder_name: Optional[str] = None
    encoder_local_path: Optional[str] = None
    freeze_encoder: bool = True
    add_trm_segment_embeddings: bool = False


def _load_encoder(encoder_name: Optional[str], encoder_local_path: Optional[str]) -> nn.Module:
    encoder_name_or_path = encoder_local_path or encoder_name
    encoder_kwargs = {"local_files_only": True} if encoder_local_path else {}
    encoder = AutoModel.from_pretrained(encoder_name_or_path, **encoder_kwargs)
    if encoder_name is not None:
        encoder.config._name_or_path = encoder_name
    return encoder


class BertTRMRerankerInner(nn.Module):
    def __init__(self, config: BertTRMRerankerConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        if self.config.encoder_name is None and self.config.encoder_local_path is None:
            raise ValueError("BertTRMReranker requires encoder_name or encoder_local_path")
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

        self.segment_emb = CastedEmbedding(self.config.num_segment_types, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.score_head = CastedLinear(self.config.hidden_size, 1, bias=True)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        self.L_level = TRMReasoningModule(layers=[_build_block(self.config) for _ in range(self.config.L_layers)])

        self.register_buffer("H_init", trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.register_buffer("L_init", trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bert_token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bert_token_type_ids is None:
            bert_token_type_ids = (token_type_ids == 2).to(torch.long)
        encoder_kwargs = {
            "input_ids": input_ids.to(torch.long),
            "attention_mask": attention_mask.to(torch.long),
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

    def empty_carry(self, batch_size: int) -> TRMInnerCarry:
        device = self.H_init.device
        return TRMInnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMInnerCarry) -> TRMInnerCarry:
        reset_view = reset_flag.view(-1, 1, 1)
        return TRMInnerCarry(
            z_H=torch.where(reset_view, self.H_init, carry.z_H),
            z_L=torch.where(reset_view, self.L_init, carry.z_L),
        )

    def _l_injection(self, z_H: torch.Tensor, input_embeddings: torch.Tensor) -> torch.Tensor:
        if self.config.disable_input_injection:
            return z_H
        return z_H + input_embeddings

    def forward(self, carry: TRMInnerCarry, batch: Dict[str, torch.Tensor]):
        seq_info = {
            "cos_sin": self.rotary_emb() if hasattr(self, "rotary_emb") else None,
            "attention_mask": build_additive_attention_mask(batch["attention_mask"], self.forward_dtype),
        }
        input_embeddings = self._input_embeddings(
            batch["input_ids"],
            batch["token_type_ids"],
            batch["attention_mask"],
            batch.get("bert_token_type_ids"),
        )
        z_H, z_L = carry.z_H, carry.z_L

        def run_cycles(z_H, z_L, num_cycles):
            for _ in range(num_cycles):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, self._l_injection(z_H, input_embeddings), **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
            return z_H, z_L

        if self.config.full_backprop:
            z_H, z_L = run_cycles(z_H, z_L, self.config.H_cycles)
        else:
            with torch.no_grad():
                z_H, z_L = run_cycles(z_H, z_L, self.config.H_cycles - 1)
            z_H, z_L = run_cycles(z_H, z_L, 1)

        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        cls_state = z_H[:, 0]
        scores = self.score_head(cls_state).squeeze(-1)
        q_logits = self.q_head(cls_state).to(torch.float32)
        return new_carry, scores, (q_logits[..., 0], q_logits[..., 1])


class BertTRMReranker(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = BertTRMRerankerConfig(**config_dict)
        self.inner = BertTRMRerankerInner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TRMCarry:
        batch_size = batch["input_ids"].shape[0]
        return TRMCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=batch["input_ids"].device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=batch["input_ids"].device),
            current_data={key: torch.empty_like(value) for key, value in batch.items()},
        )

    def forward(self, carry: TRMCarry, batch: Dict[str, torch.Tensor]):
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
        new_current_data = {
            key: torch.where(carry.halted.view((-1,) + (1,) * (value.ndim - 1)), batch[key], value)
            for key, value in carry.current_data.items()
        }
        new_inner_carry, scores, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)
        outputs = {
            "scores": scores,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step
            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(
                    new_steps, low=2, high=self.config.halt_max_steps + 1
                )
                halted = halted & (new_steps >= min_halt_steps)
                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits))
                    )
        return TRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs


class BertScoringReranker(nn.Module):
    """Pretrained encoder + linear scoring head.

    freeze_encoder=True  -> linear probe (E12a competitor)
    freeze_encoder=False -> fully fine-tuned BERT cross-encoder baseline (E1)
    """

    def __init__(
        self,
        encoder_name: str = None,
        encoder_local_path: Optional[str] = None,
        freeze_encoder: bool = True,
        dropout: float = 0.1,
        **_ignored,
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

        self.config = SimpleNamespace(
            halt_max_steps=1,
            architecture="bert_scoring_reranker",
            encoder_name=encoder_name,
            encoder_local_path=encoder_local_path,
            freeze_encoder=self.freeze_encoder,
            hidden_size=hidden_size,
            dropout=float(dropout),
        )

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        return None

    def _bert_token_type_ids(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "bert_token_type_ids" in batch:
            return batch["bert_token_type_ids"].to(torch.long)
        if "token_type_ids" in batch:
            token_type_ids = batch["token_type_ids"].to(torch.long)
            return (token_type_ids == 2).to(torch.long)
        return torch.zeros_like(batch["input_ids"], dtype=torch.long)

    def forward(self, carry, batch: Dict[str, torch.Tensor]):
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


__all__ = ["BertTRMReranker", "BertTRMRerankerConfig", "BertScoringReranker"]
