"""TRM/HRM-derived two-state recurrent cross-encoder.

Faithful port of `TinyRecursiveReasoningModel_ACTV1` from the legacy notebooks
(numerics unchanged for the default flags), plus two experiment switches:

- ``disable_input_injection`` (E5): drop the ``+ input_embeddings`` term in the
  z_L update, isolating the contribution of input re-injection.
- ``full_backprop`` (E0 follow-up): backpropagate through all H-cycles instead
  of the HRM one-step gradient approximation (default keeps the original
  behaviour: the first ``H_cycles - 1`` cycles run under ``no_grad``).

The recursion/ACT machinery shared with the BERT-encoder variant lives in
``RecursiveInnerBase`` (two-state cycles, carry, heads) and
``RecursiveACTReranker`` (halting wrapper); subclasses only define how input
embeddings are produced.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from .blocks import (
    CastedEmbedding,
    CastedLinear,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    TransformerBlock,
    build_additive_attention_mask,
    rms_norm,
    trunc_normal_init_,
)


@dataclass
class TRMInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TRMCarry:
    inner_carry: TRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: dict[str, torch.Tensor]


@dataclass
class TRMRerankerConfig:
    batch_size: int
    seq_len: int
    vocab_size: int
    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int = 1
    halt_exploration_prob: float = 0.0
    forward_dtype: str = "float32"
    mlp_t: bool = False
    no_ACT_continue: bool = True
    num_segment_types: int = 3
    disable_input_injection: bool = False
    full_backprop: bool = False


class MLPTBlock(nn.Module):
    """Token-mixing SwiGLU over the sequence dimension (TRM `mlp_t` variant)."""

    def __init__(self, config: TRMRerankerConfig) -> None:
        super().__init__()
        self.mlp_t = SwiGLU(hidden_size=config.seq_len, expansion=config.expansion)
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(
        self, hidden_states: torch.Tensor, cos_sin: CosSin = None, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        del cos_sin, attention_mask
        hidden_states = hidden_states.transpose(1, 2)
        out = self.mlp_t(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        hidden_states = hidden_states.transpose(1, 2)
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


def _build_block(config: TRMRerankerConfig) -> nn.Module:
    if config.mlp_t:
        return MLPTBlock(config)
    return TransformerBlock(
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        expansion=config.expansion,
        rms_norm_eps=config.rms_norm_eps,
    )


class TRMReasoningModule(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class RecursiveInnerBase(nn.Module):
    """Two-state (z_H, z_L) recursion shared by the TRM and BERT-TRM variants.

    Subclasses build their own embedding path in ``__init__`` (finishing with
    ``_register_recursive_state()`` so buffer/RNG order stays stable) and
    implement ``_compute_input_embeddings``.
    """

    config: TRMRerankerConfig

    def _register_recursive_state(self) -> None:
        self.register_buffer(
            "H_init",
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.register_buffer(
            "L_init",
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _compute_input_embeddings(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def empty_carry(self, batch_size: int) -> TRMInnerCarry:
        device = self.H_init.device
        return TRMInnerCarry(
            z_H=torch.empty(
                batch_size, self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device
            ),
            z_L=torch.empty(
                batch_size, self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device
            ),
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

    def forward(self, carry: TRMInnerCarry, batch: dict[str, torch.Tensor]):
        seq_info = {
            "cos_sin": self.rotary_emb() if hasattr(self, "rotary_emb") else None,
            "attention_mask": build_additive_attention_mask(batch["attention_mask"], self.forward_dtype),
        }
        input_embeddings = self._compute_input_embeddings(batch)
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


class TRMRerankerInner(RecursiveInnerBase):
    def __init__(self, config: TRMRerankerConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )
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

    def _input_embeddings(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(input_ids.to(torch.int64))
        embedding = embedding + self.segment_emb(token_type_ids.to(torch.int64))
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight[: input_ids.shape[1]].to(self.forward_dtype)
            )
        return self.embed_scale * embedding

    def _compute_input_embeddings(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._input_embeddings(batch["input_ids"], batch["token_type_ids"])


class RecursiveACTReranker(nn.Module):
    """ACT halting wrapper shared by the recursive variants.

    Subclasses set ``config_cls`` / ``inner_cls``; the carry bookkeeping,
    halting decisions, and Q-learning targets are identical across variants.
    """

    config_cls = TRMRerankerConfig
    inner_cls = TRMRerankerInner

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = self.config_cls(**config_dict)
        self.inner = self.inner_cls(self.config)

    def initial_carry(self, batch: dict[str, torch.Tensor]) -> TRMCarry:
        batch_size = batch["input_ids"].shape[0]
        return TRMCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=batch["input_ids"].device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=batch["input_ids"].device),
            current_data={key: torch.empty_like(value) for key, value in batch.items()},
        )

    def forward(self, carry: TRMCarry, batch: dict[str, torch.Tensor]):
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
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                ) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)
                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)
                        )
                    )
        return TRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs


class TRMReranker(RecursiveACTReranker):
    config_cls = TRMRerankerConfig
    inner_cls = TRMRerankerInner


def load_legacy_state_dict(model: "TRMReranker", state_dict: dict[str, torch.Tensor]) -> None:
    """Load a checkpoint produced by the legacy notebooks.

    Legacy module paths are identical except the block class names, so keys map
    one-to-one; kept as a function to make the parity test explicit.
    """
    model.load_state_dict(state_dict)


__all__ = [
    "MLPTBlock",
    "RecursiveACTReranker",
    "RecursiveInnerBase",
    "TRMCarry",
    "TRMInnerCarry",
    "TRMReasoningModule",
    "TRMReranker",
    "TRMRerankerConfig",
    "TRMRerankerInner",
    "load_legacy_state_dict",
]
