"""Shared transformer building blocks.

Single implementation used by every comparison arm (TRM, vanilla shallow/deep,
weight-tied deep, BERT+TRM head). Keeping one copy is what guarantees the K1
alignment requirement of the WSDM plan: all arms share norm type/placement,
activation, positional encoding, initialisation and bias settings by
construction. Ported verbatim from the legacy notebooks.
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

CosSin = Tuple[torch.Tensor, torch.Tensor]


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower**2)
            pdf_l = c * math.exp(-0.5 * upper**2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor


def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    seq_len = q.shape[1]
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features**0.5)))
        self.bias = nn.Parameter(torch.zeros((out_features,))) if bias else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(inputs.dtype) if self.bias is not None else None
        return F.linear(inputs, self.weight.to(inputs.dtype), bias=bias)


class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.embedding(inputs, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self) -> CosSin:
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int, num_heads: int, num_key_value_heads: int, causal: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.qkv_proj = CastedLinear(hidden_size, (num_heads + 2 * num_key_value_heads) * head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, : self.num_heads]
        key = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attn_mask = attention_mask.to(query.dtype) if attention_mask is not None else None
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, attn_mask=attn_mask, is_causal=self.causal)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


class TransformerBlock(nn.Module):
    """Post-norm RMSNorm + SwiGLU block; identical across all arms (K1)."""

    def __init__(self, hidden_size: int, num_heads: int, expansion: float, rms_norm_eps: float = 1e-5):
        super().__init__()
        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion)
        self.norm_eps = rms_norm_eps

    def forward(self, hidden_states: torch.Tensor, cos_sin: CosSin = None, attention_mask: torch.Tensor = None) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states, attention_mask=attention_mask),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


def build_additive_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    attention_mask = attention_mask.to(torch.bool)
    additive_mask = torch.zeros(attention_mask.shape, dtype=dtype, device=attention_mask.device)
    additive_mask = additive_mask.masked_fill(~attention_mask, torch.finfo(dtype).min)
    return additive_mask[:, None, None, :]
