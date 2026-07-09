"""Model factory: variant name -> model instance.

Variant names match the config files in ``configs/variants/``. All knowledge about
architecture peculiarities (kwargs-style constructor, no model dims, BERT
token_type_ids) lives here, so callers never special-case arch names.
"""

import inspect
from dataclasses import fields

from torch import nn

from .bert import BertScoringReranker, BertTRMReranker
from .tied import TiedReranker
from .trm import TRMReranker
from .vanilla import VanillaReranker

ARCHITECTURES = {
    "trm": TRMReranker,
    "vanilla": VanillaReranker,
    "tied": TiedReranker,
    "bert_trm": BertTRMReranker,
    "bert_scoring": BertScoringReranker,
}

# Variants whose encode_pair must also emit BERT-native token_type_ids.
BERT_INPUT_ARCHITECTURES = {"bert_trm", "bert_scoring"}

# Variants that take plain keyword arguments and do not need seq_len/vocab_size etc.
MODEL_DIM_FREE_ARCHITECTURES = {"bert_scoring"}


def _known_param_names(model_cls) -> set:
    config_cls = getattr(model_cls, "config_cls", None)
    if config_cls is not None:
        return {f.name for f in fields(config_cls)}
    return set(inspect.signature(model_cls.__init__).parameters) - {"self"}


def build_model(arch: str, model_config: dict) -> nn.Module:
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}. Available: {sorted(ARCHITECTURES)}")
    model_cls = ARCHITECTURES[arch]
    known = _known_param_names(model_cls)
    unknown = sorted(set(model_config) - known)
    if unknown:
        raise ValueError(f"Unknown model.params keys for arch {arch!r}: {unknown}. Known keys: {sorted(known)}")
    if arch in MODEL_DIM_FREE_ARCHITECTURES:
        return model_cls(**model_config)
    return model_cls(model_config)


def needs_bert_token_type_ids(arch: str) -> bool:
    return arch in BERT_INPUT_ARCHITECTURES


def needs_model_dims(arch: str) -> bool:
    """Whether build_model expects seq_len/vocab_size/batch_size/... to be injected."""
    return arch not in MODEL_DIM_FREE_ARCHITECTURES


__all__ = ["ARCHITECTURES", "build_model", "needs_bert_token_type_ids", "needs_model_dims"]
