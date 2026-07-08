"""Model factory: arm name -> model instance.

Arm names match the config files in ``configs/arms/``.
"""

from typing import Dict

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
}

# Arms whose encode_pair must also emit BERT-native token_type_ids.
BERT_INPUT_ARCHITECTURES = {"bert_trm", "bert_scoring"}


def build_model(arch: str, model_config: Dict) -> nn.Module:
    if arch == "bert_scoring":
        return BertScoringReranker(**model_config)
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}. Available: {sorted(ARCHITECTURES) + ['bert_scoring']}")
    return ARCHITECTURES[arch](model_config)


def needs_bert_token_type_ids(arch: str) -> bool:
    return arch in BERT_INPUT_ARCHITECTURES


__all__ = ["ARCHITECTURES", "build_model", "needs_bert_token_type_ids"]
