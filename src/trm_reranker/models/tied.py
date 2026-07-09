"""Weight-tied deep transformer (ALBERT variant, experiment E3).

Cross-layer parameter sharing without the TRM machinery: no two latent states,
no input re-injection — a block stack of ``num_layers`` unique layers applied
``num_repeats`` times. Together with E5 this decomposes what exactly in TRM
contributes on top of plain tying.

Effective depth = ``num_layers * num_repeats``; unique body params match a
``num_layers``-deep vanilla model. Everything except the repeat count is
inherited from the vanilla variant.
"""

from dataclasses import dataclass

from .vanilla import VanillaReranker, VanillaRerankerConfig, VanillaRerankerInner


@dataclass
class TiedRerankerConfig(VanillaRerankerConfig):
    num_repeats: int = 1


class TiedRerankerInner(VanillaRerankerInner):
    @property
    def num_body_passes(self) -> int:
        return int(self.config.num_repeats)


class TiedReranker(VanillaReranker):
    config_cls = TiedRerankerConfig
    inner_cls = TiedRerankerInner


__all__ = ["TiedReranker", "TiedRerankerConfig", "TiedRerankerInner"]
