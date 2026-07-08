"""Loss, LR schedule and step accounting (ported from the legacy notebooks)."""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .distributed import unwrap_model


def run_model_once(model, batch: Dict[str, torch.Tensor]):
    stateful_model = unwrap_model(model)
    carry = stateful_model.initial_carry(batch)

    if stateful_model.config.halt_max_steps == 1:
        carry, outputs = model(carry, batch)
        return outputs["scores"], outputs

    outputs = None
    for _ in range(stateful_model.config.halt_max_steps):
        carry, outputs = model(carry, batch)
        if bool(carry.halted.all()):
            break

    if outputs is None:
        raise RuntimeError("Model produced no outputs")

    return outputs["scores"], outputs


def concat_batches(pos_batch, neg_batch):
    pos_keys = set(pos_batch)
    neg_keys = set(neg_batch)
    if pos_keys != neg_keys:
        missing_in_neg = sorted(pos_keys - neg_keys)
        missing_in_pos = sorted(neg_keys - pos_keys)
        raise KeyError(
            f"pos_batch and neg_batch must contain the same keys: missing_in_neg={missing_in_neg}, missing_in_pos={missing_in_pos}"
        )

    pair_batch = {}
    for key in pos_batch:
        pos_value = pos_batch[key]
        neg_value = neg_batch[key]
        if not isinstance(pos_value, torch.Tensor) or not isinstance(neg_value, torch.Tensor):
            raise TypeError(f"Batch key {key!r} cannot be concatenated because it is not a tensor")
        if pos_value.dim() == 0 or neg_value.dim() == 0:
            raise ValueError(f"Batch key {key!r} must have a batch dimension")
        if pos_value.shape[1:] != neg_value.shape[1:]:
            raise ValueError(
                f"Batch key {key!r} has incompatible shapes for concat: pos={tuple(pos_value.shape)}, neg={tuple(neg_value.shape)}"
            )
        pair_batch[key] = torch.cat([pos_value, neg_value], dim=0)

    return pair_batch


def compute_pairwise_batch_metrics(model, pos_batch, neg_batch):
    """Pairwise logistic loss -logsigmoid(pos - neg); identical for all arms (K2)."""
    pair_batch = concat_batches(pos_batch, neg_batch)

    scores, _ = run_model_once(model, pair_batch)
    pos_scores, neg_scores = scores.chunk(2, dim=0)

    margin = pos_scores - neg_scores
    loss = -F.logsigmoid(margin).mean()
    pairwise_acc = (pos_scores > neg_scores).float().mean()
    return {
        "loss": loss,
        "pairwise_acc": pairwise_acc,
        "margin": margin.mean(),
        "pos_score": pos_scores.mean(),
        "neg_score": neg_scores.mean(),
    }


def build_warmup_steps(total_steps: int, warmup_ratio: float) -> int:
    if total_steps <= 0:
        return 0
    return max(0, min(total_steps - 1, int(math.ceil(total_steps * warmup_ratio))))


def resolve_step_count(
    explicit_steps: Optional[int],
    epoch_fraction: Optional[float],
    steps_per_epoch: int,
    default_steps: Optional[int] = None,
) -> Optional[int]:
    if explicit_steps is not None:
        explicit_steps = int(explicit_steps)
        if explicit_steps <= 0:
            raise ValueError("explicit step count must be positive")
        return explicit_steps
    if epoch_fraction is not None:
        epoch_fraction = float(epoch_fraction)
        if epoch_fraction <= 0:
            raise ValueError("epoch fraction must be positive")
        return max(1, int(math.ceil(int(steps_per_epoch) * epoch_fraction)))
    return default_steps


def build_linear_warmup_decay_lambda(warmup_steps: int, total_steps: Optional[int]):
    def lr_lambda(current_step: int) -> float:
        if total_steps is None or total_steps <= 0:
            return 1.0
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    return lr_lambda


def count_model_parameters(model) -> int:
    return sum(parameter.numel() for parameter in unwrap_model(model).parameters())


def count_trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in unwrap_model(model).parameters() if parameter.requires_grad)


def get_trainable_parameters(model):
    return [parameter for parameter in unwrap_model(model).parameters() if parameter.requires_grad]
