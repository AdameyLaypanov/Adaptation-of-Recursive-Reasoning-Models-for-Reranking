"""Loss, LR schedule and step accounting (ported from the legacy notebooks)."""

import math

import torch
import torch.nn.functional as F

from ..models.inference import run_model_once
from ..utils import unwrap_model


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
    """Pairwise logistic loss -logsigmoid(pos - neg); identical for all variants (K2)."""
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


def compute_infonce_batch_metrics(model, group_batch, group_size: int, temperature: float = 1.0):
    """InfoNCE (softmax cross-entropy) over (positive, K negatives) score groups.

    ``group_batch`` holds ``B * group_size`` encoded pairs laid out group-by-group
    with the positive first (see ``make_grouped_collate``). With ``group_size=2``
    and ``temperature=1`` this reduces exactly to the pairwise logistic loss.

    Reported metrics keep the pairwise keys: ``pairwise_acc`` is top-1 accuracy
    (positive beats every negative), ``margin`` is positive minus the hardest
    negative.
    """
    scores, _ = run_model_once(model, group_batch)
    if scores.shape[0] % int(group_size) != 0:
        raise ValueError(f"Batch of {scores.shape[0]} pairs is not divisible by group_size={group_size}")
    grouped_scores = scores.view(-1, int(group_size))
    logits = grouped_scores / float(temperature)
    targets = torch.zeros(grouped_scores.shape[0], dtype=torch.long, device=grouped_scores.device)
    loss = F.cross_entropy(logits, targets)

    pos_scores = grouped_scores[:, 0]
    neg_scores = grouped_scores[:, 1:]
    hardest_negative = neg_scores.max(dim=1).values
    return {
        "loss": loss,
        "pairwise_acc": (pos_scores > hardest_negative).float().mean(),
        "margin": (pos_scores - hardest_negative).mean(),
        "pos_score": pos_scores.mean(),
        "neg_score": neg_scores.mean(),
    }


def build_warmup_steps(total_steps: int, warmup_ratio: float) -> int:
    if total_steps <= 0:
        return 0
    return max(0, min(total_steps - 1, math.ceil(total_steps * warmup_ratio)))


def resolve_step_count(
    explicit_steps: int | None,
    epoch_fraction: float | None,
    steps_per_epoch: int,
    default_steps: int | None = None,
) -> int | None:
    """Priority: explicit step count > fraction of an epoch (ceil, >=1) > default.

    Used for both the session window (run_train_steps / run_epoch_fraction) and
    the checkpoint period (checkpoint_every_n_steps / checkpoint_epoch_fraction).
    """
    if explicit_steps is not None:
        explicit_steps = int(explicit_steps)
        if explicit_steps <= 0:
            raise ValueError("explicit step count must be positive")
        return explicit_steps
    if epoch_fraction is not None:
        epoch_fraction = float(epoch_fraction)
        if epoch_fraction <= 0:
            raise ValueError("epoch fraction must be positive")
        return max(1, math.ceil(int(steps_per_epoch) * epoch_fraction))
    return default_steps


def build_linear_warmup_decay_lambda(warmup_steps: int, total_steps: int | None):
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
