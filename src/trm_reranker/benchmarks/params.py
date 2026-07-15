"""Parameter and memory accounting (E2 table columns).

Reports separately, as the plan requires:
- total params / body (non-embedding) params / trainable params,
- fp16 checkpoint size estimate,
- peak CUDA memory during a forward pass (when on GPU).
"""

import torch

from ..models.inference import run_model_once
from ..utils import model_device, unwrap_model

EMBEDDING_NAME_MARKERS = ("embed_tokens", "segment_emb", "embed_pos", "embeddings")


def count_parameters(model) -> dict[str, int]:
    raw_model = unwrap_model(model)
    total = 0
    embedding = 0
    trainable = 0
    for name, parameter in raw_model.named_parameters():
        numel = parameter.numel()
        total += numel
        if parameter.requires_grad:
            trainable += numel
        if any(marker in name for marker in EMBEDDING_NAME_MARKERS):
            embedding += numel
    return {
        "params_total": total,
        "params_embedding": embedding,
        "params_body": total - embedding,
        "params_trainable": trainable,
    }


def checkpoint_size_fp16_mb(model) -> float:
    total = sum(parameter.numel() for parameter in unwrap_model(model).parameters())
    return total * 2 / (1024**2)


@torch.no_grad()
def measure_peak_inference_memory(model, sample_batch: dict[str, torch.Tensor]) -> dict[str, float]:
    device = model_device(model)
    batch = {key: value.to(device) for key, value in sample_batch.items()}
    if device.type != "cuda":
        return {"peak_inference_memory_mb": float("nan"), "device": str(device)}
    torch.cuda.reset_peak_memory_stats(device)
    run_model_once(model, batch)
    torch.cuda.synchronize(device)
    return {
        "peak_inference_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "device": str(device),
    }


def summarize_model_footprint(model, sample_batch=None) -> dict[str, float]:
    summary: dict[str, float] = {}
    summary.update(count_parameters(model))
    summary["checkpoint_fp16_mb"] = checkpoint_size_fp16_mb(model)
    if sample_batch is not None:
        summary.update(measure_peak_inference_memory(model, sample_batch))
    return summary
