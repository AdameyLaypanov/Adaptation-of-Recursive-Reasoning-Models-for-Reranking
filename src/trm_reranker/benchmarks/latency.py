"""Latency measurement with the unified protocol required by the plan:

fixed device, fixed batch size and sequence length, warmup excluded, CUDA
synchronisation around each step, and mean/median/p95 reported per config —
never copied between rows.
"""

import time
from typing import Dict

import numpy as np
import torch

from ..training.distributed import model_device
from ..training.optim import run_model_once


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_forward_latency(
    model,
    sample_batch: Dict[str, torch.Tensor],
    warmup_steps: int = 10,
    measure_steps: int = 50,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    device = model_device(model)
    batch = {key: value.to(device) for key, value in sample_batch.items()}

    warmup_steps = max(0, int(warmup_steps))
    measure_steps = int(measure_steps)
    if measure_steps <= 0:
        raise ValueError("measure_steps must be positive")

    for _ in range(warmup_steps):
        run_model_once(model, batch)
    synchronize_if_cuda(device)

    values_ms = []
    for _ in range(measure_steps):
        synchronize_if_cuda(device)
        start = time.perf_counter()
        run_model_once(model, batch)
        synchronize_if_cuda(device)
        values_ms.append((time.perf_counter() - start) * 1000.0)

    if was_training:
        model.train()

    values = np.asarray(values_ms, dtype=np.float64)
    batch_size = int(next(iter(batch.values())).shape[0])
    return {
        "device": str(device),
        "batch_size": batch_size,
        "seq_len": int(batch["input_ids"].shape[1]),
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "latency_ms_mean": float(values.mean()),
        "latency_ms_std": float(values.std()),
        "latency_ms_p50": float(np.percentile(values, 50)),
        "latency_ms_p95": float(np.percentile(values, 95)),
        "latency_ms_min": float(values.min()),
        "latency_ms_max": float(values.max()),
        "latency_ms_per_pair_p50": float(np.percentile(values, 50) / batch_size),
    }
