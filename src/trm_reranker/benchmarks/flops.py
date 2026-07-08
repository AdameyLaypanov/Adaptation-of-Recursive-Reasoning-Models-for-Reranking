"""FLOPs measurement (E0/E2): profiler-based, independent of latency.

Primary path: ``torch.profiler`` with ``with_flops=True``. Optional cross-check
with fvcore (``pip install .[bench]``); fvcore counts one multiply-add as one
FLOP and misses some ops (e.g. SDPA), so the torch profiler number is the one
to report.
"""

from typing import Dict

import torch

from ..training.distributed import model_device
from ..training.optim import run_model_once


@torch.no_grad()
def measure_forward_flops(model, sample_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    device = model_device(model)
    batch = {key: value.to(device) for key, value in sample_batch.items()}

    # Warm up once so lazy initialisation does not pollute the profile.
    run_model_once(model, batch)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities, with_flops=True) as prof:
        run_model_once(model, batch)

    total_flops = sum(event.flops for event in prof.key_averages() if event.flops)

    if was_training:
        model.train()

    batch_size = int(next(iter(batch.values())).shape[0])
    return {
        "device": str(device),
        "batch_size": batch_size,
        "seq_len": int(batch["input_ids"].shape[1]),
        "total_flops": float(total_flops),
        "gflops_per_batch": float(total_flops) / 1e9,
        "gflops_per_pair": float(total_flops) / batch_size / 1e9,
    }
