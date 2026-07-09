"""DDP helpers (ported from the legacy notebooks)."""

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_dist_available_and_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size(use_ddp: bool = True) -> int:
    if is_dist_available_and_initialized():
        return dist.get_world_size()
    if use_ddp and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return int(os.environ.get("WORLD_SIZE", "1"))
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def ddp_barrier() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if is_dist_available_and_initialized():
        reduced = tensor.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= dist.get_world_size()
        return reduced
    return tensor


def setup_distributed(use_ddp: bool) -> torch.device:
    world_size = get_world_size(use_ddp)
    if use_ddp and world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP was requested, but CUDA is not available.")
        if not is_dist_available_and_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def cleanup_distributed() -> None:
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def resolve_precision(requested_precision: str, device_kind: str) -> str:
    if requested_precision == "bf16-mixed":
        if device_kind == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return requested_precision
        if is_main_process():
            logger.warning(
                'Falling back from precision=%r to "32-true": bf16 is not available in this runtime.',
                requested_precision,
            )
        return "32-true"
    if requested_precision == "16-mixed" and device_kind != "cuda":
        if is_main_process():
            logger.warning(
                'Falling back from precision=%r to "32-true": fp16 autocast needs CUDA.', requested_precision
            )
        return "32-true"
    return requested_precision
