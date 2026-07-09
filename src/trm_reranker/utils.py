"""Small shared helpers used across the package."""

import json
import logging
import pickle
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging for scripts; call once per entry point."""
    logging.basicConfig(format=LOG_FORMAT, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def make_tqdm(*args, **kwargs):
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(*args, **kwargs)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def load_pickle(path: Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def is_empty_path(path) -> bool:
    return path is None or str(path).strip() in {"", "."}


def resolve_relative_or_absolute_path(value: str, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir) / path).resolve()


def unwrap_model(model):
    """Strip a DDP (or similar) wrapper; no-op for plain modules."""
    return model.module if hasattr(model, "module") else model


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def get_autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16-mixed":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "16-mixed":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()
