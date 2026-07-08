"""Small shared helpers used across the package."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm


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


def is_empty_path(path) -> bool:
    return path is None or str(path).strip() in {"", "."}


def resolve_relative_or_absolute_path(value: str, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir) / path).resolve()
