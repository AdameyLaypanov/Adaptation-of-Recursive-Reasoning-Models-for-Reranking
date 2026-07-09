"""Sharded flat-token passage store (read side)."""

import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..utils import make_tqdm, resolve_relative_or_absolute_path


def load_passage_token_shard_index(index_path: Path) -> dict:
    return json.loads(Path(index_path).read_text())


def build_passage_token_subset_loader(
    index_path: Path,
    artifact_dir: Path,
    shards_dir_path: Path | None = None,
    shard_cache_size: int = 512,
    show_progress: bool = True,
):
    """Returns (index, get_passage_tokens, load_passage_tokens_for_subset)."""
    index = load_passage_token_shard_index(index_path)
    if index.get("format") != "sharded_flat_token_arrays_v1":
        raise ValueError(
            f"Unsupported passage token store format: {index.get('format')!r}. "
            "Expected a sharded flat token store described by passage_token_shards_index_json."
        )

    shard_entries = index.get("shards") or []
    if not shard_entries:
        raise ValueError(f"Passage token shard index is missing shard entries: {index_path}")

    shard_size = int(index["shard_size"])
    shards_by_id = {int(entry["shard_id"]): entry for entry in shard_entries}
    resolved_shards_dir = Path(shards_dir_path).resolve() if shards_dir_path is not None else None

    def resolve_shard_path(shard_entry: dict) -> Path:
        raw_path_value = shard_entry.get("path")
        if raw_path_value:
            path = resolve_relative_or_absolute_path(str(raw_path_value), artifact_dir)
            return path

        filename = (
            shard_entry.get("filename")
            or shard_entry.get("file_name")
            or shard_entry.get("basename")
            or shard_entry.get("name")
        )
        if filename and resolved_shards_dir is not None:
            return resolved_shards_dir / str(filename)

        raise FileNotFoundError(
            f"Passage token shard entry does not provide a path or filename: shard_id={shard_entry.get('shard_id')}"
        )

    @lru_cache(maxsize=shard_cache_size)
    def load_shard(shard_id: int):
        shard_entry = shards_by_id.get(int(shard_id))
        if shard_entry is None:
            raise KeyError(f"No passage token shard found for shard_id={shard_id}")
        shard_path = resolve_shard_path(shard_entry)
        with np.load(shard_path, allow_pickle=False) as shard_data:
            shard_pids = shard_data["pid"]
            shard_offsets = shard_data["offsets"]
            shard_token_ids = shard_data["token_ids"]
        pid_lookup = {int(pid): idx for idx, pid in enumerate(shard_pids.tolist())}
        return {
            "pid": shard_pids,
            "offsets": shard_offsets,
            "token_ids": shard_token_ids,
            "pid_lookup": pid_lookup,
        }

    def get_passage_tokens(pid: int) -> list[int]:
        shard = load_shard(int(pid) // shard_size)
        pid_idx = shard["pid_lookup"].get(int(pid))
        if pid_idx is None:
            raise KeyError(f"Missing cached passage tokens for pid={pid}")
        start = int(shard["offsets"][pid_idx])
        end = int(shard["offsets"][pid_idx + 1])
        return shard["token_ids"][start:end].tolist()

    def load_passage_tokens_for_subset(pid_list: Iterable[int]) -> dict[int, list[int]]:
        unique_pids = sorted({int(pid) for pid in pid_list})
        pids_by_shard: dict[int, list[int]] = {}
        for pid in unique_pids:
            pids_by_shard.setdefault(int(pid) // shard_size, []).append(int(pid))

        subset: dict[int, list[int]] = {}
        with make_tqdm(total=len(unique_pids), desc="Load cached train passages", disable=not show_progress) as pbar:
            for shard_id in sorted(pids_by_shard):
                shard = load_shard(shard_id)
                for pid in pids_by_shard[shard_id]:
                    pid_idx = shard["pid_lookup"].get(pid)
                    if pid_idx is None:
                        raise KeyError(f"Missing cached passage tokens for pid={pid}")
                    start = int(shard["offsets"][pid_idx])
                    end = int(shard["offsets"][pid_idx + 1])
                    subset[pid] = shard["token_ids"][start:end].tolist()
                    pbar.update(1)
        return subset

    return index, get_passage_tokens, load_passage_tokens_for_subset
