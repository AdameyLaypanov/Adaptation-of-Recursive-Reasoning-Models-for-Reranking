"""Assembly of a run from an ExperimentConfig: data artifacts, encoder, model.

Shared by scripts/train.py and scripts/evaluate.py so both resolve manifests
and build models identically.
"""

import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from transformers import AutoTokenizer

from .config import ExperimentConfig
from .data.encoding import PairEncoder
from .data.manifests import (
    resolve_run_manifest_artifact_path,
    validate_prep_manifest,
    validate_run_data_compatibility,
    validate_run_data_manifest,
)
from .data.passage_store import build_passage_token_subset_loader
from .models.registry import build_model, needs_bert_token_type_ids
from .utils import is_empty_path, load_json, resolve_relative_or_absolute_path


def load_pickle(path: Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def resolve_run_id(explicit: Optional[str] = None) -> str:
    return (
        explicit
        or os.environ.get("TRM_RUN_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or time.strftime("%Y%m%d-%H%M%S")
    )


@dataclass
class RunDataBundle:
    tokenizer: object
    seq_len: int
    max_query_len: int
    max_doc_len: int
    encoder: PairEncoder
    train_query_tokens: Dict
    train_passage_tokens: Dict
    dev_query_tokens: Dict
    epoch_dev_candidates: object
    epoch_dev_qrels: object
    final_dev_candidates: object
    final_dev_qrels: object
    run_final_full_dev: bool
    epoch_dev_mode_label: str
    sampled_train_triples_path: Path
    passage_token_getter: Callable[[int], List[int]]
    run_manifest: dict = field(repr=False, default=None)
    prep_manifest: dict = field(repr=False, default=None)
    prep_manifest_path: Path = None
    run_data_manifest_path: Path = None
    artifact_dir: Path = None


def load_run_data(cfg: ExperimentConfig, for_arch: Optional[str] = None, load_train: bool = True) -> RunDataBundle:
    if is_empty_path(cfg.data.run_data_manifest_path):
        raise ValueError("data.run_data_manifest_path is not configured")
    run_data_manifest_path = Path(cfg.data.run_data_manifest_path).expanduser().resolve()
    run_manifest = load_json(run_data_manifest_path)
    validate_run_data_manifest(run_manifest)
    run_data_cache_dir = run_data_manifest_path.parent

    prep_manifest_path = cfg.data.prep_manifest_path
    if is_empty_path(prep_manifest_path):
        prep_manifest_path = resolve_relative_or_absolute_path(run_manifest["base_prep_manifest_path"], run_data_cache_dir)
    prep_manifest_path = Path(prep_manifest_path).expanduser().resolve()
    artifact_dir = prep_manifest_path.parent
    prep_manifest = load_json(prep_manifest_path)
    validate_prep_manifest(prep_manifest)
    validate_run_data_compatibility(run_manifest, prep_manifest, cfg.training.train_triples_sample, cfg.experiment.seed)

    seq_len = int(prep_manifest["seq_len"])
    max_query_len = int(prep_manifest["max_query_len"])
    max_doc_len = int(prep_manifest["max_doc_len"])
    tokenizer_name = str(prep_manifest.get("tokenizer_name", run_manifest.get("tokenizer_name", "bert-base-uncased")))
    tokenizer_local_path = resolve_relative_or_absolute_path(prep_manifest.get("tokenizer_local_path", "tokenizer"), artifact_dir)
    if tokenizer_local_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_local_path, use_fast=True, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must provide cls_token_id, sep_token_id, and pad_token_id")

    encoder = PairEncoder(
        cls_id=tokenizer.cls_token_id,
        sep_id=tokenizer.sep_token_id,
        pad_id=tokenizer.pad_token_id,
        seq_len=seq_len,
        max_query_len=max_query_len,
        max_doc_len=max_doc_len,
        emit_bert_token_type_ids=needs_bert_token_type_ids(for_arch) if for_arch else False,
    )

    def artifact(key: str, base=run_data_cache_dir, aliases=None) -> Path:
        return resolve_run_manifest_artifact_path(run_manifest, key, base, aliases=aliases)

    run_final_full_dev = bool(run_manifest.get("run_final_full_dev", True))

    sampled_train_triples_path = artifact("sampled_train_triples_tsv")
    train_query_tokens = load_pickle(artifact("train_query_tokens_pkl")) if load_train else {}
    train_passage_tokens = load_pickle(artifact("train_passage_tokens_pkl")) if load_train else {}
    dev_query_tokens = load_pickle(artifact("dev_query_tokens_pkl", base=artifact_dir))
    epoch_dev_candidates = load_pickle(artifact("epoch_dev_candidates_pkl"))
    epoch_dev_qrels = load_pickle(artifact("epoch_dev_qrels_pkl"))
    final_dev_candidates = load_pickle(artifact("final_dev_candidates_pkl", base=artifact_dir)) if run_final_full_dev else None
    final_dev_qrels = load_pickle(artifact("final_dev_qrels_pkl", base=artifact_dir)) if run_final_full_dev else None

    shards_dir = artifact("passage_token_shards_dir", base=artifact_dir, aliases=["passage_tokens_shards_dir"])
    shards_index = artifact("passage_token_shards_index_json", base=artifact_dir, aliases=["passage_tokens_store_index_json"])
    _, get_passage_tokens, _ = build_passage_token_subset_loader(shards_index, artifact_dir, shards_dir_path=shards_dir)

    epoch_dev_mode_label = str(run_manifest.get("epoch_dev_mode_label") or run_manifest.get("dev_eval_mode"))

    return RunDataBundle(
        tokenizer=tokenizer,
        seq_len=seq_len,
        max_query_len=max_query_len,
        max_doc_len=max_doc_len,
        encoder=encoder,
        train_query_tokens=train_query_tokens,
        train_passage_tokens=train_passage_tokens,
        dev_query_tokens=dev_query_tokens,
        epoch_dev_candidates=epoch_dev_candidates,
        epoch_dev_qrels=epoch_dev_qrels,
        final_dev_candidates=final_dev_candidates,
        final_dev_qrels=final_dev_qrels,
        run_final_full_dev=run_final_full_dev,
        epoch_dev_mode_label=epoch_dev_mode_label,
        sampled_train_triples_path=sampled_train_triples_path,
        passage_token_getter=get_passage_tokens,
        run_manifest=run_manifest,
        prep_manifest=prep_manifest,
        prep_manifest_path=prep_manifest_path,
        run_data_manifest_path=run_data_manifest_path,
        artifact_dir=artifact_dir,
    )


def build_model_from_config(cfg: ExperimentConfig, seq_len: int, vocab_size: int, forward_dtype: str):
    arch = cfg.model.arch
    params = dict(cfg.model.params)
    if arch == "bert_scoring":
        return build_model(arch, params)
    params.setdefault("batch_size", cfg.training.per_device_batch_size)
    params.setdefault("seq_len", seq_len)
    params.setdefault("vocab_size", vocab_size)
    params.setdefault("forward_dtype", forward_dtype)
    params.setdefault("num_segment_types", 3)
    return build_model(arch, params)


def build_synthetic_batch(arch: str, batch_size: int, seq_len: int, vocab_size: int = 30522, query_len: int = 32):
    """Synthetic batch for latency/FLOPs/memory benchmarks (no data needed)."""
    import torch

    input_ids = torch.randint(1000, vocab_size, (batch_size, seq_len), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    token_type_ids[:, 1 : 1 + query_len] = 1
    token_type_ids[:, query_len + 2 : seq_len - 1] = 2
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    batch = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}
    if needs_bert_token_type_ids(arch):
        batch["bert_token_type_ids"] = (token_type_ids == 2).to(torch.long)
    return batch
