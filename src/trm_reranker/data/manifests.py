"""Validation and resolution of prep/run-data manifests produced by data prep."""

from pathlib import Path
from typing import Dict, List, Optional

from ..utils import resolve_relative_or_absolute_path


def get_manifest_mapping(manifest: dict, section_name: str) -> Dict[str, object]:
    value = manifest.get(section_name) or {}
    if not isinstance(value, dict):
        return {}
    return value


def manifest_value_candidates(primary_key: str, aliases: Optional[List[str]] = None) -> List[str]:
    keys = [primary_key]
    if aliases:
        keys.extend(aliases)
    unique: List[str] = []
    seen = set()
    for key in keys:
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def get_manifest_mapping_value(mapping: Dict[str, object], keys: List[str]):
    for key in keys:
        value = mapping.get(key)
        if value:
            return value, key
    return None, None


def validate_prep_manifest(manifest: dict) -> None:
    schema_version = int(manifest.get("schema_version", 0))
    if schema_version < 4:
        raise ValueError(
            "Prep manifest schema_version is too old for the dataset-level cache workflow. "
            "Re-run the data prep step to build the sharded full-collection passage cache."
        )

    required_artifact_keys = {
        "train_query_tokens_pkl": [],
        "dev_query_tokens_pkl": [],
        "passage_token_shards_dir": ["passage_tokens_shards_dir"],
        "passage_token_shards_index_json": ["passage_tokens_store_index_json"],
        "passage_token_store_stats_json": ["passage_tokens_store_stats_json"],
        "dev_candidates_pkl": [],
        "dev_qrels_pkl": [],
    }
    artifacts = get_manifest_mapping(manifest, "artifacts")
    missing_artifacts = [
        key
        for key, aliases in required_artifact_keys.items()
        if get_manifest_mapping_value(artifacts, manifest_value_candidates(key, aliases))[0] is None
    ]
    if missing_artifacts:
        raise ValueError(
            "Prep manifest is missing required dataset-level artifacts. "
            f"Re-run data prep. Missing keys: {missing_artifacts}. "
            f"Available artifact keys: {sorted(artifacts)}"
        )


def validate_run_data_manifest(run_manifest: dict) -> None:
    schema_version = int(run_manifest.get("schema_version", 0))
    if schema_version < 1:
        raise ValueError("run_data_manifest.json has an unsupported schema_version. Re-run run-data prep.")
    required_top_level = [
        "base_prep_manifest_path",
        "base_artifact_dir",
        "train_triples_sample",
        "seed",
        "dev_eval_mode",
        "dev_eval_seed",
        "epoch_dev_mode_label",
        "run_final_full_dev",
        "counts",
        "artifacts",
    ]
    missing_top_level = [key for key in required_top_level if run_manifest.get(key) is None]
    if missing_top_level:
        raise ValueError(f"run_data_manifest.json is missing required top-level keys. Missing: {missing_top_level}")

    dev_eval_mode = str(run_manifest.get("dev_eval_mode"))
    if dev_eval_mode == "quick_count" and run_manifest.get("dev_eval_query_count") is None:
        raise ValueError("run_data_manifest.json is missing dev_eval_query_count for quick_count mode.")
    if dev_eval_mode == "quick_fraction" and run_manifest.get("dev_eval_fraction") is None:
        raise ValueError("run_data_manifest.json is missing dev_eval_fraction for quick_fraction mode.")
    if dev_eval_mode not in {"quick_count", "quick_fraction", "full"}:
        raise ValueError(f"run_data_manifest.json has unsupported dev_eval_mode={dev_eval_mode!r}")

    required_artifact_keys = [
        "sampled_train_triples_tsv",
        "train_query_tokens_pkl",
        "train_passage_tokens_pkl",
        "dev_query_tokens_pkl",
        "epoch_dev_candidates_pkl",
        "epoch_dev_qrels_pkl",
        "passage_token_shards_dir",
        "passage_token_shards_index_json",
        "passage_token_store_stats_json",
    ]
    if bool(run_manifest.get("run_final_full_dev", True)):
        required_artifact_keys.extend(["final_dev_candidates_pkl", "final_dev_qrels_pkl"])
    artifacts = get_manifest_mapping(run_manifest, "artifacts")
    missing_artifacts = [key for key in required_artifact_keys if artifacts.get(key) is None]
    if missing_artifacts:
        raise ValueError(
            "run_data_manifest.json is missing required artifact entries. "
            f"Missing: {missing_artifacts}. Available: {sorted(artifacts)}"
        )


def validate_run_data_compatibility(run_manifest: dict, prep_manifest: dict, train_triples_sample: int, seed: int) -> None:
    mismatches = {}
    prep_expected = {
        "tokenizer_name": str(prep_manifest.get("tokenizer_name", "")),
        "seq_len": int(prep_manifest.get("seq_len", -1)),
        "max_query_len": int(prep_manifest.get("max_query_len", -1)),
        "max_doc_len": int(prep_manifest.get("max_doc_len", -1)),
    }
    run_actual = {
        "tokenizer_name": str(run_manifest.get("tokenizer_name", "")),
        "seq_len": int(run_manifest.get("seq_len", -1)),
        "max_query_len": int(run_manifest.get("max_query_len", -1)),
        "max_doc_len": int(run_manifest.get("max_doc_len", -1)),
    }
    for key, expected_value in prep_expected.items():
        if run_actual[key] != expected_value:
            mismatches[key] = {"expected": expected_value, "actual": run_actual[key]}

    run_config_expected = {
        "train_triples_sample": int(train_triples_sample),
        "seed": int(seed),
    }
    run_config_actual = {
        "train_triples_sample": int(run_manifest.get("train_triples_sample", -1)),
        "seed": int(run_manifest.get("seed", -1)),
    }
    for key, expected_value in run_config_expected.items():
        if run_config_actual[key] != expected_value:
            mismatches[key] = {"expected": expected_value, "actual": run_config_actual[key]}

    if mismatches:
        raise ValueError(
            "run_data_manifest.json is incompatible with the prep manifest or current training run settings. "
            f"Mismatches: {mismatches}. Re-run run-data prep with matching train sample settings."
        )


def resolve_run_manifest_artifact_path(
    run_manifest: dict,
    key: str,
    base_dir: Path,
    aliases: Optional[List[str]] = None,
) -> Path:
    artifacts = get_manifest_mapping(run_manifest, "artifacts")
    value, _ = get_manifest_mapping_value(artifacts, manifest_value_candidates(key, aliases))
    if value is None:
        raise KeyError(
            f"Run-data manifest is missing required artifact key {key!r}. Available artifact keys: {sorted(artifacts)}"
        )
    return resolve_relative_or_absolute_path(str(value), base_dir)
