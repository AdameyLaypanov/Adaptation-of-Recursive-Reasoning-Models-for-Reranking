"""Experiment configuration: YAML files + dot-key CLI overrides -> dataclasses.

Usage pattern (see scripts/train.py):

    cfg = load_experiment_config(["configs/base.yaml", "configs/variants/trm.yaml"],
                                 overrides=["experiment.seed=17"])

Later files win over earlier ones (deep merge); overrides win over files.
Every hyperparameter that ends up in the paper tables lives here, so a saved
``training_config.json`` fully describes a run.
"""

import copy
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# YAML 1.1 (PyYAML) parses exponent-form numbers without a dot ("1e-5") as
# strings; only such values are coerced to float. Everything else keeps the
# type YAML gave it, so string values like run_id="007" stay strings.
_EXPONENT_FLOAT_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)[eE][+-]?\d+$")


@dataclass
class ExperimentSection:
    name: str = "experiment"
    seed: int = 13
    output_root: str = ""
    run_id: str | None = None


@dataclass
class DataSection:
    run_data_manifest_path: str = ""
    prep_manifest_path: str | None = None
    # Output of scripts/mine_hard_negatives.py; required for loss=infonce,
    # optional for pairwise (replaces the official-triples negatives).
    hard_negatives_path: str | None = None
    num_workers: int = 0


@dataclass
class ModelSection:
    arch: str = "trm"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingSection:
    per_device_batch_size: int = 512
    eval_batch_size: int = 512
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.06
    precision: str = "bf16-mixed"
    epochs: int = 1
    train_triples_sample: int = 1_000_000
    max_train_steps: int | None = None
    # Loss is part of the shared recipe (K2): change it in base.yaml so every
    # comparison variant trains identically. For "infonce" one batch element is
    # a (query, positive, num_negatives negatives) group, i.e. the forward cost
    # per element is (1 + num_negatives) pairs.
    loss: str = "pairwise_logistic"
    num_negatives: int = 7
    infonce_temperature: float = 1.0
    run_epoch_fraction: float | None = None
    run_train_steps: int | None = None
    checkpoint_epoch_fraction: float | None = 0.005
    checkpoint_every_n_steps: int | None = None
    keep_last_step_checkpoints: int | None = None
    dev_eval_query_limit: int = 500
    devices: int = 1
    use_ddp: bool = False
    grad_accum_steps: int = 1
    resume_from_checkpoint: bool = False
    resume_checkpoint_path: str | None = None
    allow_resume_lr_override: bool = False
    run_epoch_dev_eval_on_partial: bool = False
    freeze_token_embeddings: bool = False
    freeze_segment_embeddings: bool = False
    tqdm_postfix_every_n_steps: int = 200


@dataclass
class ExperimentConfig:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    data: DataSection = field(default_factory=DataSection)
    model: ModelSection = field(default_factory=ModelSection)
    training: TrainingSection = field(default_factory=TrainingSection)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_override_value(raw_value: str) -> Any:
    value = yaml.safe_load(raw_value)
    if isinstance(value, str) and _EXPONENT_FLOAT_RE.match(value):
        return float(value)
    return value


def _apply_override(config_dict: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    keys = dotted_key.split(".")
    target = config_dict
    for key in keys[:-1]:
        target = target.setdefault(key, {})
        if not isinstance(target, dict):
            raise ValueError(f"Cannot override {dotted_key!r}: {key!r} is not a mapping")
    target[keys[-1]] = _parse_override_value(raw_value)


def _build_section(section_cls, section_dict: dict[str, Any]):
    known = {f.name for f in fields(section_cls)}
    unknown = sorted(set(section_dict) - known)
    if unknown:
        raise ValueError(f"Unknown keys for {section_cls.__name__}: {unknown}. Known keys: {sorted(known)}")
    return section_cls(**section_dict)


def load_experiment_config(config_paths: list[str], overrides: list[str] | None = None) -> ExperimentConfig:
    known_sections = {f.name for f in fields(ExperimentConfig)}
    merged: dict[str, Any] = {}
    for config_path in config_paths:
        loaded = yaml.safe_load(Path(config_path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file {config_path} must contain a mapping at the top level")
        unknown_sections = sorted(set(loaded) - known_sections)
        if unknown_sections:
            raise ValueError(
                f"Config file {config_path} has unknown top-level sections: {unknown_sections}. "
                f"Known sections: {sorted(known_sections)}"
            )
        merged = _deep_merge(merged, loaded)

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must look like section.key=value, got {override!r}")
        dotted_key, raw_value = override.split("=", 1)
        dotted_key = dotted_key.strip()
        if dotted_key.split(".")[0] not in known_sections:
            raise ValueError(
                f"Override {override!r} targets unknown section {dotted_key.split('.')[0]!r}. "
                f"Known sections: {sorted(known_sections)}"
            )
        _apply_override(merged, dotted_key, raw_value.strip())

    return ExperimentConfig(
        experiment=_build_section(ExperimentSection, merged.get("experiment", {})),
        data=_build_section(DataSection, merged.get("data", {})),
        model=_build_section(ModelSection, merged.get("model", {})),
        training=_build_section(TrainingSection, merged.get("training", {})),
    )
