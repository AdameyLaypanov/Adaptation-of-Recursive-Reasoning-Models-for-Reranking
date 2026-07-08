"""Experiment configuration: YAML files + dot-key CLI overrides -> dataclasses.

Usage pattern (see scripts/train.py):

    cfg = load_experiment_config(["configs/base.yaml", "configs/arms/trm.yaml"],
                                 overrides=["experiment.seed=17"])

Later files win over earlier ones (deep merge); overrides win over files.
Every hyperparameter that ends up in the paper tables lives here, so a saved
``training_config.json`` fully describes a run.
"""

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentSection:
    name: str = "experiment"
    seed: int = 13
    output_root: str = ""
    run_id: Optional[str] = None


@dataclass
class DataSection:
    run_data_manifest_path: str = ""
    prep_manifest_path: Optional[str] = None
    num_workers: int = 0


@dataclass
class ModelSection:
    arch: str = "trm"
    params: Dict[str, Any] = field(default_factory=dict)


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
    max_train_steps: Optional[int] = None
    run_epoch_fraction: Optional[float] = None
    run_train_steps: Optional[int] = None
    checkpoint_epoch_fraction: Optional[float] = 0.005
    checkpoint_every_n_steps: Optional[int] = None
    dev_eval_query_limit: int = 500
    devices: int = 1
    use_ddp: bool = False
    grad_accum_steps: int = 1
    resume_from_checkpoint: bool = False
    resume_checkpoint_path: Optional[str] = None
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_override_value(raw_value: str) -> Any:
    value = yaml.safe_load(raw_value)
    # YAML 1.1 parses "1e-5" (no dot) as a string; coerce numeric-looking strings.
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _apply_override(config_dict: Dict[str, Any], dotted_key: str, raw_value: str) -> None:
    keys = dotted_key.split(".")
    target = config_dict
    for key in keys[:-1]:
        target = target.setdefault(key, {})
        if not isinstance(target, dict):
            raise ValueError(f"Cannot override {dotted_key!r}: {key!r} is not a mapping")
    target[keys[-1]] = _parse_override_value(raw_value)


def _build_section(section_cls, section_dict: Dict[str, Any]):
    known = {f.name for f in fields(section_cls)}
    unknown = sorted(set(section_dict) - known)
    if unknown:
        raise ValueError(f"Unknown keys for {section_cls.__name__}: {unknown}. Known keys: {sorted(known)}")
    return section_cls(**section_dict)


def load_experiment_config(config_paths: List[str], overrides: Optional[List[str]] = None) -> ExperimentConfig:
    merged: Dict[str, Any] = {}
    for config_path in config_paths:
        loaded = yaml.safe_load(Path(config_path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file {config_path} must contain a mapping at the top level")
        merged = _deep_merge(merged, loaded)

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must look like section.key=value, got {override!r}")
        dotted_key, raw_value = override.split("=", 1)
        _apply_override(merged, dotted_key.strip(), raw_value.strip())

    return ExperimentConfig(
        experiment=_build_section(ExperimentSection, merged.get("experiment", {})),
        data=_build_section(DataSection, merged.get("data", {})),
        model=_build_section(ModelSection, merged.get("model", {})),
        training=_build_section(TrainingSection, merged.get("training", {})),
    )
