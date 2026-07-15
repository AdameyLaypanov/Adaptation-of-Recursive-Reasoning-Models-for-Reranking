#!/usr/bin/env python
"""Train a reranker variant from YAML configs.

Examples (see docs/running.md for the full mode-by-mode guide):
    # smoke-прогон TRM-варианта
    uv run python scripts/train.py \
        --config configs/base.yaml configs/variants/trm.yaml configs/local.yaml \
        --set training.max_train_steps=20

    # DDP на 2 GPU
    uv run torchrun --nproc_per_node=2 scripts/train.py \
        --config configs/base.yaml configs/variants/vanilla_deep.yaml configs/local.yaml \
        --set training.use_ddp=true --set training.devices=2

    # мультисид (E2)
    for seed in 13 17 42; do
        uv run python scripts/train.py \
            --config configs/base.yaml configs/variants/trm.yaml configs/local.yaml \
            --seed $seed --run-id seed$seed
    done
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from trm_reranker.config import load_experiment_config
from trm_reranker.runtime import build_model_from_config, load_run_data, resolve_run_id
from trm_reranker.training.checkpoints import RunPaths
from trm_reranker.training.distributed import (
    cleanup_distributed,
    get_world_size,
    is_main_process,
    resolve_precision,
    setup_distributed,
)
from trm_reranker.training.trainer import Trainer
from trm_reranker.utils import save_json, seed_everything, setup_logging

logger = logging.getLogger("scripts.train")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", required=True, help="YAML configs, merged left to right (base first)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override: section.key=value")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--print-config", action="store_true", help="Print the merged config as JSON and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = list(args.overrides)
    if args.seed is not None:
        overrides.append(f"experiment.seed={args.seed}")
    # json.dumps quotes flag values so YAML keeps them as strings ("007" stays "007").
    if args.run_id is not None:
        overrides.append(f"experiment.run_id={json.dumps(args.run_id)}")
    if args.output_root is not None:
        overrides.append(f"experiment.output_root={json.dumps(args.output_root)}")
    if args.resume_from is not None:
        overrides.append("training.resume_from_checkpoint=true")
        overrides.append(f"training.resume_checkpoint_path={json.dumps(args.resume_from)}")
    cfg = load_experiment_config(args.config, overrides)

    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
        return

    seed_everything(cfg.experiment.seed)

    # Fail fast on run-level paths before touching multi-minute data loading.
    if not cfg.experiment.output_root:
        raise ValueError("experiment.output_root is not configured (configs/local.yaml or --output-root)")
    run_id = resolve_run_id(cfg.experiment.run_id)
    experiment_name = f"{cfg.experiment.name}_seed{cfg.experiment.seed}_{run_id}"
    paths = RunPaths.create(cfg.experiment.output_root, experiment_name)

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    should_use_ddp = cfg.training.use_ddp and world_size_env > 1
    if cfg.training.use_ddp and world_size_env == 1 and is_main_process():
        logger.warning(
            "use_ddp=true but WORLD_SIZE=1; run with torchrun for real multi-GPU training. Falling back to single process."
        )
    device = setup_distributed(should_use_ddp)
    world_size = get_world_size(should_use_ddp) if should_use_ddp else 1

    device_kind = "cuda" if torch.cuda.is_available() else "cpu"
    effective_precision = resolve_precision(cfg.training.precision, device_kind)
    forward_dtype = "bfloat16" if effective_precision == "bf16-mixed" else "float32"

    bundle = load_run_data(cfg, for_arch=cfg.model.arch, load_train=True)
    model = build_model_from_config(cfg, bundle.seq_len, len(bundle.tokenizer), forward_dtype)

    extra_summary = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "arch": cfg.model.arch,
        "model_params": cfg.model.params,
        "prep_manifest_path": str(bundle.prep_manifest_path),
        "run_data_manifest_path": str(bundle.run_data_manifest_path),
        "train_triples_sample": cfg.training.train_triples_sample,
        "config": cfg.to_dict(),
    }

    trainer = Trainer(
        model=model,
        device=device,
        cfg=cfg.training,
        paths=paths,
        bundle=bundle,
        seed=cfg.experiment.seed,
        effective_precision=effective_precision,
        should_use_ddp=should_use_ddp,
        world_size=world_size,
        extra_training_summary=extra_summary,
    )
    if is_main_process():
        save_json(paths.run_artifacts_path, extra_summary)
        logger.info("experiment=%s run_dir=%s device=%s", experiment_name, paths.run_dir, device)

    try:
        fit_summary = trainer.fit()
        if is_main_process():
            print(json.dumps(fit_summary, indent=2))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    setup_logging()
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except (ValueError, FileNotFoundError, KeyError) as error:
        logger.error("%s", error)
        sys.exit(2)
