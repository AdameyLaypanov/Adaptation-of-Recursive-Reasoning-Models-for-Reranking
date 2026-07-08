#!/usr/bin/env python
"""Train a reranker arm from YAML configs.

Examples:
    # smoke-прогон TRM-арма
    uv run python scripts/train.py \
        --config configs/base.yaml configs/arms/trm.yaml \
        --set training.max_train_steps=20 \
        --set data.run_data_manifest_path=/path/to/run_data_manifest.json \
        --set experiment.output_root=/path/to/output

    # DDP на 2 GPU
    uv run torchrun --nproc_per_node=2 scripts/train.py \
        --config configs/base.yaml configs/arms/vanilla_deep.yaml \
        --set training.use_ddp=true --set training.devices=2

    # мультисид (E2)
    for seed in 13 17 42; do
        uv run python scripts/train.py --config configs/base.yaml configs/arms/trm.yaml \
            --set experiment.seed=$seed --run-id seed$seed
    done
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from trm_reranker.config import load_experiment_config  # noqa: E402
from trm_reranker.data.datasets import PairwiseTripleDataset, make_pairwise_collate  # noqa: E402
from trm_reranker.runtime import build_model_from_config, load_run_data, resolve_run_id  # noqa: E402
from trm_reranker.training.checkpoints import RunPaths  # noqa: E402
from trm_reranker.training.distributed import (  # noqa: E402
    cleanup_distributed,
    get_world_size,
    is_main_process,
    resolve_precision,
    setup_distributed,
)
from trm_reranker.training.trainer import Trainer  # noqa: E402
from trm_reranker.utils import save_json, seed_everything  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", required=True, help="YAML configs, merged left to right (base first)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override: section.key=value")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume-from", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = list(args.overrides)
    if args.seed is not None:
        overrides.append(f"experiment.seed={args.seed}")
    if args.output_root is not None:
        overrides.append(f"experiment.output_root={args.output_root}")
    if args.resume_from is not None:
        overrides.append("training.resume_from_checkpoint=true")
        overrides.append(f"training.resume_checkpoint_path={args.resume_from}")
    cfg = load_experiment_config(args.config, overrides)

    seed_everything(cfg.experiment.seed)

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    should_use_ddp = cfg.training.use_ddp and world_size_env > 1
    if cfg.training.use_ddp and world_size_env == 1 and is_main_process():
        print("use_ddp=true but WORLD_SIZE=1; run with torchrun for real multi-GPU training. Falling back to single process.")
    device = setup_distributed(should_use_ddp)
    world_size = get_world_size(should_use_ddp) if should_use_ddp else 1

    device_kind = "cuda" if torch.cuda.is_available() else "cpu"
    effective_precision = resolve_precision(cfg.training.precision, device_kind)
    forward_dtype = "bfloat16" if effective_precision == "bf16-mixed" else "float32"

    bundle = load_run_data(cfg, for_arch=cfg.model.arch, load_train=True)
    model = build_model_from_config(cfg, bundle.seq_len, len(bundle.tokenizer), forward_dtype)

    run_id = resolve_run_id(cfg.experiment.run_id)
    experiment_name = f"{cfg.experiment.name}_seed{cfg.experiment.seed}_{run_id}"
    if not cfg.experiment.output_root:
        raise ValueError("experiment.output_root is not configured")
    paths = RunPaths.create(cfg.experiment.output_root, experiment_name)

    train_dataset = PairwiseTripleDataset(bundle.sampled_train_triples_path, bundle.train_query_tokens, bundle.train_passage_tokens)
    pairwise_collate = make_pairwise_collate(bundle.encoder)

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
        train_dataset=train_dataset,
        pairwise_collate=pairwise_collate,
        encoder=bundle.encoder,
        dev_query_tokens=bundle.dev_query_tokens,
        epoch_dev_candidates=bundle.epoch_dev_candidates,
        epoch_dev_qrels=bundle.epoch_dev_qrels,
        passage_token_getter=bundle.passage_token_getter,
        seed=cfg.experiment.seed,
        effective_precision=effective_precision,
        should_use_ddp=should_use_ddp,
        world_size=world_size,
        epoch_dev_mode_label=bundle.epoch_dev_mode_label,
        final_dev_candidates=bundle.final_dev_candidates,
        final_dev_qrels=bundle.final_dev_qrels,
        run_final_full_dev=bundle.run_final_full_dev,
        extra_training_summary=extra_summary,
    )
    if is_main_process():
        save_json(paths.run_artifacts_path, extra_summary)
        print(json.dumps({"experiment_name": experiment_name, "run_dir": str(paths.run_dir), "device": str(device)}, indent=2))

    try:
        fit_summary = trainer.fit()
        if is_main_process():
            print(json.dumps(fit_summary, indent=2))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
