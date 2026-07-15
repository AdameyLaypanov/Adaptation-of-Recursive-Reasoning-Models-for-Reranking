#!/usr/bin/env python
"""Evaluate a trained checkpoint on the dev candidates (with per-query dump for E2).

Example:
    uv run python scripts/evaluate.py \
        --config configs/base.yaml configs/variants/trm.yaml configs/local.yaml \
        --checkpoint /path/to/best_mrr.pt \
        --split final --out-dir /path/to/eval_out
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from trm_reranker.config import load_experiment_config
from trm_reranker.evaluation.reranker_eval import evaluate_reranker
from trm_reranker.runtime import build_model_from_config, load_run_data
from trm_reranker.training.checkpoints import load_model_weights_for_eval
from trm_reranker.training.distributed import resolve_precision
from trm_reranker.utils import save_json, seed_everything, setup_logging

logger = logging.getLogger("scripts.evaluate")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split", choices=["epoch", "final"], default="final", help="epoch = quick dev subset, final = full dev"
    )
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default=None, help="Model tag in the TREC run file; default = experiment name")
    parser.add_argument("--print-config", action="store_true", help="Print the merged config as JSON and exit")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config, args.overrides)
    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
        return
    seed_everything(cfg.experiment.seed)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    device_kind = "cuda" if device.type == "cuda" else "cpu"
    effective_precision = resolve_precision(cfg.training.precision, device_kind)
    forward_dtype = "bfloat16" if effective_precision == "bf16-mixed" else "float32"

    bundle = load_run_data(cfg, for_arch=cfg.model.arch, load_train=False)
    model = build_model_from_config(cfg, bundle.seq_len, len(bundle.tokenizer), forward_dtype).to(device)
    load_model_weights_for_eval(model, checkpoint_path, device)

    if args.split == "final":
        if bundle.final_dev_candidates is None:
            raise ValueError("Run-data manifest has run_final_full_dev=false; use --split epoch")
        candidates, qrels = bundle.final_dev_candidates, bundle.final_dev_qrels
    else:
        candidates, qrels = bundle.epoch_dev_candidates, bundle.epoch_dev_qrels

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or cfg.experiment.name
    metrics = evaluate_reranker(
        model,
        bundle.encoder,
        candidates,
        qrels,
        bundle.dev_query_tokens,
        run_path=out_dir / f"{tag}_{args.split}.run",
        passage_token_getter=bundle.passage_token_getter,
        eval_batch_size=cfg.training.eval_batch_size,
        precision=effective_precision,
        query_limit=args.query_limit,
        per_query_path=out_dir / f"{tag}_{args.split}_per_query.csv",
        model_tag=tag,
    )
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["arch"] = cfg.model.arch
    metrics["seed"] = cfg.experiment.seed
    save_json(out_dir / f"{tag}_{args.split}_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


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
