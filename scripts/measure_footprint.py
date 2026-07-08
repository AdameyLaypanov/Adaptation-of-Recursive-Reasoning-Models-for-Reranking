#!/usr/bin/env python
"""Единый замер params / FLOPs / latency / peak memory для всех рук (E0, E2).

Каждая строка меряется реально (никаких скопированных latency): один и тот же
девайс, фиксированные batch и seq_len, warmup, mean/median/p95.

Example:
    uv run python scripts/measure_footprint.py \
        --arms configs/arms/trm.yaml configs/arms/vanilla_shallow.yaml \
               configs/arms/vanilla_deep.yaml configs/arms/tied_deep.yaml \
        --base configs/base.yaml \
        --batch-size 1 --seq-len 256 \
        --out footprint_table.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from trm_reranker.benchmarks.flops import measure_forward_flops  # noqa: E402
from trm_reranker.benchmarks.latency import measure_forward_latency  # noqa: E402
from trm_reranker.benchmarks.params import summarize_model_footprint  # noqa: E402
from trm_reranker.config import load_experiment_config  # noqa: E402
from trm_reranker.runtime import build_model_from_config, build_synthetic_batch  # noqa: E402
from trm_reranker.utils import seed_everything  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", required=True, help="Arm YAML configs to measure")
    parser.add_argument("--base", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=50)
    parser.add_argument("--device", default=None, help="cuda / cpu; default: cuda if available")
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"))
    seed_everything(13)

    rows = []
    for arm_config in args.arms:
        cfg = load_experiment_config([args.base, arm_config], args.overrides)
        model = build_model_from_config(cfg, args.seq_len, args.vocab_size, "float32").to(device).eval()
        batch = build_synthetic_batch(cfg.model.arch, args.batch_size, args.seq_len, args.vocab_size)
        batch = {key: value.to(device) for key, value in batch.items()}

        row = {"arm": cfg.experiment.name, "arch": cfg.model.arch, "config": str(arm_config)}
        row.update(summarize_model_footprint(model, batch))
        row.update(measure_forward_latency(model, batch, warmup_steps=args.warmup_steps, measure_steps=args.measure_steps))
        if not args.skip_flops:
            row.update(measure_forward_flops(model, batch))
        rows.append(row)
        print(json.dumps(row, indent=2))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"device": str(device), "rows": rows}, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
