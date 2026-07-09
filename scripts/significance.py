#!/usr/bin/env python
"""Парные тесты значимости по запросам (E2): paired bootstrap + paired t-test.

Вход — per-query CSV, которые пишут scripts/evaluate.py и epoch/final eval
трейнера (*_per_query.csv).

Example:
    uv run python scripts/significance.py \
        --run-a trm_final_per_query.csv \
        --run-b vanilla_deep_final_per_query.csv \
        --metrics trm_mrr@10 trm_ndcg@10 \
        --out significance_trm_vs_deep.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trm_reranker.evaluation.significance import compare_runs
from trm_reranker.utils import setup_logging

logger = logging.getLogger("scripts.significance")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-a", required=True, help="per-query CSV of run A")
    parser.add_argument("--run-b", required=True, help="per-query CSV of run B")
    parser.add_argument("--metrics", nargs="+", default=["trm_mrr@10", "trm_ndcg@10"])
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = []
    for metric in args.metrics:
        result = compare_runs(args.run_a, args.run_b, metric=metric, n_bootstrap=args.n_bootstrap, seed=args.seed)
        results.append(result.to_dict())
        print(json.dumps(result.to_dict(), indent=2))

    if args.out:
        payload = {"run_a": str(args.run_a), "run_b": str(args.run_b), "results": results}
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved: %s", args.out)


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
