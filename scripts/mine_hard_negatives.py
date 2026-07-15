#!/usr/bin/env python
"""Mine hard negatives for training from retriever rankings over train queries.

Inputs are rankings from ANY first-stage retriever (BM25 run, dense retriever,
or the published msmarco-hard-negatives converted to a run): mining itself does
not run a retriever. For each query the script drops known positives (qrels),
keeps candidates inside the [--rank-min, --rank-max] band (the "hard" zone:
high-ranked but not relevant), and samples up to --num-negatives per positive.

Output: JSONL with one {"qid", "pos_pid", "neg_pids"} object per line — feed it
to training via ``data.hard_negatives_path`` (see docs/running.md). Store more
negatives than ``training.num_negatives``: the dataset re-samples per epoch.

Supported candidate formats (autodetected per line):
  - TREC run:  qid Q0 pid rank score tag
  - TSV:       qid \\t pid \\t rank
Supported qrels formats:
  - MS MARCO:  qid 0 pid rel   (rel > 0 = positive)
  - TSV:       qid \\t pid

Example:
    uv run python scripts/mine_hard_negatives.py \\
        --candidates /path/to/train_bm25_top1000.run \\
        --qrels /path/to/qrels.train.tsv \\
        --rank-min 10 --rank-max 200 --num-negatives 30 \\
        --out /path/to/hard_negatives.jsonl
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trm_reranker.utils import make_tqdm, setup_logging

logger = logging.getLogger("scripts.mine_hard_negatives")


def read_candidates(path: Path):
    """qid -> [(pid, rank), ...] from a TREC run or a qid/pid/rank TSV."""
    candidates = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(make_tqdm(handle, desc=f"Read {Path(path).name}"), start=1):
            parts = line.split()
            if not parts:
                continue
            try:
                if len(parts) >= 6:  # TREC: qid Q0 pid rank score tag
                    qid, pid, rank = int(parts[0]), int(parts[2]), int(parts[3])
                elif len(parts) >= 3:  # TSV: qid pid rank
                    qid, pid, rank = int(parts[0]), int(parts[1]), int(parts[2])
                else:
                    raise ValueError(f"expected 3 (TSV) or 6+ (TREC run) columns, got {len(parts)}")
            except ValueError as error:
                raise ValueError(f"Bad candidates line {path}:{line_number}: {error}") from error
            candidates[qid].append((pid, rank))
    if not candidates:
        raise ValueError(f"Candidates file is empty: {path}")
    return candidates


def read_qrels(path: Path):
    """qid -> {positive pids} from MS MARCO qrels (qid 0 pid rel) or a qid/pid TSV."""
    positives = defaultdict(set)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue
            try:
                if len(parts) >= 4:  # qid 0 pid rel
                    qid, pid, relevance = int(parts[0]), int(parts[2]), int(parts[3])
                    if relevance <= 0:
                        continue
                elif len(parts) >= 2:  # qid pid
                    qid, pid = int(parts[0]), int(parts[1])
                else:
                    raise ValueError(f"expected 2 or 4+ columns, got {len(parts)}")
            except ValueError as error:
                raise ValueError(f"Bad qrels line {path}:{line_number}: {error}") from error
            positives[qid].add(pid)
    if not positives:
        raise ValueError(f"Qrels file has no positives: {path}")
    return positives


def mine_hard_negatives(
    candidates,
    positives,
    num_negatives: int,
    rank_min: int,
    rank_max: int,
    min_negatives: int,
    seed: int,
    max_queries=None,
):
    """Yield (qid, pos_pid, [neg_pids]) groups; returns (records, stats)."""
    rng = random.Random(seed)
    records = []
    stats = {
        "queries_with_candidates": len(candidates),
        "queries_without_qrels": 0,
        "queries_too_few_negatives": 0,
        "queries_emitted": 0,
        "groups_emitted": 0,
    }
    for qid in sorted(candidates):
        if max_queries is not None and stats["queries_emitted"] >= max_queries:
            break
        positive_pids = positives.get(qid)
        if not positive_pids:
            stats["queries_without_qrels"] += 1
            continue
        negative_pids = [
            pid for pid, rank in candidates[qid] if rank_min <= rank <= rank_max and pid not in positive_pids
        ]
        negative_pids = list(dict.fromkeys(negative_pids))
        if len(negative_pids) < min_negatives:
            stats["queries_too_few_negatives"] += 1
            continue
        stats["queries_emitted"] += 1
        for pos_pid in sorted(positive_pids):
            sampled = rng.sample(negative_pids, min(num_negatives, len(negative_pids)))
            records.append({"qid": qid, "pos_pid": pos_pid, "neg_pids": sampled})
            stats["groups_emitted"] += 1
    return records, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--candidates", required=True, help="Retriever rankings for train queries (TREC run or qid/pid/rank TSV)"
    )
    parser.add_argument("--qrels", required=True, help="Train qrels (MS MARCO 'qid 0 pid rel' or qid/pid TSV)")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument(
        "--num-negatives", type=int, default=30, help="Negatives stored per positive (training re-samples)"
    )
    parser.add_argument("--rank-min", type=int, default=10, help="Skip top ranks likely to be unlabeled positives")
    parser.add_argument("--rank-max", type=int, default=200, help="Below this rank negatives get too easy")
    parser.add_argument("--min-negatives", type=int, default=4, help="Skip queries with fewer in-band negatives")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if args.rank_min > args.rank_max:
        raise ValueError(f"--rank-min {args.rank_min} must be <= --rank-max {args.rank_max}")
    if args.num_negatives < 1:
        raise ValueError("--num-negatives must be >= 1")

    candidates = read_candidates(Path(args.candidates))
    positives = read_qrels(Path(args.qrels))
    records, stats = mine_hard_negatives(
        candidates,
        positives,
        num_negatives=args.num_negatives,
        rank_min=args.rank_min,
        rank_max=args.rank_max,
        min_negatives=args.min_negatives,
        seed=args.seed,
        max_queries=args.max_queries,
    )
    if not records:
        raise ValueError("Mining produced no groups; check qrels/candidates qid overlap and the rank band")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    stats.update(
        {
            "out_path": str(out_path),
            "num_negatives": args.num_negatives,
            "rank_band": [args.rank_min, args.rank_max],
            "seed": args.seed,
        }
    )
    print(json.dumps(stats, indent=2))


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
