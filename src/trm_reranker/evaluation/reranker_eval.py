"""Reranker evaluation over cached BM25 candidates.

Behaviour matches the legacy ``evaluate_reranker`` (same aggregate metrics,
same TREC run output, BM25/candidate-order baseline included), with one
addition needed for E2: optional per-query metric dump (``per_query_path``)
so paired significance tests can be run afterwards.
"""

import csv
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from ..data.datasets import iter_grouped_candidates
from ..data.encoding import PairEncoder, collate_encoded_pairs, move_batch_to_device
from ..models.inference import run_model_once
from ..utils import get_autocast_context, make_tqdm, model_device
from .metrics import RANKING_METRIC_NAMES, ranking_metrics_at_10


def score_pid_batch(
    model,
    encoder: PairEncoder,
    query_tokens: list[int],
    pid_batch: Iterable[int],
    passage_token_getter,
    precision: str = "32-true",
):
    encoded_pairs = [encoder.encode_pair(query_tokens, list(passage_token_getter(int(pid)))) for pid in pid_batch]
    device = model_device(model)
    batch = move_batch_to_device(collate_encoded_pairs(encoded_pairs), device)
    with torch.no_grad(), get_autocast_context(device, precision):
        scores, _ = run_model_once(model, batch)
    return scores.detach().float().cpu().tolist()


def evaluate_reranker(
    model,
    encoder: PairEncoder,
    candidates_artifact,
    qrels: dict[int, set],
    query_token_map: dict[int, list[int]],
    run_path: Path,
    passage_token_getter,
    eval_batch_size: int = 512,
    precision: str = "32-true",
    query_limit: int | None = None,
    per_query_path: Path | None = None,
    model_tag: str = "model",
    show_progress: bool = True,
):
    was_training = model.training
    model.eval()
    run_path = Path(run_path)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_metric_values = {name: [] for name in RANKING_METRIC_NAMES}
    model_metric_values = {name: [] for name in RANKING_METRIC_NAMES}
    per_query_rows = []
    evaluated_queries = 0

    total_queries = len(candidates_artifact["qid_order"])
    if query_limit is not None:
        total_queries = min(int(query_limit), total_queries)

    with run_path.open("w", encoding="utf-8") as run_handle:
        for qid, pid_values, bm25_ranks in make_tqdm(
            iter_grouped_candidates(candidates_artifact, query_limit=query_limit),
            total=total_queries,
            desc=f"Evaluate {run_path.name}",
            disable=not show_progress,
        ):
            qid = int(qid)
            if qid not in query_token_map or qid not in qrels:
                continue
            candidate_pids = [int(pid) for pid in pid_values]
            candidate_bm25_ranks = [int(rank) for rank in bm25_ranks]
            if not candidate_pids:
                continue
            if len(candidate_pids) != len(candidate_bm25_ranks):
                raise ValueError(f"Candidate pid count and BM25 rank count differ for qid={qid}")
            scores: list[float] = []
            for start in range(0, len(candidate_pids), eval_batch_size):
                pid_batch = candidate_pids[start : start + eval_batch_size]
                scores.extend(
                    score_pid_batch(
                        model, encoder, query_token_map[qid], pid_batch, passage_token_getter, precision=precision
                    )
                )
            reranked = sorted(zip(candidate_pids, scores, strict=True), key=lambda item: item[1], reverse=True)
            reranked_pids = [pid for pid, _ in reranked]
            bm25_ranked = sorted(zip(candidate_pids, candidate_bm25_ranks, strict=True), key=lambda item: item[1])
            bm25_ranked_pids = [pid for pid, _ in bm25_ranked]
            relevant_pids = qrels[qid]
            bm25_metrics = ranking_metrics_at_10(bm25_ranked_pids, relevant_pids)
            model_metrics = ranking_metrics_at_10(reranked_pids, relevant_pids)
            for metric_name, metric_value in bm25_metrics.items():
                bm25_metric_values[metric_name].append(metric_value)
            for metric_name, metric_value in model_metrics.items():
                model_metric_values[metric_name].append(metric_value)
            if per_query_path is not None:
                row = {"qid": qid, "num_candidates": len(candidate_pids)}
                for metric_name in RANKING_METRIC_NAMES:
                    row[f"bm25_{metric_name}"] = bm25_metrics[metric_name]
                    row[f"trm_{metric_name}"] = model_metrics[metric_name]
                per_query_rows.append(row)
            evaluated_queries += 1
            for rank, (pid, score) in enumerate(reranked, start=1):
                run_handle.write(f"{qid} Q0 {pid} {rank} {score:.6f} {model_tag}\n")

    if was_training:
        model.train()

    if per_query_path is not None:
        per_query_path = Path(per_query_path)
        per_query_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["qid", "num_candidates"] + [
            f"{prefix}_{name}" for prefix in ("bm25", "trm") for name in RANKING_METRIC_NAMES
        ]
        with per_query_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_query_rows:
                writer.writerow(row)

    metrics = {"queries_evaluated": evaluated_queries}
    for metric_name in RANKING_METRIC_NAMES:
        values = bm25_metric_values[metric_name]
        metrics[f"bm25_{metric_name}"] = float(np.mean(values)) if values else 0.0
    for metric_name in RANKING_METRIC_NAMES:
        values = model_metric_values[metric_name]
        metrics[f"trm_{metric_name}"] = float(np.mean(values)) if values else 0.0
    metrics["run_path"] = str(run_path)
    if per_query_path is not None:
        metrics["per_query_path"] = str(per_query_path)
    return metrics
