"""Ranking metrics (ported from the legacy notebooks)."""

import math

RANKING_METRIC_NAMES = ["mrr@10", "hit@1", "hit@3", "hit@5", "hit@10", "ndcg@10"]


def reciprocal_rank_at_k(ranked_pids: list[int], relevant_pids: set, k: int = 10) -> float:
    for rank, pid in enumerate(ranked_pids[:k], start=1):
        if pid in relevant_pids:
            return 1.0 / rank
    return 0.0


def hit_at_k(ranked_pids: list[int], relevant_pids: set, k: int) -> float:
    return 1.0 if any(pid in relevant_pids for pid in ranked_pids[:k]) else 0.0


def dcg_at_k(ranked_pids: list[int], relevant_pids: set, k: int) -> float:
    dcg = 0.0
    for rank, pid in enumerate(ranked_pids[:k], start=1):
        rel = 1.0 if pid in relevant_pids else 0.0
        if rel:
            dcg += rel / math.log2(rank + 1)
    return dcg


def ideal_dcg_at_k(num_relevant: int, k: int) -> float:
    ideal_count = min(int(num_relevant), k)
    if ideal_count <= 0:
        return 0.0
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))


def ndcg_at_k(ranked_pids: list[int], relevant_pids: set, k: int) -> float:
    idcg = ideal_dcg_at_k(len(relevant_pids), k)
    if idcg <= 0.0:
        return 0.0
    return dcg_at_k(ranked_pids, relevant_pids, k) / idcg


def ranking_metrics_at_10(ranked_pids: list[int], relevant_pids: set) -> dict[str, float]:
    return {
        "mrr@10": reciprocal_rank_at_k(ranked_pids, relevant_pids, k=10),
        "hit@1": hit_at_k(ranked_pids, relevant_pids, k=1),
        "hit@3": hit_at_k(ranked_pids, relevant_pids, k=3),
        "hit@5": hit_at_k(ranked_pids, relevant_pids, k=5),
        "hit@10": hit_at_k(ranked_pids, relevant_pids, k=10),
        "ndcg@10": ndcg_at_k(ranked_pids, relevant_pids, k=10),
    }
