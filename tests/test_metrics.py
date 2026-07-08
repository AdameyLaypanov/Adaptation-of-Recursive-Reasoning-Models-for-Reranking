from trm_reranker.evaluation.metrics import ndcg_at_k, ranking_metrics_at_10, reciprocal_rank_at_k


def test_mrr():
    assert reciprocal_rank_at_k([1, 2, 3], {2}) == 0.5
    assert reciprocal_rank_at_k([1, 2, 3], {9}) == 0.0
    assert reciprocal_rank_at_k(list(range(20)), {15}, k=10) == 0.0


def test_ndcg_perfect_ranking():
    assert ndcg_at_k([1, 2, 3], {1}, k=10) == 1.0


def test_ranking_metrics_keys():
    metrics = ranking_metrics_at_10([1, 2, 3], {3})
    assert set(metrics) == {"mrr@10", "hit@1", "hit@3", "hit@5", "hit@10", "ndcg@10"}
    assert metrics["hit@1"] == 0.0
    assert metrics["hit@3"] == 1.0
