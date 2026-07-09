"""Grouped multi-negative dataset/collate and the InfoNCE <-> pairwise equivalence."""

import pytest
import torch

from trm_reranker.data.datasets import (
    GroupedTripleDataset,
    group_pairwise_triples,
    load_hard_negative_groups,
    make_grouped_collate,
)
from trm_reranker.data.encoding import PairEncoder, collate_encoded_pairs
from trm_reranker.models import TRMReranker
from trm_reranker.training.optim import compute_infonce_batch_metrics, compute_pairwise_batch_metrics

SEQ = 20
VOCAB = 100

QUERY_TOKENS = {1: [10, 11], 2: [12, 13]}
PASSAGE_TOKENS = {100: [20] * 4, 101: [21] * 4, 102: [22] * 4, 103: [23] * 4}


def make_encoder():
    return PairEncoder(cls_id=1, sep_id=2, pad_id=0, seq_len=SEQ, max_query_len=4, max_doc_len=8)


def test_group_pairwise_triples_merges_negatives():
    triples = [(1, 100, 101), (1, 100, 102), (1, 100, 101), (2, 103, 100)]
    groups = dict(((qid, pos), negs) for qid, pos, negs in group_pairwise_triples(triples))
    assert groups[(1, 100)] == (101, 102)  # deduplicated, order preserved
    assert groups[(2, 103)] == (100,)


def test_grouped_dataset_samples_with_replacement_when_pool_is_small():
    groups = [(1, 100, (101, 102))]
    dataset = GroupedTripleDataset(groups, QUERY_TOKENS, PASSAGE_TOKENS.__getitem__, num_negatives=5, seed=13)
    item = dataset[0]
    assert len(item["neg_tokens_list"]) == 5
    assert item["pos_tokens"] == PASSAGE_TOKENS[100]


def test_grouped_dataset_skips_qids_without_query_tokens():
    groups = [(1, 100, (101,)), (999, 100, (101,))]
    dataset = GroupedTripleDataset(groups, QUERY_TOKENS, PASSAGE_TOKENS.__getitem__, num_negatives=1, seed=13)
    assert len(dataset) == 1
    with pytest.raises(ValueError, match="No usable groups"):
        GroupedTripleDataset([(999, 100, (101,))], QUERY_TOKENS, PASSAGE_TOKENS.__getitem__, num_negatives=1)


def test_grouped_collate_layout_positive_first(tmp_path):
    encoder = make_encoder()
    groups = [(1, 100, (101, 102)), (2, 103, (100, 101))]
    dataset = GroupedTripleDataset(groups, QUERY_TOKENS, PASSAGE_TOKENS.__getitem__, num_negatives=2, seed=13)
    collate = make_grouped_collate(encoder)
    batch = collate([dataset[0], dataset[1]])

    group_size = 3  # 1 pos + 2 negs
    assert batch["input_ids"].shape == (2 * group_size, SEQ)
    expected_pos_row = torch.tensor(encoder.encode_pair(QUERY_TOKENS[1], PASSAGE_TOKENS[100])["input_ids"])
    assert torch.equal(batch["input_ids"][0], expected_pos_row)
    expected_second_group_pos = torch.tensor(encoder.encode_pair(QUERY_TOKENS[2], PASSAGE_TOKENS[103])["input_ids"])
    assert torch.equal(batch["input_ids"][group_size], expected_second_group_pos)


def test_load_hard_negative_groups_validates(tmp_path):
    path = tmp_path / "groups.jsonl"
    path.write_text('{"qid": 1, "pos_pid": 100, "neg_pids": [101, 102]}\n')
    assert load_hard_negative_groups(path) == [(1, 100, (101, 102))]

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"qid": 1, "neg_pids": [101]}\n')
    with pytest.raises(ValueError, match="Bad hard-negatives record"):
        load_hard_negative_groups(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="no records"):
        load_hard_negative_groups(empty)


def test_infonce_with_one_negative_equals_pairwise_logistic():
    """InfoNCE over (pos, neg) groups is exactly -logsigmoid(pos - neg)."""
    torch.manual_seed(13)
    model = TRMReranker(
        dict(
            batch_size=2,
            seq_len=SEQ,
            vocab_size=VOCAB,
            H_cycles=1,
            L_cycles=1,
            H_layers=1,
            L_layers=1,
            hidden_size=32,
            expansion=2.0,
            num_heads=4,
            pos_encodings="rope",
        )
    ).eval()

    encoder = make_encoder()
    pairs = [
        (QUERY_TOKENS[1], PASSAGE_TOKENS[100], PASSAGE_TOKENS[101]),
        (QUERY_TOKENS[2], PASSAGE_TOKENS[102], PASSAGE_TOKENS[103]),
    ]
    pos_batch = collate_encoded_pairs([encoder.encode_pair(q, pos) for q, pos, _ in pairs])
    neg_batch = collate_encoded_pairs([encoder.encode_pair(q, neg) for q, _, neg in pairs])
    interleaved = collate_encoded_pairs(
        [encoded for q, pos, neg in pairs for encoded in (encoder.encode_pair(q, pos), encoder.encode_pair(q, neg))]
    )

    with torch.no_grad():
        pairwise = compute_pairwise_batch_metrics(model, pos_batch, neg_batch)
        grouped = compute_infonce_batch_metrics(model, interleaved, group_size=2, temperature=1.0)

    assert torch.allclose(pairwise["loss"], grouped["loss"], atol=1e-6)
    assert torch.allclose(pairwise["margin"], grouped["margin"], atol=1e-6)
    assert torch.allclose(pairwise["pairwise_acc"], grouped["pairwise_acc"])


def test_infonce_rejects_indivisible_batch():
    torch.manual_seed(13)
    model = TRMReranker(
        dict(
            batch_size=1,
            seq_len=SEQ,
            vocab_size=VOCAB,
            H_cycles=1,
            L_cycles=1,
            H_layers=1,
            L_layers=1,
            hidden_size=32,
            expansion=2.0,
            num_heads=4,
            pos_encodings="rope",
        )
    ).eval()
    encoder = make_encoder()
    batch = collate_encoded_pairs([encoder.encode_pair(QUERY_TOKENS[1], PASSAGE_TOKENS[100])] * 3)
    with pytest.raises(ValueError, match="not divisible"):
        compute_infonce_batch_metrics(model, batch, group_size=2)
