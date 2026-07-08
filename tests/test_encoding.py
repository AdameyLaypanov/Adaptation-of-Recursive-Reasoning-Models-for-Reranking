from trm_reranker.data.encoding import PairEncoder, collate_encoded_pairs


def make_encoder(**kwargs):
    defaults = dict(cls_id=101, sep_id=102, pad_id=0, seq_len=16, max_query_len=4, max_doc_len=6)
    defaults.update(kwargs)
    return PairEncoder(**defaults)


def test_encode_pair_layout():
    encoder = make_encoder()
    encoded = encoder.encode_pair([11, 12, 13], [21, 22])
    assert encoded["input_ids"][:8] == [101, 11, 12, 13, 102, 21, 22, 102]
    assert encoded["input_ids"][8:] == [0] * 8
    assert encoded["token_type_ids"][:8] == [0, 1, 1, 1, 0, 2, 2, 0]
    assert encoded["attention_mask"] == [1] * 8 + [0] * 8
    assert len(encoded["input_ids"]) == 16
    assert "bert_token_type_ids" not in encoded


def test_encode_pair_truncation():
    encoder = make_encoder()
    encoded = encoder.encode_pair(list(range(1, 100)), list(range(100, 200)))
    # 1 CLS + 4 query + 1 SEP + 6 doc + 1 SEP = 13 real tokens
    assert sum(encoded["attention_mask"]) == 13


def test_encode_pair_bert_token_types():
    encoder = make_encoder(emit_bert_token_type_ids=True)
    encoded = encoder.encode_pair([11, 12], [21, 22, 23])
    # [CLS] q q [SEP] d d d [SEP] -> bert types: 0 0 0 0 1 1 1 1
    assert encoded["bert_token_type_ids"][:8] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert encoded["bert_token_type_ids"][8:] == [0] * 8


def test_encode_pair_overflow_raises():
    encoder = make_encoder(seq_len=8)
    try:
        encoder.encode_pair([1, 2, 3, 4], [5, 6, 7, 8, 9, 10])
    except ValueError:
        return
    raise AssertionError("expected ValueError for overflow")


def test_collate_shapes():
    encoder = make_encoder()
    batch = collate_encoded_pairs([encoder.encode_pair([1], [2]), encoder.encode_pair([3], [4])])
    assert batch["input_ids"].shape == (2, 16)
    assert batch["input_ids"].dtype.is_floating_point is False
