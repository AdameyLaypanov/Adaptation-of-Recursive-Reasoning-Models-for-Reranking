import torch

from trm_reranker.models import TiedReranker, TRMReranker, VanillaReranker
from trm_reranker.training.optim import count_model_parameters, run_model_once

BATCH, SEQ, VOCAB, HIDDEN = 2, 16, 128, 64


def tiny_batch():
    torch.manual_seed(0)
    return {
        "input_ids": torch.randint(1, VOCAB, (BATCH, SEQ)),
        "token_type_ids": torch.zeros((BATCH, SEQ), dtype=torch.long),
        "attention_mask": torch.ones((BATCH, SEQ), dtype=torch.long),
    }


def trm_config(**kwargs):
    config = dict(
        batch_size=BATCH,
        seq_len=SEQ,
        vocab_size=VOCAB,
        H_cycles=2,
        L_cycles=2,
        H_layers=1,
        L_layers=1,
        hidden_size=HIDDEN,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
    )
    config.update(kwargs)
    return config


def vanilla_config(**kwargs):
    config = dict(
        batch_size=BATCH,
        seq_len=SEQ,
        vocab_size=VOCAB,
        num_layers=2,
        hidden_size=HIDDEN,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
    )
    config.update(kwargs)
    return config


def test_trm_forward_shape():
    torch.manual_seed(13)
    model = TRMReranker(trm_config()).eval()
    scores, outputs = run_model_once(model, tiny_batch())
    assert scores.shape == (BATCH,)
    assert "q_halt_logits" in outputs


def test_vanilla_forward_shape():
    torch.manual_seed(13)
    model = VanillaReranker(vanilla_config()).eval()
    scores, _ = run_model_once(model, tiny_batch())
    assert scores.shape == (BATCH,)


def test_tied_params_match_vanilla_but_depth_differs():
    torch.manual_seed(13)
    vanilla = VanillaReranker(vanilla_config(num_layers=2))
    torch.manual_seed(13)
    tied = TiedReranker({**vanilla_config(num_layers=2), "num_repeats": 5})
    assert count_model_parameters(tied) == count_model_parameters(vanilla)

    tied.eval()
    batch = tiny_batch()
    with torch.no_grad():
        scores_5, _ = run_model_once(tied, batch)
        tied.config.num_repeats = 1
        tied.inner.config.num_repeats = 1
        scores_1, _ = run_model_once(tied, batch)
    assert not torch.allclose(scores_5, scores_1)


def test_trm_disable_input_injection_changes_output():
    batch = tiny_batch()
    torch.manual_seed(13)
    base = TRMReranker(trm_config()).eval()
    torch.manual_seed(13)
    ablated = TRMReranker(trm_config(disable_input_injection=True)).eval()
    with torch.no_grad():
        scores_base, _ = run_model_once(base, batch)
        scores_ablated, _ = run_model_once(ablated, batch)
    assert not torch.allclose(scores_base, scores_ablated)


def test_trm_full_backprop_matches_forward_and_extends_grad():
    batch = tiny_batch()
    torch.manual_seed(13)
    one_step = TRMReranker(trm_config()).eval()
    torch.manual_seed(13)
    full = TRMReranker(trm_config(full_backprop=True)).eval()

    with torch.no_grad():
        scores_one, _ = run_model_once(one_step, batch)
        scores_full, _ = run_model_once(full, batch)
    # Forward values must be identical: the flag changes only gradient flow.
    assert torch.allclose(scores_one, scores_full, atol=1e-5)

    scores, _ = run_model_once(full, batch)
    scores.sum().backward()
    grads = [p.grad for p in full.parameters() if p.grad is not None]
    assert grads


def test_trm_one_step_gradient_flows():
    torch.manual_seed(13)
    model = TRMReranker(trm_config())
    scores, _ = run_model_once(model, tiny_batch())
    scores.sum().backward()
    assert model.inner.score_head.weight.grad is not None
    assert model.inner.embed_tokens.embedding_weight.grad is not None
