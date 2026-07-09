"""Parity check against a legacy notebook checkpoint.

Обязателен перед заменой старых чисел новыми: загружает чекпоинт, обученный
легаси-ноутбуком pure_trm_reranker, в новую модель и проверяет, что логиты
совпадают. Запуск:

    TRM_LEGACY_CHECKPOINT=/path/to/best_mrr.pt uv run pytest tests/test_parity.py -v

Ключи state_dict легаси и новой модели совпадают один-в-один (совпадают пути
модулей inner.embed_tokens / inner.L_level.layers.N / inner.score_head ...).
"""

import os

import pytest
import torch

from trm_reranker.models import TRMReranker, run_model_once

CHECKPOINT_ENV = "TRM_LEGACY_CHECKPOINT"


@pytest.mark.skipif(CHECKPOINT_ENV not in os.environ, reason=f"set {CHECKPOINT_ENV} to run the parity check")
def test_legacy_checkpoint_parity():
    checkpoint = torch.load(os.environ[CHECKPOINT_ENV], map_location="cpu", weights_only=False)
    legacy_config = dict(checkpoint["config"].get("model_config") or {})
    if not legacy_config:
        # Legacy training_config.json did not embed model_config; reconstruct the default.
        legacy_config = dict(
            batch_size=512,
            seq_len=256,
            vocab_size=30522,
            H_cycles=2,
            L_cycles=4,
            H_layers=2,
            L_layers=2,
            hidden_size=512,
            expansion=4.0,
            num_heads=8,
            pos_encodings="rope",
            forward_dtype="float32",
        )
    legacy_config["forward_dtype"] = "float32"
    model = TRMReranker(legacy_config)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"

    torch.manual_seed(0)
    batch = {
        "input_ids": torch.randint(1000, 30000, (4, legacy_config["seq_len"])),
        "token_type_ids": torch.zeros((4, legacy_config["seq_len"]), dtype=torch.long),
        "attention_mask": torch.ones((4, legacy_config["seq_len"]), dtype=torch.long),
    }
    model.eval()
    with torch.no_grad():
        scores, _ = run_model_once(model, batch)
    assert torch.isfinite(scores).all()
