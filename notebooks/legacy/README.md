# Legacy notebooks (архив)

Исходные исследовательские ноутбуки до рефакторинга 0.2.0 (2026-07-08).
Не поддерживаются; сохранены как история результатов и эталон для parity-проверок
(`tests/test_parity.py`). Новый код — в `src/trm_reranker/` + `scripts/` + `configs/`.

Соответствие ноутбук → новый код:

| Ноутбук | Замена |
|---|---|
| `main_scripts/pure_trm_reranker.ipynb` | `scripts/train.py` + `configs/variants/trm.yaml` |
| `main_scripts/vanilla_transformer_rerank.ipynb` | `scripts/train.py` + `configs/variants/vanilla_{shallow,deep}.yaml` |
| `main_scripts/bert_encoder.ipynb` | `scripts/train.py` + `configs/variants/bert_frozen_trm.yaml` |
| `main_scripts/bert_encoder_only_ablation.ipynb` | `scripts/train.py` + `configs/variants/bert_frozen_linear.yaml` |
| `main_scripts/bert_embeds.ipynb` | не портирован (вариант с frozen BERT-эмбеддингами; портировать при необходимости) |
| `main_scripts/halt_exper.ipynb` | `scripts/train.py` (+ `measure_footprint.py` для latency) |
| `main_scripts/monot5_base_full_eval.ipynb` | не портирован (offline-оценка monoT5 baseline) |
| `latency_notebooks/*` | `scripts/measure_footprint.py` |
| `data_prep/00_prepare_data_cache.ipynb` | пока используется как есть (см. docs/data.md) |
| `data_prep/01_prepare_run_data_cache.ipynb` | пока используется как есть (см. docs/data.md) |

Известная проблема легаси-конфигов: в `pure_trm_reranker` и `halt_exper`
`EXPERIMENT_NAME_PREFIX = "trm_reranker_ablation_frozen_token_embeddings"` при
фактическом `FREEZE_TOKEN_EMBEDDINGS = False` — имена старых run-директорий не
доказывают конфиг; сверяйтесь с `training_config.json` внутри них.
