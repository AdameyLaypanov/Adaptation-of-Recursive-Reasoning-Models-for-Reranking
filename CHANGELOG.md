# Changelog

Формат — [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); версии — [SemVer](https://semver.org/).
До 1.0.0 minor-версии могут ломать совместимость.

## [0.2.0] — 2026-07-08

Рефакторинг репозитория: из набора Jupyter-ноутбуков — в python-пакет со скриптами,
конфигами и тестами. Логика моделей, лосса, метрик и подготовки данных перенесена
из ноутбуков без изменения численного поведения.

### Added
- Менеджер зависимостей **uv**: `pyproject.toml` с dependency-группами
  (`eval`, `bench`, `dev`), `.python-version`.
- Пакет `src/trm_reranker/`:
  - `models/` — единые блоки (`blocks.py`: Attention, SwiGLU, RMSNorm, RoPE — одна
    реализация для всех рук, гарантия выравнивания К1), `trm.py` (TRM/HRM-derived
    two-state реранкер + новые флаги `disable_input_injection` для E5 и
    `full_backprop` для абляции one-step gradient), `vanilla.py` (shallow/deep),
    `tied.py` (weight-tied deep, ALBERT-арм для E3), `bert.py` (BERT-энкодер +
    {TRM | linear} голова, флаг `freeze_encoder`; полный файнтюн для E1),
    `registry.py` (фабрика по имени арма).
  - `data/` — валидация манифестов, шардированный passage-token store,
    `PairwiseTripleDataset`, `encode_pair`/collate.
  - `training/` — trainer (DDP, bf16, grad accum, resume, чекпоинты best-loss /
    best-MRR / periodic), distributed-утилиты, LR-schedule.
  - `evaluation/` — метрики (MRR@10, hit@k, nDCG@10), реранкер-eval с BM25/candidate-order
    baseline и **выгрузкой per-query метрик**, `significance.py` (paired bootstrap +
    paired t-test) для E2.
  - `benchmarks/` — `latency.py` (warmup, mean/median/p95, per-config),
    `flops.py` (замер FLOPs для E0/E2), `params.py` (body/total params, checkpoint fp16,
    peak memory).
- `configs/`: `base.yaml` + конфиги рук `arms/{trm,vanilla_shallow,vanilla_deep,tied_deep,bert_frozen_trm,bert_frozen_linear,bert_finetune}.yaml`.
- `scripts/`: `train.py`, `evaluate.py`, `measure_footprint.py` (единый замер
  params/FLOPs/latency/памяти по всем рукам), `significance.py`.
- Тесты `tests/` (smoke: forward всех рук, эквивалентность tied-весов, encode_pair,
  метрики, значимость).
- Документация: переписан `README.md`; `docs/architecture.md`, `docs/experiments.md`
  (маппинг E0–E13 → конфиг/команда), `docs/data.md`.
- Папка `project_tracking/` для рабочих документов проекта (статусы плана, аудиты,
  чекпоинты) — отдельно от документации репозитория:
  `2026-07-08_plan_status.md`, `e0_alignment_audit_2026-07-08.md`.
- Этот `CHANGELOG.md`.

### Changed
- Тренировочные и latency-ноутбуки перемещены в `notebooks/legacy/` (архив;
  история результатов, не поддерживаются).
- Seed, LR, batch и прочие гиперпараметры — теперь параметры конфига/CLI,
  а не константы в ячейках.

### Known issues / несоответствия, унаследованные от ноутбуков
- В legacy-ноутбуках `pure_trm_reranker`/`halt_exper` имя эксперимента
  (`..._ablation_frozen_token_embeddings`) не соответствовало фактическому конфигу
  (`FREEZE_TOKEN_EMBEDDINGS=False`) — при разборе старых артефактов сверяться с
  `training_config.json`.
- Parity прогонов: перед заменой старых чисел новыми обязателен parity-прогон
  (загрузка старого чекпоинта в новую модель, сверка логитов) — см. `tests/test_parity.py`.

## [0.1.0] — 2026-07-08 (до рефакторинга)

Исходное состояние: исследовательские Jupyter-ноутбуки.

### Added
- `data_prep/00_prepare_data_cache.ipynb`, `01_prepare_run_data_cache.ipynb` —
  кэши MS MARCO (токенизация, шардированный passage store, dev-кандидаты/qrels).
- `main_scripts/`: `pure_trm_reranker` (TRM-арм), `vanilla_transformer_rerank`
  (shallow/deep-армы), `bert_encoder` (frozen BERT + TRM-голова), `bert_encoder_only_ablation`
  (frozen BERT + linear-голова), `bert_embeds` (TRM + frozen BERT-эмбеддинги),
  `halt_exper` (TRM + замеры latency), `monot5_base_full_eval` (baseline monoT5).
- `latency_notebooks/`: замеры latency архитектур проекта и SOTA-реранкеров.
- Планы: `trm_reranker_wsdm2027_plan.md` (E0–E9), `trm_reranker_plan_addendum.md`
  (E10–E13), обзор литературы `trm_literature_review_ru.md`.
