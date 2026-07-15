# Changelog

Формат — [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); версии — [SemVer](https://semver.org/).
До 1.0.0 minor-версии могут ломать совместимость.

## [0.4.0] — 2026-07-10

Hard negatives + выбор лосса (инфраструктура E8), e2e-тесты трейнера,
инструменты качества кода (CI, Makefile, pre-commit, ruff format).

### Added
- **Выбор лосса**: `training.loss = pairwise_logistic | infonce` (часть общего
  рецепта K2 — задаётся в base.yaml для всех вариантов). InfoNCE — softmax-CE
  по группам (позитив + `training.num_negatives` негативов,
  `training.infonce_temperature`); при K=1 математически совпадает с pairwise
  logistic (закреплено тестом). Метрики лога сохраняют прежние колонки
  (`pairwise_acc` = top-1 по группе, `margin` = позитив минус самый трудный негатив).
- **Hard-negative mining**: `scripts/mine_hard_negatives.py` — из ранжирований
  train-запросов любого ретривера (TREC run / TSV) + qrels выдаёт JSONL
  `{qid, pos_pid, neg_pids}`: исключает позитивы, режет полосу рангов
  `[--rank-min, --rank-max]`, сэмплирует до `--num-negatives` на позитив.
  Подключается через `data.hard_negatives_path`; токены любых pid берутся из
  полного шард-стора (`GroupedTripleDataset` с фолбэком мимо train-сабсета).
- `training.keep_last_step_checkpoints` — прунинг старых периодических
  `step_*.pt` (именованные чекпоинты не трогаются).
- **Тесты трейнера (e2e на синтетике)**: сессия из 5 шагов -> чекпоинты ->
  resume до конца эпохи -> финальный full-dev eval; защита от смены
  step-accounting при resume; InfoNCE-обучение с hard negatives; прунинг;
  плюс тесты grouped-датасета/коллейта и майнинга (парсинг форматов, полоса
  рангов, исключение позитивов, CLI end-to-end).
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) — ruff check + format
  check + pytest на push/PR (uv с кэшем).
- `Makefile`: `install / test / lint / fmt / fmt-check / check / parity / hooks`.
- `.pre-commit-config.yaml` (локальные хуки ruff check/format через uv);
  `pre-commit` добавлен в dev-зависимости.
- Точечные докстринги: `load_run_data` (цепочка манифестов),
  `resolve_step_count` (приоритеты), `Trainer.fit` (ключи summary).
- Документация: docs/running.md — разделы про hard negatives/InfoNCE и прунинг
  чекпоинтов; docs/experiments.md — E8 переведён в «инфраструктура готова»
  с командами.

### Changed
- Ruff: включены правила `B`, `UP`, `SIM`, `RUF` (+`ruff format` как формат
  всего репозитория); весь код приведён к ним. `RUF001-003` отключены
  (кириллица в строках/комментариях).
- `Trainer` строит датасет/коллейт/лосс по `training.loss`
  (`_build_training_objective`); тренировочный цикл стал независим от формата
  батча (`move_to_device` для dict и кортежей).

## [0.3.0] — 2026-07-10

Чистка после рефакторинга 0.2.0: баги конфиг-пайплайна, дедупликация моделей,
логирование. Численное поведение не менялось (state_dict-ключи и forward
идентичны 0.2.0 — закреплено тестом `test_trm_state_dict_keys_stable`).

### Fixed
- `scripts/train.py --run-id` молча игнорировался (мультисид-пример из докстринга
  не работал): теперь пробрасывается в `experiment.run_id`; строковые значения
  флагов не коэрсятся в числа (`--run-id 007` остаётся `"007"`).
- Опечатка в имени секции конфига (`trainig:` в YAML или `--set trainig.lr=...`)
  молча игнорировалась — теперь ошибка со списком известных секций.
- Опечатки в `model.params` для `bert_scoring` молча глотались `**_ignored`
  (например, `freeze_encodr: true` оставлял энкодер размороженным) — теперь все
  архитектуры валидируют params против своего конфига (`registry.build_model`).
- `experiment.output_root` валидируется до многоминутной загрузки данных, а не после.
- Коэрсия `--set`-оверрайдов сужена до научной записи (`1e-5` → float); прочие
  строки сохраняют тип из YAML.

### Changed
- Дедупликация моделей: ACT-обёртка и two-state рекурсия вынесены в
  `RecursiveACTReranker`/`RecursiveInnerBase` (`models/trm.py`), `BertTRMReranker`
  наследует их; `TiedReranker` наследует `VanillaReranker` (отличие — только
  `num_repeats`). Ключи state_dict не изменились.
- `run_model_once` переехал в `models/inference.py` (без хака
  `SimpleNamespace`-config в `BertScoringReranker`); `unwrap_model`/`model_device`/
  `get_autocast_context`/`load_pickle` — в `utils.py`. Циклический импорт
  trainer↔eval устранён (ленивые импорты убраны).
- Все особенности `bert_scoring` (kwargs-конструктор, отсутствие model dims)
  инкапсулированы в `models/registry.py` (`needs_model_dims`).
- `Trainer` принимает `RunDataBundle` вместо десяти отдельных полей; датасет и
  collate строятся внутри.
- CSV-лог обучения — класс `TrainLogWriter` вместо приклеивания `_handle` к
  `csv.DictWriter`; alias-хелперы манифестов свёрнуты в `_first_present`.
- `print` заменён на `logging` (модульные логгеры, `setup_logging()` в скриптах);
  скрипты ловят ожидаемые ошибки (конфиг/файлы) и выходят с кодом 2 без трейсбека.

### Added
- `--print-config` (train/evaluate): показать итог merge+overrides и выйти.
- Паттерн `configs/local.yaml` (в `.gitignore`) + шаблон `configs/local.example.yaml`
  для локальных путей; `configs/base.yaml` расширен всеми ключами `TrainingSection`
  с комментариями.
- `docs/running.md` — все режимы запуска (smoke, DDP, resume, сессионные окна,
  eval, значимость, footprint); `docs/experiments.md` формализован (зачем/команда/
  выход/DoD на каждый эксперимент).
- Тесты: неизвестные секции/ключи/`model.params`, коэрсия оверрайдов, merge
  `local.example.yaml`, стабильность state_dict-ключей TRM.

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
    `tied.py` (weight-tied deep, ALBERT-вариант для E3), `bert.py` (BERT-энкодер +
    {TRM | linear} голова, флаг `freeze_encoder`; полный файнтюн для E1),
    `registry.py` (фабрика по имени варианта).
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
- `configs/`: `base.yaml` + конфиги вариантов `variants/{trm,vanilla_shallow,vanilla_deep,tied_deep,bert_frozen_trm,bert_frozen_linear,bert_finetune}.yaml`.
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
- `main_scripts/`: `pure_trm_reranker` (TRM-вариант), `vanilla_transformer_rerank`
  (shallow/deep-варианты), `bert_encoder` (frozen BERT + TRM-голова), `bert_encoder_only_ablation`
  (frozen BERT + linear-голова), `bert_embeds` (TRM + frozen BERT-эмбеддинги),
  `halt_exper` (TRM + замеры latency), `monot5_base_full_eval` (baseline monoT5).
- `latency_notebooks/`: замеры latency архитектур проекта и SOTA-реранкеров.
- Планы: `trm_reranker_wsdm2027_plan.md` (E0–E9), `trm_reranker_plan_addendum.md`
  (E10–E13), обзор литературы `trm_literature_review_ru.md`.
