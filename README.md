# Adaptation of Recursive Reasoning Models for Reranking

Исследование параметр/память-эффективности рекурсивных (TRM/HRM-derived) кросс-энкодеров
для переранжирования пассажей MS MARCO. Центральный клейм: weight-shared рекурсивный
кросс-энкодер достигает качества FLOP-matched глубокого трансформера при кратно меньшем
числе обучаемых параметров. Планы экспериментов: `trm_reranker_wsdm2027_plan.md` (E0–E9),
`trm_reranker_plan_addendum.md` (E10–E13); текущий статус — `project_tracking/`.

## Структура репозитория

```text
configs/                  # YAML-конфиги: base.yaml + configs/arms/<арм>.yaml
src/trm_reranker/
  models/                 # blocks.py (общие блоки всех рук — гарантия выравнивания К1),
                          # trm.py, vanilla.py, tied.py (E3), bert.py (E1/E12a), registry.py
  data/                   # манифесты, шардированный passage store, датасеты, encode/collate
  training/               # Trainer (DDP, bf16, resume, чекпоинты), distributed, optim
  evaluation/             # метрики, реранкер-eval (+ per-query дамп), significance (E2)
  benchmarks/             # latency (mean/p50/p95), FLOPs (профайлер), params/память
  config.py, runtime.py   # конфиги и сборка рана
scripts/                  # train.py, evaluate.py, measure_footprint.py, significance.py
tests/                    # smoke-тесты + parity-тест против легаси-чекпоинтов
notebooks/legacy/         # архив исходных ноутбуков (включая data_prep — см. docs/data.md)
docs/                     # architecture.md, experiments.md, data.md
project_tracking/         # рабочие документы проекта: статусы плана, аудиты (не документация кода)
```

## Установка (uv)

```bash
# базовое окружение (torch, transformers, ...)
uv sync

# + значимость/внешние бенчмарки (scipy, ir-measures, ir-datasets)
uv sync --extra eval

# + fvcore для кросс-проверки FLOPs, + dev-инструменты
uv sync --extra bench --dev
```

Python зафиксирован в `.python-version` (arm64-сборка: на Apple Silicon с x86_64-Anaconda
uv иначе резолвит окружение под Rosetta, где нет колёс torch).

## Данные

Пайплайн ожидает файлы MS MARCO passage ranking (`collection.tsv`,
`qidpidtriples.train.full.2.tsv`, `top1000.dev`, `qrels.dev.small.tsv`,
`queries.train.tsv`). Подготовка кэшей пока выполняется легаси-ноутбуками
`notebooks/legacy/data_prep/00_*.ipynb` и `01_*.ipynb` — подробности и формат
манифестов в [docs/data.md](docs/data.md). Результат подготовки — `prep_manifest.json`
(датасет-уровень) и `run_data_manifest.json` (run-уровень); тренировке нужен только второй.

## Запуск

Обучение одной руки:

```bash
uv run python scripts/train.py \
    --config configs/base.yaml configs/arms/trm.yaml \
    --set data.run_data_manifest_path=/path/to/run_data_manifest.json \
    --set experiment.output_root=/path/to/output
```

Смоук-проверка (20 шагов): добавьте `--set training.max_train_steps=20`.
DDP: `uv run torchrun --nproc_per_node=2 scripts/train.py ... --set training.use_ddp=true`.
Мультисид (E2): `--seed 13|17|42 --run-id seed13|...` — сид входит в имя run-директории.
Возобновление: `--resume-from /path/to/last_checkpoint.pt`.

Любой параметр перекрывается через `--set section.key=value`; полный снапшот
гиперпараметров пишется в `runs/<experiment>/training_config.json`.

Оценка чекпоинта (пишет и per-query CSV для тестов значимости):

```bash
uv run python scripts/evaluate.py \
    --config configs/base.yaml configs/arms/trm.yaml \
    --set data.run_data_manifest_path=... \
    --checkpoint /path/to/best_mrr.pt --split final --out-dir eval_out/
```

Значимость (E2, paired bootstrap + t-test):

```bash
uv run python scripts/significance.py \
    --run-a eval_out/trm_final_per_query.csv \
    --run-b eval_out/vanilla_deep_final_per_query.csv \
    --out significance.json
```

Единая таблица params / GFLOPs / latency / память по всем рукам (E0/E2; каждая строка
меряется реально, протокол: один девайс, фиксированный batch/seq, warmup, mean/p50/p95):

```bash
uv run python scripts/measure_footprint.py \
    --arms configs/arms/trm.yaml configs/arms/vanilla_shallow.yaml \
           configs/arms/vanilla_deep.yaml configs/arms/tied_deep.yaml \
    --batch-size 1 --seq-len 256 --out footprint.json
```

## Выходные артефакты рана

```text
<output_root>/runs/<experiment>_seed<seed>_<run_id>/
  training_config.json        # полный конфиг (для таблицы гиперпараметров статьи)
  run_artifacts.json          # пути к манифестам/данным
  checkpoints/                # last / best_train_loss / best_mrr / best_dev_mrr10 / step_*
  logs/train_metrics.csv      # пошаговый лог (loss, acc, margin, lr, grad_norm)
  logs/dev_metrics_by_{step,epoch}.csv
  eval/{steps,epochs,final}/  # TREC run, метрики JSON, *_per_query.csv
```

Метрики: MRR@10, hit@{1,3,5,10}, nDCG@10; в каждом eval также строка
BM25/candidate-order baseline (вход реранкера).

## Тесты

```bash
uv run pytest                 # smoke: модели, кодирование пар, метрики, bootstrap, конфиги

# parity против чекпоинта, обученного легаси-ноутбуком (обязателен перед заменой
# старых чисел новыми прогонами):
TRM_LEGACY_CHECKPOINT=/path/to/best_mrr.pt uv run pytest tests/test_parity.py -v
```

## Документация

- [docs/architecture.md](docs/architecture.md) — устройство моделей и рук сравнения
- [docs/experiments.md](docs/experiments.md) — маппинг экспериментов E0–E13 на конфиги/команды
- [docs/data.md](docs/data.md) — подготовка данных и форматы манифестов
- [CHANGELOG.md](CHANGELOG.md) — версии
- [project_tracking/](project_tracking/) — статус плана и аудит E0 (рабочие документы)
