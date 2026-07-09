# Adaptation of Recursive Reasoning Models for Reranking

Исследование параметр/память-эффективности рекурсивных (TRM/HRM-derived) кросс-энкодеров
для переранжирования пассажей MS MARCO. Центральный клейм: weight-shared рекурсивный
кросс-энкодер достигает качества FLOP-matched глубокого трансформера при кратно меньшем
числе обучаемых параметров. Планы экспериментов: `trm_reranker_wsdm2027_plan.md` (E0–E9),
`trm_reranker_plan_addendum.md` (E10–E13); текущий статус — `project_tracking/`.

## Структура репозитория

```text
configs/                  # YAML-конфиги: base.yaml + configs/variants/<вариант>.yaml
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

Один раз пропишите локальные пути (файл в `.gitignore`, подробности —
[docs/running.md](docs/running.md)):

```bash
cp configs/local.example.yaml configs/local.yaml   # указать output_root и путь к манифесту
```

Обучение одной руки:

```bash
uv run python scripts/train.py \
    --config configs/base.yaml configs/variants/trm.yaml configs/local.yaml
```

Смоук-проверка (20 шагов): добавьте `--set training.max_train_steps=20`.
Посмотреть итоговый конфиг без запуска: `--print-config`.
DDP: `uv run torchrun --nproc_per_node=2 scripts/train.py ... --set training.use_ddp=true`.
Мультисид (E2): `--seed 13|17|42 --run-id seed13|...` — сид входит в имя run-директории.
Периодичность чекпоинтов: `training.checkpoint_every_n_steps` либо
`training.checkpoint_epoch_fraction` (дефолт 0.005 эпохи).
Возобновление: `--resume-from /path/to/last_checkpoint.pt`.

Любой параметр перекрывается через `--set section.key=value`; полный снапшот
гиперпараметров пишется в `runs/<experiment>/training_config.json`. Опечатки в
секциях/ключах конфига и в `model.params` отклоняются на старте. Все режимы
запуска (сессионные окна, resume, оценка, значимость, footprint) — в
[docs/running.md](docs/running.md).

Лосс выбирается конфигом: `training.loss = pairwise_logistic | infonce` (E8);
для InfoNCE негативы добываются `scripts/mine_hard_negatives.py` из ранжирований
любого ретривера и подключаются через `data.hard_negatives_path`
([docs/running.md, раздел 12](docs/running.md)).

Оценка чекпоинта (пишет и per-query CSV для тестов значимости):

```bash
uv run python scripts/evaluate.py \
    --config configs/base.yaml configs/variants/trm.yaml \
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
    --variants configs/variants/trm.yaml configs/variants/vanilla_shallow.yaml \
           configs/variants/vanilla_deep.yaml configs/variants/tied_deep.yaml \
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

## Тесты и качество кода

```bash
make test      # pytest: модели, конфиги, e2e-трейнер (чекпоинты+resume), майнинг
make check     # ruff check + ruff format --check + pytest (то же гоняет CI)
make fmt       # автоформат и автофиксы
make hooks     # pre-commit хуки (ruff при каждом коммите)

# parity против чекпоинта, обученного легаси-ноутбуком (обязателен перед заменой
# старых чисел новыми прогонами):
TRM_LEGACY_CHECKPOINT=/path/to/best_mrr.pt make parity
```

CI (GitHub Actions) гоняет `make check` на каждый push/PR — см. `.github/workflows/ci.yml`.

## Документация

- [docs/architecture.md](docs/architecture.md) — устройство моделей и рук сравнения
- [docs/running.md](docs/running.md) — все режимы запуска: smoke, DDP, resume, сессионные окна, eval
- [docs/experiments.md](docs/experiments.md) — эксперименты E0–E13: зачем, команда, DoD
- [docs/data.md](docs/data.md) — подготовка данных и форматы манифестов
- [CHANGELOG.md](CHANGELOG.md) — версии
- [CONTRIBUTING.md](CONTRIBUTING.md) — правила для контрибьюторов (паритет с легаси, стиль, конфиги)
- [project_tracking/](project_tracking/) — статус плана и аудит E0 (рабочие документы)
