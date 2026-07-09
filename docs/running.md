# Запуск: режимы и рецепты

Все команды выполняются из корня репозитория. Конфиг собирается слева направо
(`--config base.yaml variant.yaml local.yaml`, последний побеждает), поверх
накладываются `--set section.key=value`. Итоговый снапшот гиперпараметров
каждого рана сохраняется в `runs/<experiment>/training_config.json`.

## 0. Разовая настройка машины: configs/local.yaml

Локальные пути не хранятся в git. Один раз скопируйте шаблон и пропишите свои пути:

```bash
cp configs/local.example.yaml configs/local.yaml
# отредактируйте: experiment.output_root и data.run_data_manifest_path
```

Дальше `configs/local.yaml` передаётся последним в каждый `--config`, и
`--set data.run_data_manifest_path=...` в командах больше не нужен.

## 1. Посмотреть итоговый конфиг (dry-run)

`--print-config` печатает результат merge + оверрайдов и выходит, ничего не запуская:

```bash
uv run python scripts/train.py \
    --config configs/base.yaml configs/variants/trm.yaml configs/local.yaml \
    --set model.params.L_cycles=8 --print-config
```

Проверяйте этим флагом любую нетривиальную комбинацию `--config`/`--set` перед запуском.

## 2. Смоук-прогон (проверка пайплайна за минуты)

```bash
uv run python scripts/train.py \
    --config configs/base.yaml configs/variants/trm.yaml configs/local.yaml \
    --set training.max_train_steps=20
```

`max_train_steps` жёстко останавливает обучение после N оптимизационных шагов.

## 3. Полное обучение одной руки

```bash
uv run python scripts/train.py \
    --config configs/base.yaml configs/variants/trm.yaml configs/local.yaml
```

Имя run-директории: `runs/<experiment.name>_seed<seed>_<run_id>/`;
`run_id` берётся из `--run-id`, иначе из `TRM_RUN_ID`/`TORCHELASTIC_RUN_ID`, иначе timestamp.

## 4. Чекпоинты: как часто и какие

Частота задаётся в `training` (одно из двух, шаги приоритетнее):

- `checkpoint_epoch_fraction: 0.005` — каждые 0.5% эпохи (дефолт);
- `checkpoint_every_n_steps: 2000` — явный шаг в оптимизационных шагах.

С той же периодичностью выполняется быстрый dev-eval на `dev_eval_query_limit`
запросах. В `checkpoints/` пишутся:

| Файл | Что это |
|---|---|
| `last_checkpoint.pt` | последний периодический — точка возобновления |
| `step_XXXXXXXX.pt` | периодические снапшоты |
| `epoch_NNN.pt` | на границе эпохи |
| `best_train_loss.pt` | лучший train loss |
| `best_mrr.pt` | лучший MRR@10 на step-eval'ах |
| `best_dev_mrr10.pt` | лучший MRR@10 на epoch-eval'ах |

Чтобы `step_*.pt` не съедали диск на длинных прогонах:
`training.keep_last_step_checkpoints: 3` — хранить только 3 последних
периодических; именованные (`last`/`best_*`/`epoch_*`) не удаляются никогда.

## 5. Возобновление обучения с чекпоинта

```bash
uv run python scripts/train.py \
    --config configs/base.yaml configs/variants/trm.yaml configs/local.yaml \
    --resume-from /path/to/runs/<experiment>/checkpoints/last_checkpoint.pt
```

Восстанавливаются веса, optimizer, scheduler, scaler, `global_step` и best-метрики;
CSV-лог дописывается, а не перезаписывается. Ограничение: при mid-epoch resume нельзя
менять размер train-выборки, batch size, world size и `grad_accum_steps` (проверяется,
скрипт упадёт с понятной ошибкой). `training.allow_resume_lr_override=true` позволяет
взять LR из конфига вместо сохранённого.

## 6. Сессионные окна (обучение кусками)

Для сред с лимитом времени сессии (Colab и т.п.) — обучать по куску эпохи за запуск:

```bash
# по четверти эпохи за сессию; каждый следующий запуск продолжает с last_checkpoint
uv run python scripts/train.py ... --set training.run_epoch_fraction=0.25
uv run python scripts/train.py ... --set training.run_epoch_fraction=0.25 \
    --resume-from .../last_checkpoint.pt
```

`run_train_steps` — то же самое в шагах (приоритетнее fraction). Финальный full-dev
eval запустится только когда суммарный `global_step` достигнет плана
(`epochs * steps_per_epoch`).

## 7. DDP (несколько GPU)

```bash
uv run torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/base.yaml configs/variants/vanilla_deep.yaml configs/local.yaml \
    --set training.use_ddp=true --set training.devices=2
```

Без `torchrun` (`WORLD_SIZE=1`) флаг `use_ddp` игнорируется с предупреждением.
Глобальный batch = `per_device_batch_size * world_size * grad_accum_steps` — держите
его одинаковым между руками сравнения.

## 8. Мультисид (E2)

```bash
for seed in 13 17 42; do
  uv run python scripts/train.py \
      --config configs/base.yaml configs/variants/trm.yaml configs/local.yaml \
      --seed $seed --run-id seed$seed
done
```

Сид входит и в имя директории, и в `training_config.json`.

## 9. Оценка чекпоинта

```bash
uv run python scripts/evaluate.py \
    --config configs/base.yaml configs/variants/trm.yaml configs/local.yaml \
    --checkpoint /path/to/best_mrr.pt \
    --split final --out-dir eval_out --tag trm_seed13
```

`--split epoch` — быстрый dev-сабсет, `--split final` — полный dev. Пишет TREC-run,
`*_metrics.json` и `*_per_query.csv` (вход для тестов значимости). Конфиг руки должен
совпадать с тем, чем обучали чекпоинт.

## 10. Значимость (paired bootstrap + t-test)

```bash
uv run python scripts/significance.py \
    --run-a eval_out/trm_seed13_final_per_query.csv \
    --run-b eval_out/deep_seed13_final_per_query.csv \
    --out significance.json
```

## 11. Footprint-таблица (params / GFLOPs / latency / память)

```bash
uv run python scripts/measure_footprint.py \
    --variants configs/variants/trm.yaml configs/variants/vanilla_shallow.yaml \
           configs/variants/vanilla_deep.yaml configs/variants/tied_deep.yaml \
    --batch-size 1 --seq-len 256 --out footprint.json
```

Данные не нужны (синтетический batch); каждая строка меряется реально на одном девайсе.

## 12. Hard negatives и выбор лосса (E8)

Лосс — часть общего рецепта (правило К2): задаётся в `configs/base.yaml`
(`training.loss`), а не в конфиге варианта; при смене лосса переучиваются **все**
варианты сравнения, смешивать pairwise- и InfoNCE-строки в одной таблице нельзя.

**Шаг 1 — майнинг негативов.** Скрипт не запускает ретривер сам: ему нужны
готовые ранжирования train-запросов (официальный BM25 top-1000 MS MARCO,
свой TREC-run любого ретривера или сконвертированный msmarco-hard-negatives):

```bash
uv run python scripts/mine_hard_negatives.py \
    --candidates /path/to/train_bm25_top1000.run \
    --qrels /path/to/qrels.train.tsv \
    --rank-min 10 --rank-max 200 --num-negatives 30 \
    --out /path/to/hard_negatives.jsonl
```

Полоса `[rank-min, rank-max]` — «трудная зона»: высокие ранги, но не размеченные
позитивы (топ-1..9 пропускаем — там часто неразмеченные релевантные). Храните
негативов больше, чем `training.num_negatives`: датасет пересэмплирует их каждую
эпоху. Токены любых pid берутся из полного шард-стора — отдельная токенизация
не нужна.

**Шаг 2 — обучение с InfoNCE:**

```bash
uv run python scripts/train.py \
    --config configs/base.yaml configs/variants/trm.yaml configs/local.yaml \
    --set training.loss=infonce \
    --set data.hard_negatives_path=/path/to/hard_negatives.jsonl
```

`per_device_batch_size` при InfoNCE считается в **группах** (1 позитив +
`num_negatives` негативов), т.е. forward обрабатывает `batch * (1 + K)` пар —
уменьшайте batch соответственно. `pairwise_logistic` +
`data.hard_negatives_path` тоже работает: это InfoNCE с K=1 (математически
идентично pairwise-лоссу), негативы просто становятся труднее.

## 13. Тесты и parity

```bash
make test                        # uv run pytest: модели, конфиги, трейнер (e2e + resume), майнинг
make check                       # линт + формат + тесты (как в CI)
TRM_LEGACY_CHECKPOINT=/path/to/best_mrr.pt make parity
```

Parity-прогон обязателен перед заменой старых (ноутбучных) чисел новыми.
Хуки форматирования: `make hooks` (pre-commit на ruff check/format).

## Отладка

- Опечатка в имени секции или ключа конфига — ошибка при старте, а не молчаливое
  игнорирование (это касается и `--set`, и YAML-файлов, и `model.params`).
- Пути (`output_root`, манифест) валидируются до загрузки данных — падение мгновенное.
- Строковые значения флагов не превращаются в числа (`--run-id 007` останется `"007"`);
  в `--set` числа в научной записи (`1e-5`) корректно становятся float.
