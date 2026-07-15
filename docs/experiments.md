# Эксперименты: что запускаем и зачем (E0–E13)

Планы: `artifacts/trm_reranker_wsdm2027_plan.md` (E0–E9) +
`artifacts/trm_reranker_plan_addendum.md` (E10–E13); статус —
`project_tracking/2026-07-08_plan_status.md`. Команды запуска по режимам —
[docs/running.md](running.md); все команды ниже предполагают настроенный
`configs/local.yaml`.

Каждый эксперимент описан единообразно: **Зачем** (какой вопрос закрывает) /
**Команда** / **Выход** (что идёт в статью) / **DoD** (когда считаем сделанным).

## E0 — аудит выравнивания рук

**Зачем:** гарантировать, что руки сравнения отличаются только схемой применения
слоёв (К1/К2): одинаковые блоки, рецепт, данные — иначе сравнение невалидно.

**Команда:** аудит по коду — `project_tracking/e0_alignment_audit_2026-07-08.md`;
замер единым протоколом:

```bash
uv run python scripts/measure_footprint.py \
    --variants configs/variants/trm.yaml configs/variants/vanilla_shallow.yaml \
           configs/variants/vanilla_deep.yaml configs/variants/tied_deep.yaml \
    --batch-size 1 --seq-len 256 --out footprint.json
```

Абляция one-step gradient (расхождение, найденное аудитом):
`--set model.params.full_backprop=true` к TRM-конфигу.

**Выход:** таблица params/GFLOPs/latency/память; подтверждение выравнивания.
**DoD:** закрыт вопрос batch size (512 vs 256) до запусков E2.

## E1 — починка BERT cross-encoder baseline

**Зачем:** в легаси BERT-baseline давал MRR@10 ≈ 0.15 вместо публикуемых ≈ 0.34–0.36 —
без честного baseline статья не проходит.

**Команда:**

```bash
uv run python scripts/train.py \
    --config configs/base.yaml configs/variants/bert_finetune.yaml configs/local.yaml
```

Конфиг уже содержит чек-лист: `freeze_encoder: false`, LR 2e-5 (не 2e-4!), уменьшенный batch.

**Выход:** строка BERT-base в главной таблице.
**DoD:** MRR@10 ≥ 0.33 после санити-чека eval-пайплайна готовой моделью
(прогнать через `scripts/evaluate.py` известный реранкер и сверить с публикуемой
цифрой); иначе строка заменяется цитируемыми референсами.

## E2 — эквивалентность: сиды, значимость, память

**Зачем:** главный клейм статьи — «TRM ≈ FLOP-matched deep при кратно меньших
параметрах» — должен держаться на ≥3 сидах со значимостью, а не на одном прогоне.

**Команда:**

```bash
for seed in 13 17 42; do
  uv run python scripts/train.py --config configs/base.yaml configs/variants/trm.yaml \
      configs/local.yaml --seed $seed --run-id seed$seed
  uv run python scripts/train.py --config configs/base.yaml configs/variants/vanilla_deep.yaml \
      configs/local.yaml --seed $seed --run-id seed$seed
done

# per-query метрики каждого прогона
uv run python scripts/evaluate.py --config configs/base.yaml configs/variants/trm.yaml \
    configs/local.yaml --checkpoint .../best_mrr.pt --split final \
    --out-dir eval_out --tag trm_seed13
# значимость
uv run python scripts/significance.py --run-a eval_out/trm_seed13_final_per_query.csv \
    --run-b eval_out/deep_seed13_final_per_query.csv --out sig.json
```

**Выход:** главная таблица (mean±std по сидам, p-values, params/checkpoint/peak-mem/
GFLOPs/latency из `measure_footprint.py`).
**DoD:** для каждой заголовочной строки ≥3 сида + результат paired-теста против TRM.

## E3 — weight-tied deep (ALBERT-вариант)

**Зачем:** развязать вклад weight sharing как такового от TRM-механики (два
латентных состояния + ре-инжекция входа). Если tied ≈ TRM — рекурсивная машинерия
не при чём; если TRM > tied — она даёт вклад.

```bash
uv run python scripts/train.py \
    --config configs/base.yaml configs/variants/tied_deep.yaml configs/local.yaml
```

**Выход:** строка tied_deep в таблице (та же эффективная глубина 20, те же unique params).
**DoD:** прогон на тех же сидах, что E2.

## E4 — кривая глубины рекурсии

**Зачем:** качество vs compute при фиксированных параметрах — уникальное свойство
рекурсивной руки; кривая (GFLOPs → MRR@10) — второй ключевой график статьи.

```bash
for L in 2 4 8; do for H in 1 2; do
  uv run python scripts/train.py --config configs/base.yaml configs/variants/trm.yaml \
      configs/local.yaml \
      --set model.params.H_cycles=$H --set model.params.L_cycles=$L \
      --run-id h${H}l${L}
done; done
```

Ось X (GFLOPs) для каждой точки — `measure_footprint.py` с теми же `--set`.

**Выход:** график глубина/качество.
**DoD:** ≥5 точек кривой, замеренные GFLOPs на каждую.

## E5 — абляция ре-инжекции входа

**Зачем:** ре-инжекция входных эмбеддингов в каждый L-цикл — центральное
отличие TRM от tied; измеряем её вклад изолированно.

```bash
uv run python scripts/train.py --config configs/base.yaml configs/variants/trm.yaml \
    configs/local.yaml --set model.params.disable_input_injection=true --run-id no_inject
```

**Выход:** строка абляции в таблице ablations.
**DoD:** сравнение с TRM на том же сиде (+значимость, если различие мало).

## E6/E7 — TREC DL 2019/2020, BEIR

**Зачем:** обобщение за пределы MS MARCO dev (стандартное требование ревьюеров).
**Статус: не реализовано** (следующий приоритет Фазы 2). План: скрипт
`scripts/eval_external.py` на `ir_datasets` + `ir_measures` (зависимости уже в
extra `eval`), инференс через `evaluation.reranker_eval.score_pid_batch`,
кандидаты — top-100/1000 BM25.

## E8 — hard negatives + multi-negative loss

**Зачем:** проверить, что вывод «TRM ≈ deep» не артефакт лёгких BM25-негативов.
Инфраструктура готова: `scripts/mine_hard_negatives.py` + `training.loss=infonce`
(подробности и предостережения — docs/running.md, раздел 12). Осталось достать
ранжирования train-запросов (BM25 top-1000 или msmarco-hard-negatives).

**Команда:**

```bash
uv run python scripts/mine_hard_negatives.py \
    --candidates /path/to/train_bm25_top1000.run --qrels /path/to/qrels.train.tsv \
    --rank-min 10 --rank-max 200 --num-negatives 30 --out hard_negatives.jsonl

# K2: лосс меняется в base.yaml для ВСЕХ вариантов сравнения — E8 требует
# переучить и TRM, и baselines с одинаковым лоссом/негативами.
uv run python scripts/train.py --config configs/base.yaml configs/variants/trm.yaml \
    configs/local.yaml --set training.loss=infonce \
    --set data.hard_negatives_path=hard_negatives.jsonl --run-id infonce_hard
```

**Выход:** таблица «official triples + pairwise» vs «hard negatives + InfoNCE»
для TRM и vanilla_deep (минимум).
**DoD:** вывод «TRM ≈ deep» воспроизводится (или опровергается) на трудных
негативах; go/no-go по плану — 20 июля 2026.

## E10–E13 (статья 2 — адаптивная рекурсия)

**Зачем:** адаптивная глубина (ACT): тратить больше рекурсии на трудные запросы.
ACT-механика уже в модели (`halt_max_steps > 1` + `halt_exploration_prob`);
E12a (linear probe) — `configs/variants/bert_frozen_linear.yaml`.
**Статус: не реализовано:** промежуточный скоринг по шагам, калибровка порога,
Парето (E11), DeeBERT/GRU-конкуренты (E12b–c, LoRA через peft), error analysis (E13).

## Правила честности (для любых новых прогонов)

1. Один рецепт для всех рук сравнения: менять `configs/base.yaml`, а не конфиг руки.
2. Каждая строка таблицы — реально измеренные latency (median+p95) и GFLOPs через
   `measure_footprint.py`; не копировать между строками.
3. Строка BM25/candidate-order — во всех таблицах (eval пишет её автоматически).
4. Параметры репортить раздельно: total / body (non-embedding) / trainable +
   checkpoint fp16 + peak memory.
5. ≥3 сида для заголовочных строк; per-query CSV сохранять для значимости.
6. Перед заменой легаси-чисел новыми — parity-прогон (`tests/test_parity.py`).
