# Эксперименты: маппинг плана E0–E13 на код

План: `trm_reranker_wsdm2027_plan.md` (E0–E9) + `trm_reranker_plan_addendum.md`
(E10–E13). Статус — `project_tracking/2026-07-08_plan_status.md`. Здесь — чем
каждый эксперимент запускается. Все команды предполагают
`--set data.run_data_manifest_path=... --set experiment.output_root=...`
(или правку `configs/base.yaml`).

## E0 — аудит выравнивания рук

Аудит по коду: `project_tracking/e0_alignment_audit_2026-07-08.md`.
Замер FLOPs/params/latency всех рук единым протоколом:

```bash
uv run python scripts/measure_footprint.py \
    --arms configs/arms/trm.yaml configs/arms/vanilla_shallow.yaml \
           configs/arms/vanilla_deep.yaml configs/arms/tied_deep.yaml \
    --batch-size 1 --seq-len 256 --out footprint.json
```

Абляция one-step gradient (расхождение, найденное аудитом):
`--set model.params.full_backprop=true` к TRM-конфигу.

## E1 — починка BERT cross-encoder baseline

```bash
uv run python scripts/train.py --config configs/base.yaml configs/arms/bert_finetune.yaml
```

Конфиг уже содержит чек-лист: `freeze_encoder: false`, LR 2e-5, уменьшенный batch.
Перед доверием результату — санити-чек eval-пайплайна готовой моделью
(прогнать через `scripts/evaluate.py` известный реранкер и сверить с публикуемой цифрой).
DoD: MRR@10 ≥ 0.33, иначе строка удаляется и заменяется цитируемыми референсами.

## E2 — эквивалентность: сиды, значимость, память

```bash
for seed in 13 17 42; do
  uv run python scripts/train.py --config configs/base.yaml configs/arms/trm.yaml \
      --seed $seed --run-id seed$seed
  uv run python scripts/train.py --config configs/base.yaml configs/arms/vanilla_deep.yaml \
      --seed $seed --run-id seed$seed
done

# per-query метрики каждого прогона
uv run python scripts/evaluate.py --config configs/base.yaml configs/arms/trm.yaml \
    --checkpoint .../best_mrr.pt --split final --out-dir eval_out --tag trm_seed13
# значимость
uv run python scripts/significance.py --run-a eval_out/trm_seed13_final_per_query.csv \
    --run-b eval_out/deep_seed13_final_per_query.csv --out sig.json
```

Колонки таблицы params/checkpoint/peak-mem/GFLOPs/latency — из `measure_footprint.py`.
Внимание: перед запуском закрыть вопрос batch size из аудита E0 (512 vs 256).

## E3 — weight-tied deep (ALBERT-арм)

```bash
uv run python scripts/train.py --config configs/base.yaml configs/arms/tied_deep.yaml
```

## E4 — кривая глубины рекурсии

```bash
for L in 2 4 8; do for H in 1 2; do
  uv run python scripts/train.py --config configs/base.yaml configs/arms/trm.yaml \
      --set model.params.H_cycles=$H --set model.params.L_cycles=$L \
      --run-id h${H}l${L}
done; done
```

Ось X (GFLOPs) для каждой точки — `measure_footprint.py` с теми же `--set`.

## E5 — абляция ре-инжекции входа

```bash
uv run python scripts/train.py --config configs/base.yaml configs/arms/trm.yaml \
    --set model.params.disable_input_injection=true --run-id no_inject
```

## E6/E7 — TREC DL 2019/2020, BEIR

Не реализовано (следующий приоритет Фазы 2). План: скрипт
`scripts/eval_external.py` на `ir_datasets` + `ir_measures`
(зависимости уже в extra `eval`), инференс через
`evaluation.reranker_eval.score_pid_batch`, кандидаты — top-100/1000 BM25.

## E8 — hard negatives + multi-negative loss

Не реализовано; потребует нового датасета (msmarco-hard-negatives) и
InfoNCE/gBCE-лосса в `training/optim.py`. Go/no-go по плану — 20 июля.

## E10–E13 (статья 2 — адаптивная рекурсия)

ACT-механика уже в модели (`halt_max_steps > 1` + `halt_exploration_prob`);
не реализованы: промежуточный скоринг по шагам, калибровка порога, Парето (E11),
DeeBERT/GRU-конкуренты (E12b–c), error analysis (E13). E12a (linear probe) —
`configs/arms/bert_frozen_linear.yaml`; LoRA-вариант не реализован (peft).

## Правила честности (для любых новых прогонов)

1. Один рецепт для всех рук сравнения: менять `configs/base.yaml`, а не конфиг руки.
2. Каждая строка таблицы — реально измеренные latency (median+p95) и GFLOPs
   через `measure_footprint.py`; не копировать между строками.
3. Строка BM25/candidate-order — во всех таблицах (eval пишет её автоматически).
4. Параметры репортить раздельно: total / body (non-embedding) / trainable +
   checkpoint fp16 + peak memory.
5. ≥3 сида для заголовочных строк; per-query CSV сохранять для значимости.
