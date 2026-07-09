# Архитектуры

Все руки сравнения собраны из **одних и тех же блоков**
([src/trm_reranker/models/blocks.py](../src/trm_reranker/models/blocks.py)):
post-norm RMSNorm (eps 1e-5), SwiGLU (expansion 4.0, скрытая ширина кратна 256),
полное MHA без bias, RoPE (theta 10000), инициализация trunc-normal через
`CastedLinear`/`CastedEmbedding`, без dropout. Это конструктивная гарантия
требования К1 плана: руки различаются только схемой применения блоков.
Замечание для текста статьи: «vanilla» здесь — Llama-подобный блок, а не
классический post-LN/GELU трансформер.

Общий вход всех рук: `[CLS] запрос [SEP] пассаж [SEP]` (seq_len 256, запрос ≤32,
пассаж ≤221), сегментные эмбеддинги TRM-разметки (0/1/2), скоринг — линейная
голова над состоянием `[CLS]`. Кодирование — `data/encoding.py::PairEncoder`.

## Руки

### `trm` — TRM/HRM-derived two-state recurrent cross-encoder

[models/trm.py](../src/trm_reranker/models/trm.py). Одна сеть `L_level`
(`L_layers` блоков), применяемая рекуррентно с двумя латентными состояниями:

```
повторить H_cycles раз:
    повторить L_cycles раз:
        z_L <- L_level(z_L, injection = z_H + input_embeddings)
    z_H <- L_level(z_H, injection = z_L)
score = score_head(z_H[:, 0])
```

Эффективная глубина = `H_cycles * (L_cycles + 1) * L_layers` применений блока
(конфиг статьи H=2, L=4, L_layers=2 → 20). Особенности:

- **One-step gradient (HRM)**: первые `H_cycles - 1` циклов идут под `no_grad`;
  флаг `full_backprop: true` включает полный backprop (абляция к аудиту E0).
- **Ре-инжекция входа**: `disable_input_injection: true` убирает слагаемое
  `input_embeddings` из обновления z_L (эксперимент E5).
- **ACT/halting**: механика реализована (`q_head`, `halt_max_steps`,
  `halt_exploration_prob`, `no_ACT_continue`), но в конфигах статьи 1 выключена
  (`halt_max_steps: 1`). Это заготовка E10 (статья 2).
- `mlp_t: true` заменяет attention на токен-миксующий SwiGLU (вариант из TRM).

### `vanilla` — нерекурсивный стек

[models/vanilla.py](../src/trm_reranker/models/vanilla.py). Обычный стек из
`num_layers` независимых блоков. Две руки статьи:
`vanilla_shallow` (2 слоя, param-matched к TRM) и `vanilla_deep`
(20 слоёв, FLOP/depth-matched к TRM).

### `tied` — weight-tied deep (ALBERT-вариант, E3)

[models/tied.py](../src/trm_reranker/models/tied.py). `num_layers` уникальных
блоков, применяемых `num_repeats` раз подряд — cross-layer tying без
TRM-машинерии (нет z_H/z_L, нет ре-инжекции). Вместе с E5 декомпозирует,
что в TRM даёт вклад сверх простого шаринга весов.

### `bert_trm` и `bert_scoring` — BERT-руки

[models/bert.py](../src/trm_reranker/models/bert.py).

- `bert_trm`: предобученный энкодер выдаёт входные эмбеддинги для TRM-цикла
  (проекция при несовпадении ширины + LayerNorm); `freeze_encoder` управляет
  обучением энкодера.
- `bert_scoring`: энкодер + LayerNorm + dropout + линейная голова.
  `freeze_encoder: true` → linear-probe конкурент (E12a);
  `freeze_encoder: false` → полный файнтюн BERT cross-encoder (E1, LR 2e-5).

BERT-руки дополнительно получают `bert_token_type_ids` (нативная 0/1-разметка
сегментов BERT) — `PairEncoder(emit_bert_token_type_ids=True)`, автоматически
включается реестром (`models/registry.py::needs_bert_token_type_ids`).

## Проверенное соответствие FLOPs (профайлер, batch=1, seq=256, CPU, 2026-07-08)

| Рука | body params | GFLOPs/пара |
|---|---|---|
| trm (H2/L4/2 слоя) | 6.82M | 34.93 |
| vanilla_shallow (2) | 6.82M | 3.49 |
| vanilla_deep (20) | 68.16M | 34.93 |
| tied_deep (2×10) | 6.82M | 34.93 |

Подтверждает: deep-вариант на 20 слоях FLOP-matched к TRM (H=2, L=4), tied-вариант
сочетает параметры TRM с компьютом deep. Воспроизвести:
`scripts/measure_footprint.py`.

## Рецепт обучения (одинаков для всех рук — К2)

Pairwise logistic loss `-logsigmoid(s_pos - s_neg)` на официальных триплетах
MS MARCO; AdamW (wd 0.01), linear warmup 6% + linear decay, grad clip 1.0,
bf16-mixed. Известные расхождения легаси-прогонов (batch 512 vs 256, единый LR
без per-variant свипа) — см. `project_tracking/e0_alignment_audit_2026-07-08.md`.
