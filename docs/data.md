# Данные: подготовка и форматы

## Исходные файлы (MS MARCO passage ranking)

- `collection.tsv` — пассажи (pid \t text)
- `queries.train.tsv`, `queries.dev.small.tsv`
- `qidpidtriples.train.full.2.tsv` — официальные обучающие триплеты
- `top1000.dev` — BM25-кандидаты dev (вход реранкера)
- `qrels.dev.small.tsv` — разметка dev (6 980 запросов)

## Двухуровневый кэш

Подготовка пока выполняется легаси-ноутбуками (портирование в
`scripts/prepare_data.py` — задача Фазы 2; ноутбуки линейные и рабочие):

1. **Датасет-уровень** — `notebooks/legacy/data_prep/00_prepare_data_cache.ipynb`.
   Токенизирует запросы и всю коллекцию (bert-base-uncased, seq 256 / query 32 /
   doc 221), пишет:
   - `prep_manifest.json` (schema_version ≥ 4) — точки входа ко всем артефактам;
   - токенизатор локально; pkl-карты токенов запросов;
   - **шардированный passage store**: npz-шарды `pid / offsets / token_ids`
     (формат `sharded_flat_token_arrays_v1`, shard_size 50k, роутинг pid -> шард
     по `pid // shard_size`) + индекс JSON;
   - `dev_candidates.pkl` / `dev_qrels.pkl` (кандидаты сгруппированы как
     `qid_order / qid_offsets / pid / bm25_rank`).

2. **Run-уровень** — `notebooks/legacy/data_prep/01_prepare_run_data_cache.ipynb`.
   Сэмплирует триплеты под конкретный бюджет (`train_triples_sample`, seed),
   вырезает подмножества токенов и dev-подвыборку для быстрых eval, пишет
   `run_data_manifest.json` (schema_version ≥ 1).

Тренировке (`scripts/train.py`) нужен только `run_data_manifest.json` —
путь к prep-манифесту он берёт из поля `base_prep_manifest_path`.

## Валидация совместимости

`src/trm_reranker/data/manifests.py` проверяет при старте: версию схемы обоих
манифестов, совпадение tokenizer/seq_len/max_query_len/max_doc_len между prep и
run, совпадение `train_triples_sample` и `seed` конфига с run-манифестом.
Несовпадение — ошибка с перечислением полей (пересобрать run-кэш).

## Чтение в обучении

- `PairwiseTripleDataset` — триплеты (qid, pos_pid, neg_pid) + pkl-карты токенов.
- `PairEncoder.encode_pair` — `[CLS] q [SEP] d [SEP]`, сегменты TRM 0/1/2,
  для BERT-рук дополнительно нативные 0/1 (`bert_token_type_ids`).
- Eval читает пассажи из шардированного стора через
  `build_passage_token_subset_loader` (LRU-кэш шардов).
