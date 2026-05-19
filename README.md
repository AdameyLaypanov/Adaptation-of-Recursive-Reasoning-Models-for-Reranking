# Neural Reranking Experiments

This repository contains Jupyter notebooks for preparing MS MARCO-style data, training neural reranker models, and comparing model latency against baseline/SOTA rerankers. The main workflow is: build tokenized caches, create a run-level subset for an experiment, train one of the architectures, and evaluate reranking metrics on `top1000.dev`.

## Repository Structure

```text
data_prep/
  00_prepare_data_cache.ipynb        # Dataset-level caches: tokenizer, query tokens, passage shards, dev candidates/qrels
  01_prepare_run_data_cache.ipynb    # Run-level caches: sampled triples, train token subsets, dev subset

main_scripts/
  pure_trm_reranker.ipynb            # TRM/Tiny Recursive Reasoning reranker
  halt_exper.ipynb                   # TRM reranker with forward/final-dev latency measurements
  vanilla_transformer_rerank.ipynb   # Vanilla Transformer ablation
  bert_encoder.ipynb                 # Frozen BERT encoder + TRM reasoning head
  bert_encoder_only_ablation.ipynb   # Frozen BERT encoder + scoring head without TRM
  bert_embeds.ipynb                  # TRM with frozen BERT word embeddings + projection
  monot5_base_full_eval.ipynb        # Offline full evaluation for monoT5-base baseline

latency_notebooks/
  all_architectures_latency.ipynb    # Synthetic batch=1 latency for project architectures
  halt_exper_latency.ipynb           # TRM/HAlT-style forward-loop latency
  sota_latency.ipynb                 # Latency for monoT5-base and bge-reranker-v2-m3
```

The repository does not include source datasets, local model checkpoints, or run outputs. The notebooks contain placeholder paths such as `<DATA_ROOT>`, `<ARTIFACT_DIR>`, `<OUTPUT_ROOT>`, and `<path/to/...>`; replace them with real local directories before running anything.

## Input Data

The pipeline expects MS MARCO passage ranking-style files:

- `collection.tsv`
- `qidpidtriples.train.full.2.tsv`
- `top1000.dev`
- `qrels.dev.small.tsv`
- `queries.train.tsv`
- `queries.dev.small.tsv`, optional: if this path is not configured, dev queries are derived from `top1000.dev`.

In the main data-preparation notebook, `top1000.dev` is read as `qid<TAB>pid...`. `monot5_base_full_eval.ipynb` can auto-detect multiple formats: text rows, TREC run format, or an ID-only candidate file that is later joined with `queries` and `collection`.

## Environment

The local VS Code configuration points to `.venv/bin/python`; the current `.venv` uses Python 3.13.3. A minimal environment for the notebooks is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyterlab ipykernel numpy pandas tqdm torch transformers
```

Multi-GPU training requires a CUDA-compatible PyTorch build. The notebooks use `bf16-mixed` when CUDA and bfloat16 are available; otherwise they fall back to `float32`.

## Data Preparation

1. Open `data_prep/00_prepare_data_cache.ipynb` and configure:
   - `DATA_ROOT`: directory with the dataset TSV files;
   - `ARTIFACT_DIR`: directory for dataset-level artifacts;
   - `DEV_QUERIES_PATH`, if a separate `queries.dev.small.tsv` file is available;
   - `TOKENIZER_NAME`, `SEQ_LEN`, `MAX_QUERY_LEN`, and `MAX_DOC_LEN`, if you need values different from `bert-base-uncased`, `256`, `32`, and `221`.

2. Run all cells. The notebook creates:
   - `prep_manifest.json`;
   - `cache_manifest_<tag>.json`;
   - `artifact_index.json`;
   - `train_queries.pkl`, `dev_queries.pkl`;
   - tokenized query maps;
   - a sharded passage token store in `.npz` files;
   - `dev_candidates.pkl`, `dev_qrels.pkl`.

3. Open `data_prep/01_prepare_run_data_cache.ipynb` and configure:
   - `DATA_ROOT`;
   - `PREP_MANIFEST_PATH`;
   - `OUTPUT_ROOT`;
   - `RUN_PROFILE`: `smoke`, `full`, or `full_all_dev`.

4. Run the notebook. It creates `OUTPUT_ROOT/trm_reranker_mvp/run_data_cache/<cache_name>/run_data_manifest.json` plus related run-level files: sampled triples, train query tokens, train passage tokens, and epoch dev candidates/qrels.

## Training

Choose an experiment in `main_scripts` and replace the configuration paths in the first notebook cell:

- `INPUT_ARTIFACT_DIR` or `INPUT_PREP_MANIFEST_PATH`;
- `INPUT_RUN_DATA_MANIFEST_PATH`;
- `INPUT_DATA_ROOT` and `INPUT_COLLECTION_PATH`;
- `DEFAULT_OUTPUT_ROOT`;
- for BERT variants: the path to a local `bert-base-uncased` checkpoint;
- for the monoT5 baseline: `CFG["project_root"]`, `CFG["data_dir"]`, `CFG["model_dir"]`, and `CFG["out_dir"]`.

Profiles:

- `smoke`: quick sanity check with 100k triples, one device, and a limited number of train steps.
- `full`: the main profile for the selected notebook.
- `full_all_dev`: profile with full dev evaluation.

Typical interactive launch:

```bash
source .venv/bin/activate
jupyter lab
```

For reproducible CLI notebook execution:

```bash
jupyter nbconvert --execute --to notebook --inplace data_prep/00_prepare_data_cache.ipynb
jupyter nbconvert --execute --to notebook --inplace data_prep/01_prepare_run_data_cache.ipynb
jupyter nbconvert --execute --to notebook --inplace main_scripts/pure_trm_reranker.ipynb
```

Some training notebooks support `--profile`, `--output-root`, `--resume-from-checkpoint`, and `--run-id` when executed as Python scripts. For DDP/multi-GPU runs, export the notebook to `.py` and launch it with `torchrun`:

```bash
jupyter nbconvert --to script main_scripts/pure_trm_reranker.ipynb
torchrun --nproc_per_node=2 main_scripts/pure_trm_reranker.py --profile full --output-root /path/to/output
```

## Outputs

Training notebooks write results to:

```text
<OUTPUT_ROOT>/trm_reranker_mvp/runs/<experiment_name>/
  checkpoints/
  logs/
    train_metrics.csv
    epoch_summaries.json
  eval/
  training_config.json
  run_artifacts.json
  fit_summary.json
```

The main metrics are `mrr@10`, `hit@1`, `hit@3`, `hit@5`, `hit@10`, and `ndcg@10`. Evaluation files also include a BM25/candidate-order baseline so the reranker can be compared against the input ranking.

`monot5_base_full_eval.ipynb` writes separate artifacts:

- `monot5_base_full_eval.run` in TREC run format;
- `monot5_base_full_eval.metrics.json`;
- `monot5_base_full_eval.config.json`;
- `monot5_base_full_eval.progress.json`;
- `monot5_base_full_eval.missing.json`;
- `monot5_base_full_eval.preview.tsv`.

## Latency Benchmarks

`latency_notebooks/all_architectures_latency.ipynb` compares the project architectures on synthetic input:

- `bert_encoder_trm`;
- `bert_embeddings_trm`;
- `bert_encoder_rerank_head`;
- `vanilla_transformer`.

For SOTA baselines, `latency_notebooks/sota_latency.ipynb` expects local models at:

```text
../models/monot5-base-msmarco
../models/bge-reranker-v2-m3
```

Latency results are saved as JSON files in the notebook's current working directory.

## Notes

- Notebook outputs are cleared: the repository does not contain saved execution results.
- `.venv` is present in the working directory, but it is a local environment rather than project logic.
- `monot5_base_full_eval.ipynb` forces Hugging Face offline mode, so the model must be downloaded beforehand.
- Large `full` runs require prepared caches and enough disk space for passage shards, checkpoints, and evaluation artifacts.
