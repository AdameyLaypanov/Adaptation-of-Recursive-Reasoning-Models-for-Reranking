"""End-to-end Trainer tests on synthetic data: train, checkpoints, resume, InfoNCE.

The bundle is built by hand (no manifests): tiny vocab, 8 queries, 16 passages,
32 pairwise triples. Everything runs on CPU in float32 within seconds.
"""

import csv
import dataclasses
import json

import pytest
import torch

from trm_reranker.config import TrainingSection
from trm_reranker.data.encoding import PairEncoder
from trm_reranker.models import TRMReranker
from trm_reranker.runtime import RunDataBundle
from trm_reranker.training.checkpoints import RunPaths
from trm_reranker.training.trainer import Trainer

SEQ_LEN = 24
VOCAB = 200
NUM_QUERIES = 8
NUM_PASSAGES = 16


def _tokens(rng, length):
    return [int(rng.integers(3, VOCAB)) for _ in range(length)]


def make_bundle(tmp_path, hard_negatives_path=None):
    rng = __import__("numpy").random.default_rng(0)
    query_tokens = {qid: _tokens(rng, 5) for qid in range(NUM_QUERIES)}
    passage_tokens = {pid: _tokens(rng, 10) for pid in range(100, 100 + NUM_PASSAGES)}
    pids = sorted(passage_tokens)

    triples_path = tmp_path / "triples.tsv"
    with triples_path.open("w", encoding="utf-8") as handle:
        for i in range(32):
            qid = i % NUM_QUERIES
            pos_pid = pids[i % NUM_PASSAGES]
            neg_pid = pids[(i + 7) % NUM_PASSAGES]
            handle.write(f"{qid}\t{pos_pid}\t{neg_pid}\n")

    encoder = PairEncoder(cls_id=1, sep_id=2, pad_id=0, seq_len=SEQ_LEN, max_query_len=6, max_doc_len=12)

    dev_qids = list(range(4))
    candidates_per_query = 5
    dev_candidates = {
        "qid_order": dev_qids,
        "qid_offsets": [i * candidates_per_query for i in range(len(dev_qids) + 1)],
        "pid": [pids[(qid * 3 + j) % NUM_PASSAGES] for qid in dev_qids for j in range(candidates_per_query)],
        "bm25_rank": [j + 1 for _ in dev_qids for j in range(candidates_per_query)],
    }
    dev_qrels = {qid: {pids[(qid * 3) % NUM_PASSAGES]} for qid in dev_qids}

    return RunDataBundle(
        tokenizer=None,
        seq_len=SEQ_LEN,
        max_query_len=6,
        max_doc_len=12,
        encoder=encoder,
        train_query_tokens=query_tokens,
        train_passage_tokens=passage_tokens,
        dev_query_tokens=query_tokens,
        epoch_dev_candidates=dev_candidates,
        epoch_dev_qrels=dev_qrels,
        final_dev_candidates=dev_candidates,
        final_dev_qrels=dev_qrels,
        run_final_full_dev=True,
        epoch_dev_mode_label="quick_test",
        sampled_train_triples_path=triples_path,
        passage_token_getter=lambda pid: passage_tokens[pid],
        hard_negatives_path=hard_negatives_path,
    )


def make_training_cfg(**overrides):
    cfg = TrainingSection(
        per_device_batch_size=4,
        eval_batch_size=8,
        learning_rate=1e-3,
        precision="32-true",
        epochs=1,
        run_train_steps=5,
        checkpoint_every_n_steps=2,
        checkpoint_epoch_fraction=None,
        keep_last_step_checkpoints=2,
        dev_eval_query_limit=2,
        tqdm_postfix_every_n_steps=10_000,
    )
    return dataclasses.replace(cfg, **overrides)


def tiny_model():
    torch.manual_seed(13)
    return TRMReranker(
        dict(
            batch_size=4,
            seq_len=SEQ_LEN,
            vocab_size=VOCAB,
            H_cycles=1,
            L_cycles=1,
            H_layers=1,
            L_layers=1,
            hidden_size=32,
            expansion=2.0,
            num_heads=4,
            pos_encodings="rope",
        )
    )


def make_trainer(tmp_path, cfg, bundle, run_name="exp"):
    paths = RunPaths.create(tmp_path / "out", run_name)
    return (
        Trainer(
            model=tiny_model(),
            device=torch.device("cpu"),
            cfg=cfg,
            paths=paths,
            bundle=bundle,
            seed=13,
            effective_precision="32-true",
            should_use_ddp=False,
            world_size=1,
        ),
        paths,
    )


def read_csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_trainer_pairwise_session_checkpoints_and_resume(tmp_path):
    bundle = make_bundle(tmp_path)

    # Session 1: 32 triples / batch 4 = 8 steps per epoch; the session window stops at 5.
    trainer, paths = make_trainer(tmp_path, make_training_cfg(), bundle)
    assert trainer.steps_per_epoch == 8
    summary = trainer.fit()
    assert summary["global_step"] == 5
    assert summary["training_complete"] is False
    assert paths.last_checkpoint_path.exists()

    # keep_last_step_checkpoints=2: periodic checkpoints were written at steps 2 and 4.
    step_checkpoints = sorted(p.name for p in paths.checkpoint_dir.glob("step_*.pt"))
    assert len(step_checkpoints) == 2

    train_rows = read_csv_rows(paths.train_log_path)
    assert len(train_rows) == 5
    assert all(float(row["loss"]) > 0 for row in train_rows)
    assert paths.dev_metrics_by_step_path.exists()

    checkpoint = torch.load(paths.last_checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 5
    assert checkpoint["config"]["loss"] == "pairwise_logistic"

    # Session 2: resume from last_checkpoint and finish the epoch (8 steps total).
    resume_cfg = make_training_cfg(
        run_train_steps=None,
        resume_from_checkpoint=True,
        resume_checkpoint_path=str(paths.last_checkpoint_path),
    )
    resumed_trainer, paths = make_trainer(tmp_path, resume_cfg, bundle)
    resumed_summary = resumed_trainer.fit()
    assert resumed_summary["global_step"] == 8
    assert resumed_summary["training_complete"] is True

    # Reaching target_total_steps triggers the final full dev eval.
    assert paths.final_metrics_path.exists()
    assert "final_full_dev_mrr10" in resumed_summary
    final_metrics = json.loads(paths.final_metrics_path.read_text())
    assert final_metrics["queries_evaluated"] == 4

    # The train log was appended, not rewritten.
    train_rows = read_csv_rows(paths.train_log_path)
    assert len(train_rows) == 8
    assert [int(row["global_step"]) for row in train_rows] == list(range(1, 9))


def test_trainer_resume_rejects_changed_step_accounting(tmp_path):
    bundle = make_bundle(tmp_path)
    trainer, paths = make_trainer(tmp_path, make_training_cfg(run_train_steps=2), bundle)
    trainer.fit()

    resume_cfg = make_training_cfg(
        per_device_batch_size=8,  # changes steps_per_epoch: 32/8=4 != 8
        resume_from_checkpoint=True,
        resume_checkpoint_path=str(paths.last_checkpoint_path),
    )
    broken_trainer, _ = make_trainer(tmp_path, resume_cfg, bundle)
    with pytest.raises(ValueError, match="estimated_steps_per_epoch"):
        broken_trainer.fit()


def write_hard_negatives(tmp_path, bundle):
    pids = sorted(bundle.train_passage_tokens)
    path = tmp_path / "hard_negatives.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for qid in range(NUM_QUERIES):
            record = {
                "qid": qid,
                "pos_pid": pids[qid],
                "neg_pids": [pids[(qid + offset) % NUM_PASSAGES] for offset in range(1, 6)],
            }
            handle.write(json.dumps(record) + "\n")
    return path


def test_trainer_infonce_with_hard_negatives(tmp_path):
    bundle = make_bundle(tmp_path)
    hard_negatives_path = write_hard_negatives(tmp_path, bundle)
    bundle = dataclasses.replace(bundle, hard_negatives_path=hard_negatives_path)

    cfg = make_training_cfg(loss="infonce", num_negatives=3, run_train_steps=2, checkpoint_every_n_steps=1)
    trainer, paths = make_trainer(tmp_path, cfg, bundle, run_name="exp_infonce")
    # 8 groups / batch 4 = 2 steps per epoch.
    assert trainer.steps_per_epoch == 2
    assert len(trainer.train_dataset) == NUM_QUERIES

    summary = trainer.fit()
    assert summary["global_step"] == 2
    train_rows = read_csv_rows(paths.train_log_path)
    assert len(train_rows) == 2
    for row in train_rows:
        assert 0.0 <= float(row["pairwise_acc"]) <= 1.0
        assert float(row["loss"]) > 0

    checkpoint = torch.load(paths.last_checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["config"]["loss"] == "infonce"
    assert checkpoint["config"]["num_negatives"] == 3
    assert checkpoint["config"]["hard_negatives_path"] == str(hard_negatives_path)


def test_trainer_rejects_unknown_loss(tmp_path):
    bundle = make_bundle(tmp_path)
    with pytest.raises(ValueError, match=r"Unknown training\.loss"):
        make_trainer(tmp_path, make_training_cfg(loss="triplet"), bundle)
