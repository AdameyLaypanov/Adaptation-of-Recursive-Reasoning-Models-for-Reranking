"""Hard-negative mining: candidate/qrels parsing, rank band, positive exclusion."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "mine_hard_negatives.py"

spec = importlib.util.spec_from_file_location("mine_hard_negatives", SCRIPT_PATH)
mining = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mining)


def write_trec_run(path, rankings):
    with path.open("w", encoding="utf-8") as handle:
        for qid, pids in rankings.items():
            for rank, pid in enumerate(pids, start=1):
                handle.write(f"{qid} Q0 {pid} {rank} {1.0 / rank:.4f} bm25\n")


def test_read_candidates_trec_and_tsv(tmp_path):
    trec = tmp_path / "run.trec"
    write_trec_run(trec, {1: [100, 101]})
    assert mining.read_candidates(trec) == {1: [(100, 1), (101, 2)]}

    tsv = tmp_path / "run.tsv"
    tsv.write_text("1\t100\t1\n1\t101\t2\n")
    assert mining.read_candidates(tsv) == {1: [(100, 1), (101, 2)]}


def test_read_qrels_msmarco_and_tsv(tmp_path):
    msmarco = tmp_path / "qrels.txt"
    msmarco.write_text("1 0 100 1\n1 0 101 0\n2 0 102 1\n")
    positives = mining.read_qrels(msmarco)
    assert positives[1] == {100}  # rel=0 is not a positive
    assert positives[2] == {102}

    tsv = tmp_path / "qrels.tsv"
    tsv.write_text("1\t100\n")
    assert mining.read_qrels(tsv)[1] == {100}


def test_mine_respects_band_and_excludes_positives():
    candidates = {1: [(100 + rank, rank) for rank in range(1, 21)]}
    candidates[1][4] = (777, 5)  # pid 777 at rank 5 is a known positive
    positives = {1: {777}}

    records, stats = mining.mine_hard_negatives(
        candidates, positives, num_negatives=100, rank_min=3, rank_max=10, min_negatives=1, seed=13
    )
    assert stats["groups_emitted"] == 1
    [record] = records
    assert record["qid"] == 1 and record["pos_pid"] == 777
    in_band_pids = {100 + rank for rank in range(3, 11)} - {105}  # rank 5 was replaced by the positive
    assert set(record["neg_pids"]) == in_band_pids
    assert 777 not in record["neg_pids"]


def test_mine_skips_queries_without_qrels_or_enough_negatives():
    candidates = {1: [(101, 1), (102, 2)], 2: [(103, 1)]}
    positives = {1: {999}}  # qid 2 has no qrels
    records, stats = mining.mine_hard_negatives(
        candidates, positives, num_negatives=5, rank_min=1, rank_max=10, min_negatives=3, seed=13
    )
    assert records == []
    assert stats["queries_without_qrels"] == 1
    assert stats["queries_too_few_negatives"] == 1


def test_mine_samples_at_most_num_negatives():
    candidates = {1: [(100 + rank, rank) for rank in range(1, 31)]}
    positives = {1: {999}}
    records, _ = mining.mine_hard_negatives(
        candidates, positives, num_negatives=7, rank_min=1, rank_max=30, min_negatives=1, seed=13
    )
    [record] = records
    assert len(record["neg_pids"]) == 7
    assert len(set(record["neg_pids"])) == 7


@pytest.mark.parametrize("candidates_format", ["trec", "tsv"])
def test_cli_end_to_end(tmp_path, candidates_format):
    if candidates_format == "trec":
        candidates_path = tmp_path / "run.trec"
        write_trec_run(candidates_path, {1: list(range(100, 130)), 2: list(range(200, 230))})
    else:
        candidates_path = tmp_path / "run.tsv"
        candidates_path.write_text(
            "".join(f"{qid}\t{qid * 100 + rank}\t{rank}\n" for qid in (1, 2) for rank in range(1, 31))
        )
    qrels_path = tmp_path / "qrels.txt"
    qrels_path.write_text("1 0 100 1\n2 0 999 1\n")
    out_path = tmp_path / "mined.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--candidates",
            str(candidates_path),
            "--qrels",
            str(qrels_path),
            "--out",
            str(out_path),
            "--num-negatives",
            "5",
            "--rank-min",
            "2",
            "--rank-max",
            "20",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stats = json.loads(result.stdout)
    assert stats["groups_emitted"] == 2
    records = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert {record["qid"] for record in records} == {1, 2}
    for record in records:
        assert len(record["neg_pids"]) == 5
