"""Training datasets and collate functions."""

import csv
import json
import logging
import random
from collections.abc import Callable, Sequence
from pathlib import Path

from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from ..utils import make_tqdm
from .encoding import PairEncoder, collate_encoded_pairs

logger = logging.getLogger(__name__)

# One training group: (qid, positive pid, candidate negative pids).
TripleGroup = tuple[int, int, tuple[int, ...]]


def read_triples_tsv(triples_path: Path, show_progress: bool = True) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    with Path(triples_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in make_tqdm(reader, desc=f"Load {Path(triples_path).name}", disable=not show_progress):
            if len(row) >= 3:
                triples.append((int(row[0]), int(row[1]), int(row[2])))
    return triples


class PairwiseTripleDataset(Dataset):
    def __init__(
        self,
        triples_path: Path,
        query_token_map: dict[int, list[int]],
        passage_token_map: dict[int, list[int]],
        show_progress: bool = True,
    ):
        self.query_token_map = query_token_map
        self.passage_token_map = passage_token_map
        self.triples = read_triples_tsv(triples_path, show_progress=show_progress)

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, index: int):
        qid, pos_pid, neg_pid = self.triples[index]
        if qid not in self.query_token_map:
            raise KeyError(f"Missing query tokens for qid={qid}")
        if pos_pid not in self.passage_token_map or neg_pid not in self.passage_token_map:
            raise KeyError(f"Missing cached passage tokens for one of {pos_pid}, {neg_pid}")
        return {
            "qid": qid,
            "query_tokens": self.query_token_map[qid],
            "pos_tokens": self.passage_token_map[pos_pid],
            "neg_tokens": self.passage_token_map[neg_pid],
        }


def make_pairwise_collate(encoder: PairEncoder):
    def pairwise_collate(batch_items: list[dict[str, object]]):
        pos_pairs = [encoder.encode_pair(item["query_tokens"], item["pos_tokens"]) for item in batch_items]
        neg_pairs = [encoder.encode_pair(item["query_tokens"], item["neg_tokens"]) for item in batch_items]
        return collate_encoded_pairs(pos_pairs), collate_encoded_pairs(neg_pairs)

    return pairwise_collate


def load_hard_negative_groups(path: Path) -> list[TripleGroup]:
    """Read groups mined by scripts/mine_hard_negatives.py (JSONL: qid/pos_pid/neg_pids)."""
    groups: list[TripleGroup] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                qid = int(record["qid"])
                pos_pid = int(record["pos_pid"])
                neg_pids = tuple(int(pid) for pid in record["neg_pids"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Bad hard-negatives record at {path}:{line_number}: {error}") from error
            groups.append((qid, pos_pid, neg_pids))
    if not groups:
        raise ValueError(f"Hard-negatives file has no records: {path}")
    return groups


def group_pairwise_triples(triples: Sequence[tuple[int, int, int]]) -> list[TripleGroup]:
    """Collect (qid, pos) -> all negatives seen in pairwise triples.

    Fallback for multi-negative losses when no mined hard-negatives file is
    configured; official triples usually give only a few negatives per group,
    so sampling will repeat them.
    """
    negatives: dict[tuple[int, int], list[int]] = {}
    for qid, pos_pid, neg_pid in triples:
        negatives.setdefault((qid, pos_pid), []).append(neg_pid)
    return [(qid, pos_pid, tuple(dict.fromkeys(negs))) for (qid, pos_pid), negs in negatives.items()]


class GroupedTripleDataset(Dataset):
    """(query, positive, num_negatives sampled negatives) groups for multi-negative losses.

    Negatives are re-sampled on every ``__getitem__`` from the group's candidate
    pool (without replacement when the pool is large enough). ``passage_token_lookup``
    must cover every pid in the groups — pass a shard-store getter, not just the
    train subset map, when groups come from mined hard negatives.
    """

    def __init__(
        self,
        groups: Sequence[TripleGroup],
        query_token_map: dict[int, list[int]],
        passage_token_lookup: Callable[[int], list[int]],
        num_negatives: int,
        seed: int = 13,
    ):
        if num_negatives < 1:
            raise ValueError("num_negatives must be >= 1")
        self.query_token_map = query_token_map
        self.passage_token_lookup = passage_token_lookup
        self.num_negatives = int(num_negatives)
        self.rng = random.Random(seed)

        self.groups: list[TripleGroup] = []
        skipped_queries = 0
        for qid, pos_pid, neg_pids in groups:
            if qid not in query_token_map:
                skipped_queries += 1
                continue
            if not neg_pids:
                continue
            self.groups.append((qid, pos_pid, tuple(neg_pids)))
        if not self.groups:
            raise ValueError("No usable groups: none of the qids have cached query tokens")
        if skipped_queries:
            logger.warning(
                "GroupedTripleDataset: skipped %d groups whose qid has no cached query tokens", skipped_queries
            )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int):
        qid, pos_pid, neg_pids = self.groups[index]
        if len(neg_pids) >= self.num_negatives:
            sampled = self.rng.sample(neg_pids, self.num_negatives)
        else:
            sampled = self.rng.choices(neg_pids, k=self.num_negatives)
        return {
            "qid": qid,
            "query_tokens": self.query_token_map[qid],
            "pos_tokens": self.passage_token_lookup(pos_pid),
            "neg_tokens_list": [self.passage_token_lookup(pid) for pid in sampled],
        }


def make_grouped_collate(encoder: PairEncoder):
    """Flatten groups into one batch of pairs: [pos, neg_1..neg_K] per group, groups in order."""

    def grouped_collate(batch_items: list[dict[str, object]]):
        encoded_pairs = []
        for item in batch_items:
            encoded_pairs.append(encoder.encode_pair(item["query_tokens"], item["pos_tokens"]))
            for neg_tokens in item["neg_tokens_list"]:
                encoded_pairs.append(encoder.encode_pair(item["query_tokens"], neg_tokens))
        return collate_encoded_pairs(encoded_pairs)

    return grouped_collate


class ResumeDistributedSampler(DistributedSampler):
    """DistributedSampler with a per-rank start offset for efficient mid-epoch resume."""

    def __init__(self, *args, start_index: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_index = max(0, int(start_index))

    def __iter__(self):
        indices = list(super().__iter__())
        if self.start_index:
            indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return max(0, super().__len__() - self.start_index)


def iter_grouped_candidates(candidates_artifact, query_limit: int | None = None):
    qid_order = candidates_artifact["qid_order"]
    qid_offsets = candidates_artifact["qid_offsets"]
    pid_values = candidates_artifact["pid"]
    bm25_ranks = candidates_artifact["bm25_rank"]
    limit = len(qid_order) if query_limit is None else min(len(qid_order), query_limit)
    for idx in range(limit):
        qid = int(qid_order[idx])
        start = int(qid_offsets[idx])
        end = int(qid_offsets[idx + 1])
        yield qid, pid_values[start:end], bm25_ranks[start:end]
