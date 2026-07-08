"""Training datasets and collate functions."""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from ..utils import make_tqdm
from .encoding import PairEncoder, collate_encoded_pairs


class PairwiseTripleDataset(Dataset):
    def __init__(self, triples_path: Path, query_token_map: Dict[int, List[int]], passage_token_map: Dict[int, List[int]], show_progress: bool = True):
        self.query_token_map = query_token_map
        self.passage_token_map = passage_token_map
        self.triples: List[Tuple[int, int, int]] = []
        with Path(triples_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in make_tqdm(reader, desc=f"Load {Path(triples_path).name}", disable=not show_progress):
                if len(row) >= 3:
                    self.triples.append((int(row[0]), int(row[1]), int(row[2])))

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
    def pairwise_collate(batch_items: List[Dict[str, object]]):
        pos_pairs = [encoder.encode_pair(item["query_tokens"], item["pos_tokens"]) for item in batch_items]
        neg_pairs = [encoder.encode_pair(item["query_tokens"], item["neg_tokens"]) for item in batch_items]
        return collate_encoded_pairs(pos_pairs), collate_encoded_pairs(neg_pairs)

    return pairwise_collate


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


def iter_grouped_candidates(candidates_artifact, query_limit: int = None):
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
