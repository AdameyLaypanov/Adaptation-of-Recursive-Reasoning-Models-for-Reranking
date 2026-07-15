"""Query-passage pair encoding shared by training and evaluation."""

from dataclasses import dataclass

import torch


@dataclass
class PairEncoder:
    """Encodes a (query tokens, passage tokens) pair into model inputs.

    Layout (identical to the legacy notebooks):
      input_ids       = [CLS] q [SEP] d [SEP] [PAD]...
      token_type_ids  = 0, 1*len(q), 0, 2*len(d), 0, 0...   (TRM segments)
      attention_mask  = 1 for real tokens, 0 for padding

    With ``emit_bert_token_type_ids=True`` (BERT variants) it additionally emits
      bert_token_type_ids = 0, 0*len(q), 0, 1*len(d), 1, 0...
    """

    cls_id: int
    sep_id: int
    pad_id: int
    seq_len: int
    max_query_len: int
    max_doc_len: int
    emit_bert_token_type_ids: bool = False

    def encode_pair(self, query_tokens: list[int], passage_tokens: list[int]) -> dict[str, list[int]]:
        q_tokens = list(query_tokens[: self.max_query_len])
        d_tokens = list(passage_tokens[: self.max_doc_len])

        input_ids = [self.cls_id, *q_tokens, self.sep_id, *d_tokens, self.sep_id]
        token_type_ids = [0] + [1] * len(q_tokens) + [0] + [2] * len(d_tokens) + [0]
        attention_mask = [1] * len(input_ids)

        pad_len = self.seq_len - len(input_ids)
        if pad_len < 0:
            raise ValueError(f"Pair length {len(input_ids)} exceeds configured seq_len={self.seq_len}")

        encoded = {
            "input_ids": input_ids + [self.pad_id] * pad_len,
            "token_type_ids": token_type_ids + [0] * pad_len,
            "attention_mask": attention_mask + [0] * pad_len,
        }
        if self.emit_bert_token_type_ids:
            bert_token_type_ids = [0] + [0] * len(q_tokens) + [0] + [1] * len(d_tokens) + [1]
            encoded["bert_token_type_ids"] = bert_token_type_ids + [0] * pad_len
        return encoded


def collate_encoded_pairs(encoded_pairs: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    keys = encoded_pairs[0].keys()
    return {key: torch.tensor([item[key] for item in encoded_pairs], dtype=torch.long) for key in keys}


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def move_to_device(batch, device: torch.device):
    """Move a collated batch to a device: a tensor dict or a tuple/list of them."""
    if isinstance(batch, dict):
        return move_batch_to_device(batch, device)
    return type(batch)(move_to_device(item, device) for item in batch)
