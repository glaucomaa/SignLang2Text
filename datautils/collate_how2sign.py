from typing import List, Tuple, Callable
from functools import partial
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_i3d_fn(pad_idx: int, samples: List[Tuple[torch.Tensor, torch.Tensor]]):
    feats_list, tgt_list = zip(*samples)
    feats = pad_sequence(feats_list, batch_first=True, padding_value=0.0)     # [B, T_max, 1024]
    targets = pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx) # [B, L_max]
    return feats, targets


def make_collate_fn(pad_idx: int) -> Callable:
    return partial(collate_i3d_fn, pad_idx)
