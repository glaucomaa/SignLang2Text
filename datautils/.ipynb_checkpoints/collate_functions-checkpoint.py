from __future__ import annotations

from functools import partial
from typing import Callable, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def _collate_core(
    pad_idx: int, samples: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    pad_idx : int
    samples : list[(frames, target)]
    """
    frames_list, tgt_list = zip(*samples)

    padded_frames = pad_sequence(
        frames_list, batch_first=True, padding_value=0.0
    )  # [B,T,C,H,W]

    targets = pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)  # [B,L]

    return padded_frames.contiguous(), targets.contiguous()


def make_collate_fn(pad_idx: int) -> Callable:
    return partial(_collate_core, pad_idx)
