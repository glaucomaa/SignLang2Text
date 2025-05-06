from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def save_checkpoint(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, *init_args, **init_kwargs):
        model = cls(*init_args, **init_kwargs)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        return model
