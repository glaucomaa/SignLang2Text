from __future__ import annotations
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import TRANSFORM_REGISTRY

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


class Vocab:
    def __init__(self, texts: List[str]):
        specials = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        chars = sorted({c for t in texts for c in t})
        self.itos = specials + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, text: str) -> List[int]:
        unk = self.stoi[UNK_TOKEN]
        return (
            [self.stoi[SOS_TOKEN]]
            + [self.stoi.get(c, unk) for c in text]
            + [self.stoi[EOS_TOKEN]]
        )

    @property
    def pad_index(self) -> int:
        return self.stoi[PAD_TOKEN]

    def __len__(self):
        return len(self.itos)


class SignLanguageDataset(Dataset):
    """
    data_dir : str | Path
    annotation_file : str | Path
    transform : str
    image_mode : "RGB" | "L"
    """

    def __init__(
        self,
        data_dir: str | Path,
        annotation_file: str | Path,
        transform: str = "default",
        image_mode: str = "RGB",
    ):
        self.data_dir = Path(data_dir)
        ids, texts = [], []
        with open(annotation_file, encoding="utf-8") as fh:
            for line in fh:
                sample_id, sentence = line.rstrip("\n").split("\t")
                ids.append(sample_id)
                texts.append(sentence)

        self.sample_ids = ids
        self.texts = texts
        self.vocab = Vocab(texts)
        self.transform = TRANSFORM_REGISTRY[transform]
        self.image_mode = image_mode

    def __len__(self):
        return len(self.sample_ids)

    # def _load_frames(self, sample_id: str) -> torch.Tensor:
    #     frame_dir = self.data_dir / sample_id
    #     frames = sorted(frame_dir.glob("*.jpg"))
    #     imgs = [self.transform(Image.open(p).convert(self.image_mode)) for p in frames]
    #     return torch.stack(imgs)  # [T,C,H,W]

    def _load_frames(self, sample_id: str) -> torch.Tensor:
        frame_dir = self.data_dir / sample_id
        frames = sorted(frame_dir.glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(f"No frames found in: {frame_dir}")
        imgs = [self.transform(Image.open(p).convert("RGB")) for p in frames]
        return torch.stack(imgs)  # [T, C, H, W]

    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]
        frames = self._load_frames(sample_id)
        target = torch.tensor(self.vocab.encode(self.texts[idx]), dtype=torch.long)
        return frames, target
