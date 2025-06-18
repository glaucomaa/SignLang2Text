from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset
import sentencepiece as spm
import pandas as pd
import numpy as np


class How2SignI3DDataset(Dataset):
    """
    Dataset для How2Sign с I3D-фичами и SentencePiece токенами.
    Требует:
    - data_dir: путь до .npy файлов (e.g., data/how2sign/train/)
    - annotation_file: путь до .tsv файла с колонками ['id', 'translation']
    - sp_model_path: путь до .model-файла SentencePiece
    """

    def __init__(self, data_dir: str | Path, annotation_file: str | Path, sp_model_path: str):
        self.data_dir = Path(data_dir)
        self.samples = pd.read_csv(annotation_file, sep="\t", usecols=["id", "translation"])
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples.iloc[idx]
        sample_id = row["id"]
        text = row["translation"]
        path = self.data_dir / f"{sample_id}.npy"
        #if idx < 5:
        #    print(f"[DEBUG] idx={idx} id={sample_id} text_len={len(text)} chars, first50=\"{text[:50]}\"")

        feats = torch.from_numpy(np.load(path))  # [T, 1024]
        tokens = [self.sp.bos_id()] + self.sp.encode(text, out_type=int) + [self.sp.eos_id()]
        #if idx < 5:
        #    print(f"[DEBUG] idx={idx} tokens_len={len(tokens)}")
        return feats, torch.tensor(tokens, dtype=torch.long)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    @property
    def pad_index(self):
        return self.sp.pad_id()
