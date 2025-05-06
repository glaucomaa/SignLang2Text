from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

from utils.model_base import BaseModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10_000, dtype=torch.float32):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=dtype)
        pos = torch.arange(0, max_len, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=dtype) * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # [max_len,1,E]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x:[S,B,E]
        return x + self.pe[: x.size(0)]


class CNNTransformerModel(BaseModel):
    def __init__(self, cfg):
        super().__init__()
        self.pad_idx = getattr(cfg, "pad_idx", 0)
        d_model = cfg.hidden_dim
        max_len = getattr(cfg, "max_len", 256)

        cnn = models.resnet18(pretrained=cfg.pretrained)
        if getattr(cfg, "freeze_cnn", False):
            for p in cnn.parameters():
                p.requires_grad = False
        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.frame_proj = nn.Linear(512, d_model)

        self.embed = nn.Embedding(cfg.vocab_size, d_model, padding_idx=self.pad_idx)
        self.pos_frames = PositionalEncoding(d_model)
        self.pos_tokens = PositionalEncoding(d_model)

        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1)
        self.register_buffer("causal_mask", mask)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_layers)
        self.out_proj = nn.Linear(d_model, cfg.vocab_size)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames : [B,T,C,H,W] → memory : [T,B,E]
        """
        B, T, C, H, W = frames.shape
        feats = self.cnn(frames.contiguous().view(B * T, C, H, W))  # [B*T,512]
        feats = self.frame_proj(feats).reshape(B, T, -1).permute(1, 0, 2)
        return self.pos_frames(feats)  # [T,B,E]

    def forward(
        self,
        frames: torch.Tensor,
        tgt_tokens: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        frames     : [B,T,C,H,W]      (video batch)
        tgt_tokens : [B,L]            (includes SOS at index 0)
        returns    : logits [B,L,V]
        """
        memory = self.encode_frames(frames)  # [T,B,E]

        tgt_emb = self.embed(tgt_tokens).permute(1, 0, 2)  # [L,B,E]
        tgt_emb = self.pos_tokens(tgt_emb)

        L = tgt_emb.size(0)
        tgt_mask = self.causal_mask[:L, :L]  # slice pre-comp mask
        tgt_key_padding_mask = tgt_tokens.eq(self.pad_idx)  # [B,L] bool

        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [L,B,E]

        logits = self.out_proj(out.permute(1, 0, 2))  # [B,L,V]
        return logits
