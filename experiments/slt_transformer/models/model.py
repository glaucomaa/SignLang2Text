from __future__ import annotations
import torch
import torch.nn as nn
from utils.model_base import BaseModel
import math



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.pe = self._build_pe(max_len)

    def _build_pe(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            self.pe = self._build_pe(seq_len * 2).to(x.device)  # удвоим с запасом
        return x + self.pe[:seq_len].unsqueeze(0).to(x.device) # hot fix


class SLTTransformerModel(BaseModel):
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg.input_dim
        vocab_size = cfg.vocab_size
        d_model = cfg.d_model
        nhead = cfg.nhead
        num_encoder_layers = cfg.num_encoder_layers
        num_decoder_layers = cfg.num_decoder_layers
        dim_feedforward = cfg.dim_feedforward
        dropout = cfg.dropout
        pad_idx = cfg.pad_idx
        max_len = cfg.max_len

        self.pad_idx = pad_idx
        self.input_fc = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_src = PositionalEncoding(d_model, max_len)
        self.pos_tgt = PositionalEncoding(d_model, max_len)
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1)
        self.register_buffer("causal_mask", mask)  

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src_feats: torch.Tensor,          
        tgt_tokens: torch.Tensor,         
        memory_key_padding_mask: torch.Tensor | None = None, 
    ) -> torch.Tensor:
        """
        src_feats: тензор размерности [B, T_src, input_dim] (I3D-фичи)
        tgt_tokens: тензор [B, T_tgt], включающий начальный <SOS> в индексе 0
        memory_key_padding_mask: булев маск [B, T_src], где True = pad-позиция
        Возвращает logits [B, T_tgt, vocab_size]
        """
        src_emb = self.pos_src(self.input_fc(src_feats)) 
        memory = self.encoder(src_emb, src_key_padding_mask=memory_key_padding_mask)  
        tgt_emb = self.pos_tgt(self.embedding(tgt_tokens)) 
        L = tgt_emb.size(1) 
        tgt_mask = self.causal_mask[:L, :L] 
        tgt_key_padding_mask = tgt_tokens.eq(self.pad_idx) 
        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  
        logits = self.out_proj(out) 
        return logits
