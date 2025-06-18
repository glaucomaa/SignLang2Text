from __future__ import annotations
import math, logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from omegaconf import II
from fairseq import checkpoint_utils, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.data.sign_language import SignFeatsType
from fairseq.models import (
    FairseqEncoder, FairseqEncoderDecoderModel, register_model
)
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import (
    FairseqDropout, LayerNorm, PositionalEmbedding, TransformerEncoderLayer
)

logger = logging.getLogger(__name__)

@dataclass
class SLTTransformerConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu")
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1

    encoder_embed_dim: int = 512
    encoder_ffn_embed_dim: int = 2048
    encoder_layers: int = 12
    encoder_attention_heads: int = 8
    encoder_normalize_before: bool = True

    decoder_embed_dim: int = 512
    decoder_ffn_embed_dim: int = 2048
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    decoder_output_dim: int = 512
    decoder_normalize_before: bool = True
    share_decoder_input_output_embed: bool = False

    layernorm_embedding: bool = False
    no_scale_embedding: bool = False

    load_pretrained_encoder_from: Optional[str] = None
    load_pretrained_decoder_from: Optional[str] = None

    max_source_positions: int = II("task.max_source_positions")
    max_target_positions: int = II("task.max_target_positions")
    feats_type: ChoiceEnum([x.name for x in SignFeatsType]) = \
        II("task.feats_type")

@register_model("slt_transformer_fairseq", dataclass=SLTTransformerConfig)
class SLTTransformerModel(FairseqEncoderDecoderModel):

    @classmethod
    def _build_encoder(cls, cfg, feats_type, feat_dim):
        enc = SLTTransformerEncoder(cfg, feats_type, feat_dim)
        path = getattr(cfg, "load_pretrained_encoder_from", None)
        if path and Path(path).exists():
            enc = checkpoint_utils.load_pretrained_component_from_model(
                component=enc, checkpoint=path
            )
            logger.info(f"Loaded pretrained encoder from {path}")
        return enc

    @classmethod
    def _build_decoder(cls, cfg, task, embed_tokens):
        dec = TransformerDecoder(cfg, task.target_dictionary, embed_tokens)
        path = getattr(cfg, "load_pretrained_decoder_from", None)
        if path and Path(path).exists():
            dec = checkpoint_utils.load_pretrained_component_from_model(
                component=dec, checkpoint=path
            )
            logger.info(f"Loaded pretrained decoder from {path}")
        return dec

    @classmethod
    def build_model(cls, cfg, task):
        feat_dim = 1024 if cfg.feats_type == SignFeatsType.i3d else 195

        def build_embedding(dictionary, dim):
            return Embedding(len(dictionary), dim, dictionary.pad())

        dec_embed_tokens = build_embedding(
            task.target_dictionary, cfg.decoder_embed_dim
        )
        encoder = cls._build_encoder(cfg, cfg.feats_type, feat_dim)
        decoder = cls._build_decoder(cfg, task, dec_embed_tokens)
        return cls(encoder, decoder)

    def forward(
        self,
        src_feats: torch.Tensor,        
        tgt_tokens: torch.Tensor,      
        memory_key_padding_mask: torch.Tensor | None = None,
    ):
        enc_out = self.encoder(
            src_tokens=src_feats,
            encoder_padding_mask=memory_key_padding_mask,
        )
        dec_out, _ = self.decoder(
            prev_output_tokens=tgt_tokens,
            encoder_out=enc_out
        )
        return dec_out


class SLTTransformerEncoder(FairseqEncoder):
    def __init__(self, cfg, feats_type: SignFeatsType, feat_dim: int):
        super().__init__(None)
        self.num_updates = 0
        self.padding_idx = 1
        self.feats_type = feats_type
        if feats_type in {SignFeatsType.mediapipe, SignFeatsType.openpose}:
            self.feat_proj = nn.Linear(feat_dim * 3, cfg.encoder_embed_dim)
        else:  # i3d
            self.feat_proj = nn.Linear(feat_dim, cfg.encoder_embed_dim)
        self.embed_scale = math.sqrt(cfg.encoder_embed_dim)
        if cfg.no_scale_embedding:
            self.embed_scale = 1.0
        self.embed_positions = PositionalEmbedding(
            cfg.max_source_positions, cfg.encoder_embed_dim, self.padding_idx
        )
        self.dropout_module = FairseqDropout(cfg.dropout, self.__class__.__name__)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(cfg) for _ in range(cfg.encoder_layers)]
        )
        self.layer_norm = LayerNorm(cfg.encoder_embed_dim) \
            if cfg.encoder_normalize_before else None

    def forward(
        self,
        src_tokens: torch.Tensor,              
        encoder_padding_mask: torch.Tensor | None,  
        return_all_hiddens: bool = False,
    ):
        if self.feats_type == SignFeatsType.mediapipe:
            B, T, _ = src_tokens.shape
            src_tokens = src_tokens.view(B, T, -1)

        x = self.feat_proj(src_tokens)          
        x = self.embed_scale * x
        pos = self.embed_positions(
            torch.zeros_like(x[..., 0], dtype=torch.long)  
        )
        x = (x + pos)                            
        x = self.dropout_module(x)
        x = x.transpose(0, 1)                    

        encoder_states = []
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],                   
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask is not None else [],
            "encoder_embedding": [],
            "encoder_states": encoder_states,
            "src_tokens": [],
        }
    def reorder_encoder_out(self, encoder_out, new_order):
        def _reord(tensors):
            return [] if len(tensors) == 0 else [t.index_select(1, new_order) for t in tensors]

        return {
            "encoder_out": _reord(encoder_out["encoder_out"]),
            "encoder_padding_mask": _reord(encoder_out["encoder_padding_mask"]),
            "encoder_embedding": _reord(encoder_out["encoder_embedding"]),
            "encoder_states": [
                s.index_select(1, new_order) for s in encoder_out["encoder_states"]
            ] if len(encoder_out["encoder_states"]) else [],
            "src_tokens": [],
        }
