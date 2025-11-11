from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        length = x.size(1)
        return x + self.pe[:, :length]


class CoordTransformer(nn.Module):
    def __init__(
        self,
        obs_len: int,
        pred_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=obs_len + pred_len + 10)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder_queries = nn.Parameter(torch.randn(pred_len, d_model))
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, obs: Tensor) -> Tensor:
        """Predict future coordinates from observed (x,y) history."""
        if obs.dim() != 3 or obs.size(-1) != 2:
            raise ValueError("obs must be shaped [B, T_obs, 2]")
        B = obs.size(0)
        src = self.pos_encoder(self.input_proj(obs))
        memory = self.encoder(src)
        queries = self.decoder_queries.unsqueeze(0).expand(B, -1, -1)
        queries = self.pos_encoder(queries)
        decoded = self.decoder(queries, memory)
        future = self.output_proj(decoded)
        return future


__all__ = ["CoordTransformer"]
