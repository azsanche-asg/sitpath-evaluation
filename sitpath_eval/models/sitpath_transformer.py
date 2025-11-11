from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn

from ..tokenizer.inverse_decoder import DecodeState, SitPathInverseDecoder

from .coord_transformer import SinusoidalPositionalEncoding


class SitPathTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
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
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
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
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return logits over vocabulary for each future step."""
        if token_ids.dim() != 2:
            raise ValueError("token_ids must be [B, T_obs]")
        B = token_ids.size(0)
        src = self.pos_encoder(self.token_embed(token_ids))
        memory = self.encoder(src)
        queries = self.decoder_queries.unsqueeze(0).expand(B, -1, -1)
        queries = self.pos_encoder(queries)
        decoded = self.decoder(queries, memory)
        logits = self.output_proj(decoded)
        return logits

    @torch.no_grad()
    def decode_to_coords(
        self,
        logits: Tensor,
        inverse_decoder: SitPathInverseDecoder,
        token_lookup: Dict[int, Dict],
        last_positions: Tensor,
    ) -> Tensor:
        """Convert logits -> coords via greedy token decoding + inverse decoder."""
        token_ids = torch.argmax(logits, dim=-1)
        B, T = token_ids.shape
        coords = torch.zeros(B, T, 2, device=logits.device)
        for b in range(B):
            state = DecodeState(position=last_positions[b].detach().cpu().numpy())
            for t in range(T):
                token_dict = token_lookup.get(int(token_ids[b, t].item()))
                if token_dict is None:
                    coords[b, t] = torch.tensor(state.position, device=logits.device)
                    continue
                new_pos = inverse_decoder.decode_step(state, token_dict)
                coords[b, t] = torch.tensor(new_pos, device=logits.device)
        return coords


__all__ = ["SitPathTransformer"]
