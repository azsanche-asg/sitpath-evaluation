from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor, nn


class SitPathGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        obs_len: int,
        pred_len: int,
        hidden_size: int = 128,
        mode: Literal["token", "coord"] = "token",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.decoder_cell = nn.GRUCell(hidden_size, hidden_size)
        if mode == "token":
            self.output_proj = nn.Linear(hidden_size, vocab_size)
            self.feedback_proj: Optional[nn.Linear] = None
        else:
            self.output_proj = nn.Linear(hidden_size, 2)
            self.feedback_proj = nn.Linear(2, hidden_size)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.dim() != 2:
            raise ValueError("tokens must be shaped [B, T]")
        emb = self.embedding(tokens)
        _, h = self.encoder(emb)
        h = h.squeeze(0)
        prev_input = torch.zeros(tokens.size(0), self.hidden_size, device=tokens.device, dtype=emb.dtype)
        outputs = []
        for _ in range(self.pred_len):
            h = self.decoder_cell(prev_input, h)
            logits = self.output_proj(h)
            outputs.append(logits.unsqueeze(1))
            if self.mode == "token":
                prev_ids = torch.argmax(logits, dim=-1)
                prev_input = self.embedding(prev_ids)
            else:
                prev_input = self.feedback_proj(logits)
        return torch.cat(outputs, dim=1)


__all__ = ["SitPathGRU"]
