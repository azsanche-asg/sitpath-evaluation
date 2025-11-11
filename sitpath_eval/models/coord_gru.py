from __future__ import annotations

import torch
from torch import Tensor, nn


class CoordGRU(nn.Module):
    def __init__(self, obs_len: int, pred_len: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.encoder = nn.GRU(input_size=2, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.decoder_cell = nn.GRUCell(input_size=2, hidden_size=hidden_size)
        self.output_proj = nn.Linear(hidden_size, 2)

    def forward(self, obs: Tensor) -> Tensor:
        if obs.dim() != 3 or obs.size(-1) != 2:
            raise ValueError("obs must be shaped [B, T_obs, 2]")
        B = obs.size(0)
        _, h = self.encoder(obs)
        h = h.squeeze(0)
        prev_disp = torch.zeros(B, 2, device=obs.device, dtype=obs.dtype)
        fut = []
        for _ in range(self.pred_len):
            h = self.decoder_cell(prev_disp, h)
            disp = self.output_proj(h)
            fut.append(disp.unsqueeze(1))
            prev_disp = disp
        return torch.cat(fut, dim=1)


__all__ = ["CoordGRU"]
