from __future__ import annotations

import torch
from torch import Tensor, nn


class RasterEncoder(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.net(x)
        return feat.flatten(1)


class RasterGRU(nn.Module):
    def __init__(self, obs_len: int, pred_len: int, in_channels: int = 3, hidden_size: int = 128) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.encoder = RasterEncoder(in_channels)
        self.context_proj = nn.Linear(64, hidden_size)
        self.decoder_cell = nn.GRUCell(input_size=2, hidden_size=hidden_size)
        self.output_proj = nn.Linear(hidden_size, 2)

    def forward(self, raster: Tensor, last_pos: Tensor) -> Tensor:
        """Decode future displacements conditioned on raster context and last position."""
        if raster.dim() != 4:
            raise ValueError("raster must be [B, C, H, W]")
        context = self.context_proj(self.encoder(raster))
        h = torch.tanh(context)
        prev_disp = torch.zeros(raster.size(0), 2, device=raster.device, dtype=raster.dtype)
        fut = []
        for _ in range(self.pred_len):
            h = self.decoder_cell(prev_disp, h)
            disp = self.output_proj(h)
            fut.append(disp.unsqueeze(1))
            prev_disp = disp
        preds = torch.cumsum(torch.cat(fut, dim=1), dim=1) + last_pos.unsqueeze(1)
        return preds


__all__ = ["RasterGRU"]
