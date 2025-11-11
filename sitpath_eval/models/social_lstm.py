from __future__ import annotations

import torch
from torch import Tensor, nn


def _build_grid_indices(rel_positions: Tensor, cell_size: float, grid_dim: int) -> Tensor:
    half = grid_dim // 2
    idx = torch.floor(rel_positions / cell_size).long().clamp(min=-half, max=half - 1)
    idx = idx + half
    mask = (idx >= 0) & (idx < grid_dim)
    valid = mask.all(dim=-1)
    flat_idx = idx[..., 0] * grid_dim + idx[..., 1]
    flat_idx[~valid] = -1
    return flat_idx


class SocialPooling(nn.Module):
    def __init__(self, grid_dim: int = 4, cell_size: float = 0.5, embed_dim: int = 32) -> None:
        super().__init__()
        self.grid_dim = grid_dim
        self.cell_size = cell_size
        self.embed = nn.Embedding(grid_dim * grid_dim + 1, embed_dim)

    def forward(self, rel_positions: Tensor) -> Tensor:
        # rel_positions: [B, N, 2]
        B, N, _ = rel_positions.shape
        if N == 0:
            return torch.zeros(B, self.embed.embedding_dim, device=rel_positions.device)
        flat_idx = _build_grid_indices(rel_positions, self.cell_size, self.grid_dim)
        pad_idx = torch.full_like(flat_idx, self.grid_dim * self.grid_dim)
        flat_idx = torch.where(flat_idx >= 0, flat_idx, pad_idx)
        embeddings = self.embed(flat_idx)
        pooled = embeddings.sum(dim=1)
        return pooled


class SocialLSTM(nn.Module):
    def __init__(
        self,
        obs_len: int,
        pred_len: int,
        hidden_size: int = 128,
        grid_dim: int = 4,
        cell_size: float = 0.5,
    ) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.pool = SocialPooling(grid_dim=grid_dim, cell_size=cell_size, embed_dim=hidden_size // 4)
        self.input_embed = nn.Linear(2, hidden_size)
        self.social_proj = nn.Linear(hidden_size // 4, hidden_size)
        self.lstm = nn.LSTMCell(input_size=hidden_size * 2, hidden_size=hidden_size)
        self.output_proj = nn.Linear(hidden_size, 2)

    def forward(self, obs: Tensor, neighbors: Tensor) -> Tensor:
        """obs: [B, T, 2], neighbors: [B, N, T, 2]."""
        if obs.dim() != 3:
            raise ValueError("obs must be [B,T,2]")
        B, T, _ = obs.shape
        if neighbors.dim() != 4:
            raise ValueError("neighbors must be [B,N,T,2]")
        h = torch.zeros(B, self.hidden_size, device=obs.device)
        c = torch.zeros_like(h)
        for t in range(T):
            agent_in = self.input_embed(obs[:, t])
            rel = neighbors[:, :, t] - obs[:, None, t]
            social = self.social_proj(self.pool(rel))
            h, c = self.lstm(torch.cat([agent_in, social], dim=-1), (h, c))
        prev_disp = torch.zeros(B, 2, device=obs.device)
        fut = []
        for _ in range(self.pred_len):
            agent_in = self.input_embed(prev_disp)
            social = torch.zeros_like(agent_in)
            h, c = self.lstm(torch.cat([agent_in, social], dim=-1), (h, c))
            disp = self.output_proj(h)
            fut.append(disp.unsqueeze(1))
            prev_disp = disp
        return torch.cat(fut, dim=1)


__all__ = ["SocialLSTM"]
