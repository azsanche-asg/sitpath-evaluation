from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn


def l2_displacement(pred: Tensor, target: Tensor, reduction: Literal["mean", "sum", "none"] = "mean") -> Tensor:
    """Per-step L2 displacement error (ADE-style)."""
    diff = torch.norm(pred - target, dim=-1)
    if reduction == "mean":
        return diff.mean()
    if reduction == "sum":
        return diff.sum()
    if reduction == "none":
        return diff
    raise ValueError(f"Unsupported reduction {reduction}")


def ade_loss(pred: Tensor, target: Tensor) -> Tensor:
    return l2_displacement(pred, target, reduction="mean")


def fde_loss(pred: Tensor, target: Tensor) -> Tensor:
    return torch.norm(pred[:, -1] - target[:, -1], dim=-1).mean()


def token_cross_entropy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    return nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index)


def gaussian_endpoint_nll(mean: Tensor, logvar: Tensor, target: Tensor, reduction: Literal["mean", "sum", "none"] = "mean") -> Tensor:
    """Assume diagonal covariance parameterized by log-variance."""
    diff = target - mean
    var = torch.exp(logvar).clamp_min(1e-6)
    constant = math.log(2 * math.pi)
    nll = 0.5 * (logvar + (diff ** 2) / var + constant)
    nll = nll.sum(dim=-1)
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll


__all__ = [
    "l2_displacement",
    "ade_loss",
    "fde_loss",
    "token_cross_entropy",
    "gaussian_endpoint_nll",
]
