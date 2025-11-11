from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from .metrics import ade_per_traj

ForwardFn = Callable[[nn.Module, dict, torch.device], tuple[Tensor, Optional[Tensor]]]
LossFn = Callable[[Tensor, Optional[Tensor], dict], Tensor]


def default_forward_fn(model: nn.Module, batch: dict, device: torch.device) -> tuple[Tensor, Optional[Tensor]]:
    model_input = batch.get("tokens") or batch.get("obs")
    if model_input is None:
        raise ValueError("Batch must contain 'tokens' or 'obs'")
    model_input = model_input.to(device)
    target = batch.get("fut") or batch.get("targets")
    if target is not None:
        target = target.to(device)
    output = model(model_input)
    return output, target


def default_loss_fn(pred: Tensor, target: Optional[Tensor], batch: dict) -> Tensor:
    if target is None:
        raise ValueError("Target tensor required for default loss")
    return torch.nn.functional.mse_loss(pred, target)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    forward_fn: ForwardFn = default_forward_fn,
    loss_fn: LossFn = default_loss_fn,
    grad_clip: Optional[float] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        pred, target = forward_fn(model, batch, device)
        loss = loss_fn(pred, target, batch)
        loss.backward()
        if grad_clip is not None:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(1, total_batches)


def validate(
    model: nn.Module,
    dataloader,
    device: torch.device,
    *,
    forward_fn: ForwardFn = default_forward_fn,
    loss_fn: LossFn = default_loss_fn,
) -> Dict[str, float]:
    model.eval()
    losses = []
    ade_vals = []
    with torch.no_grad():
        for batch in dataloader:
            pred, target = forward_fn(model, batch, device)
            loss = loss_fn(pred, target, batch)
            losses.append(float(loss.detach().cpu()))
            if target is not None:
                ade_vals.append(ade_per_traj(pred, target))
    metrics = {"loss": float(np.mean(losses)) if losses else 0.0}
    if ade_vals:
        ade_concat = np.concatenate([np.atleast_1d(v) for v in ade_vals])
        metrics["ADE"] = float(ade_concat.mean())
    return metrics


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.0
    best_metric: float = float("inf")
    num_bad_epochs: int = 0

    def step(self, metric: float) -> bool:
        if metric + self.min_delta < self.best_metric:
            self.best_metric = metric
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


__all__ = ["train_one_epoch", "validate", "EarlyStopping"]
