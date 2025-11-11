from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn

from ..training.metrics import aggregate_metrics
from ..utils.bootstrapping import bootstrap_metric
from ..utils.io import write_csv
from ..utils.timer import Timer

InferFn = Callable[[nn.Module, Dict[str, torch.Tensor], torch.device], Dict[str, torch.Tensor]]


@dataclass
class EvalResult:
    metric: str
    mean: float
    ci95_low: float
    ci95_high: float


def evaluate_model(
    model: nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    *,
    infer_fn: InferFn,
    csv_path: str | Path,
    n_boot: int = 1000,
) -> Dict[str, EvalResult]:
    model.eval()
    metric_arrays: Dict[str, list[np.ndarray]] = {}
    with torch.no_grad(), Timer("eval"):
        for batch in dataloader:
            outputs = infer_fn(model, batch, device)
            target = outputs.get("target") or batch.get("fut")
            metrics = aggregate_metrics(
                outputs.get("pred"),
                target,
                samples=outputs.get("samples"),
                pred_mean=outputs.get("pred_mean"),
                pred_var=outputs.get("pred_var"),
            )
            for name, arr in metrics.items():
                metric_arrays.setdefault(name, []).append(np.asarray(arr))
    rows = []
    summary: Dict[str, EvalResult] = {}
    for name, parts in metric_arrays.items():
        concat = np.concatenate([np.atleast_1d(p) for p in parts])
        stats = bootstrap_metric(concat, n_boot=n_boot)
        rows.append({"metric": name, **stats})
        summary[name] = EvalResult(metric=name, **stats)
    write_csv(csv_path, rows, fieldnames=["metric", "mean", "ci95_low", "ci95_high"])
    return summary


__all__ = ["evaluate_model", "EvalResult"]
