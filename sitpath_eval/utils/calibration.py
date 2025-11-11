from __future__ import annotations

import math
from typing import Iterable

import numpy as np

_TWO_PI = 2.0 * math.pi


def _prepare_arrays(pred_mean: Iterable[Iterable[float]], pred_var: Iterable[Iterable[float]] | Iterable[float], target: Iterable[Iterable[float]]):
    mean = np.asarray(pred_mean, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    var = np.asarray(pred_var, dtype=float)
    if mean.shape != target_arr.shape:
        raise ValueError("pred_mean and target must have matching shapes")
    if var.ndim == 1:
        var = np.repeat(var[:, None], mean.shape[1], axis=1)
    if var.shape != mean.shape:
        raise ValueError("pred_var must match pred_mean after broadcasting")
    return mean, var, target_arr


def gaussian_endpoint_nll(
    pred_mean: Iterable[Iterable[float]],
    pred_var: Iterable[Iterable[float]] | Iterable[float],
    target: Iterable[Iterable[float]],
    *,
    reduce: str = "mean",
) -> float | np.ndarray:
    mean, var, target_arr = _prepare_arrays(pred_mean, pred_var, target)
    var = np.clip(var, 1e-6, None)
    diff = target_arr - mean
    maha = ((diff ** 2) / var).sum(axis=1)
    log_det = np.log(var).sum(axis=1)
    dim = mean.shape[1]
    nll = 0.5 * (log_det + maha + dim * math.log(_TWO_PI))

    if reduce == "mean":
        return float(np.mean(nll))
    if reduce == "sum":
        return float(np.sum(nll))
    if reduce == "none":
        return nll
    raise ValueError(f"Unsupported reduce='{reduce}'")


def endpoint_ece(
    pred_mean: Iterable[Iterable[float]],
    pred_var: Iterable[Iterable[float]] | Iterable[float],
    target: Iterable[Iterable[float]],
    *,
    threshold: float = 2.0,
    n_bins: int = 10,
) -> float:
    mean, var, target_arr = _prepare_arrays(pred_mean, pred_var, target)
    sigma = np.sqrt(np.clip(np.mean(var, axis=1), 1e-9, None))
    conf = 1.0 - np.exp(-(threshold ** 2) / (2.0 * sigma ** 2))
    errors = np.linalg.norm(target_arr - mean, axis=1)
    hits = (errors <= threshold).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = conf.size
    ece = 0.0
    for idx in range(n_bins):
        start, end = bin_edges[idx], bin_edges[idx + 1]
        if idx < n_bins - 1:
            mask = (conf >= start) & (conf < end)
        else:
            mask = (conf >= start) & (conf <= end)
        if not np.any(mask):
            continue
        weight = mask.sum() / total
        acc = hits[mask].mean()
        avg_conf = conf[mask].mean()
        ece += weight * abs(acc - avg_conf)
    return float(ece)


__all__ = ["gaussian_endpoint_nll", "endpoint_ece"]
