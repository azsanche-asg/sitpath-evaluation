from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from ..utils.calibration import endpoint_ece, gaussian_endpoint_nll

try:
    import torch
except ImportError:
    torch = None  # type: ignore


def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    if torch is not None and isinstance(arr, torch.Tensor):  # type: ignore
        return arr.detach().cpu().numpy()
    return np.asarray(arr)

def ade_per_traj(pred, target) -> np.ndarray:
    pred = _to_numpy(pred)
    target = _to_numpy(target)
    dists = np.linalg.norm(pred - target, axis=-1)
    return dists.mean(axis=-1)


def fde_per_traj(pred, target) -> np.ndarray:
    pred = _to_numpy(pred)
    target = _to_numpy(target)
    dists = np.linalg.norm(pred[:, -1] - target[:, -1], axis=-1)
    return dists


def minade_k_per_traj(samples, target) -> np.ndarray:
    samples = _to_numpy(samples)  # [K,B,T,2]
    target = _to_numpy(target)
    dists = np.linalg.norm(samples - target[None, ...], axis=-1).mean(axis=-1)
    return dists.min(axis=0)


def mr2m_per_traj(samples, target, threshold: float = 2.0) -> np.ndarray:
    samples = _to_numpy(samples)
    target = _to_numpy(target)
    final_dists = np.linalg.norm(samples[:, :, -1] - target[None, :, -1], axis=-1)
    best = final_dists.min(axis=0)
    return (best > threshold).astype(np.float32)


def nll_per_traj(pred_mean, pred_var, target) -> np.ndarray:
    pred_mean = _to_numpy(pred_mean)
    pred_var = _to_numpy(pred_var)
    target = _to_numpy(target)
    nll = gaussian_endpoint_nll(pred_mean, pred_var, target, reduce="none")
    return np.asarray(nll)


def ece_value(pred_mean, pred_var, target, threshold: float = 2.0, n_bins: int = 10) -> float:
    pred_mean = _to_numpy(pred_mean)
    pred_var = _to_numpy(pred_var)
    target = _to_numpy(target)
    return endpoint_ece(pred_mean, pred_var, target, threshold=threshold, n_bins=n_bins)


def diversity_per_traj(samples) -> np.ndarray:
    samples = _to_numpy(samples)
    endpoints = samples[:, :, -1]  # [K,B,2]
    K, B, _ = endpoints.shape
    div = np.zeros(B)
    if K < 2:
        return div
    for b in range(B):
        pts = endpoints[:, b]
        dists = np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=-1)
        tri = np.triu_indices(K, k=1)
        div[b] = dists[tri].mean()
    return div


def aggregate_metrics(pred, target, *, samples=None, pred_mean=None, pred_var=None, threshold: float = 2.0) -> Dict[str, np.ndarray | float]:
    metrics: Dict[str, np.ndarray | float] = {}
    if pred is not None:
        metrics["ADE"] = ade_per_traj(pred, target)
        metrics["FDE"] = fde_per_traj(pred, target)
    if samples is not None:
        metrics["minADE20"] = minade_k_per_traj(samples, target)
        metrics["MR2m"] = mr2m_per_traj(samples, target, threshold=threshold)
        metrics["diversity"] = diversity_per_traj(samples)
    if pred_mean is not None and pred_var is not None:
        metrics["NLL"] = nll_per_traj(pred_mean, pred_var, target)
        metrics["ECE"] = np.array([ece_value(pred_mean, pred_var, target, threshold=threshold)])
    return metrics


__all__ = [
    "ade_per_traj",
    "fde_per_traj",
    "minade_k_per_traj",
    "mr2m_per_traj",
    "nll_per_traj",
    "ece_value",
    "diversity_per_traj",
    "aggregate_metrics",
]
