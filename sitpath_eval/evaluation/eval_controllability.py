from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np

from ..utils.io import write_csv

StrategyRunner = Callable[[str], Dict[str, np.ndarray]]


@dataclass
class StrategySpec:
    name: str
    constraint_fn: Callable[[np.ndarray, np.ndarray, Dict[str, float]], np.ndarray]
    params: Dict[str, float]


def _ade_np(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=-1).mean(axis=-1)


def _heading(obs: np.ndarray) -> np.ndarray:
    vec = obs[:, -1] - obs[:, -2]
    norms = np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-6
    return vec / norms


def _constraint_avoid_front(obs: np.ndarray, pred: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    k = int(params.get("k", 5))
    heading = _heading(obs)
    ref = obs[:, -1][:, None, :]
    steps = pred[:, :k] - ref
    dots = np.einsum("bij,bi->bj", steps, heading)
    return (dots <= 0).all(axis=1)


def _constraint_keep_right(obs: np.ndarray, pred: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    heading = _heading(obs)
    ref = obs[:, -1][:, None, :]
    steps = pred - ref
    cross = heading[:, 0:1] * steps[:, :, 1] - heading[:, 1:2] * steps[:, :, 0]
    return (cross <= 0).all(axis=1)


def _constraint_tempo_slow(obs: np.ndarray, pred: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    ratio = params.get("ratio", 0.8)
    speeds = np.linalg.norm(np.diff(pred, axis=1, prepend=obs[:, -1:, :]), axis=-1)
    avg_speed = speeds.mean(axis=1)
    baseline = params.get("baseline_speeds")
    if baseline is None:
        return np.ones(len(pred), dtype=bool)
    return avg_speed <= baseline * ratio


def evaluate_controllability(
    runner: StrategyRunner,
    *,
    csv_path: str | Path,
    avoid_front_k: int = 5,
) -> Dict[str, Dict[str, float]]:
    strategies = [
        StrategySpec("avoid_front", _constraint_avoid_front, {"k": avoid_front_k}),
        StrategySpec("keep_right", _constraint_keep_right, {}),
        StrategySpec("tempo_slow", _constraint_tempo_slow, {"ratio": 0.8}),
    ]
    rows = []
    summary: Dict[str, Dict[str, float]] = {}
    for spec in strategies:
        outputs = runner(spec.name)
        obs = outputs["obs"]
        target = outputs["target"]
        orig_pred = outputs["orig_pred"]
        edited_pred = outputs["edited_pred"]
        params = dict(spec.params)
        if spec.name == "tempo_slow":
            baseline_speeds = np.linalg.norm(
                np.diff(orig_pred, axis=1, prepend=obs[:, -1:, :]),
                axis=-1,
            ).mean(axis=1)
            params["baseline_speeds"] = baseline_speeds
        constraint_mask = spec.constraint_fn(obs, edited_pred, params)
        goal_dev = np.linalg.norm(edited_pred[:, -1] - orig_pred[:, -1], axis=-1)
        delta_ade = _ade_np(edited_pred, target) - _ade_np(orig_pred, target)
        row = {
            "strategy": spec.name,
            "constraint_rate": float(constraint_mask.mean()),
            "goal_retention": float(goal_dev.mean()),
            "delta_ADE": float(delta_ade.mean()),
        }
        rows.append(row)
        summary[spec.name] = row
    write_csv(csv_path, rows, fieldnames=list(rows[0].keys()) if rows else ["strategy"])
    return summary


__all__ = ["evaluate_controllability"]
