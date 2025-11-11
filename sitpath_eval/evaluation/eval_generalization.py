from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ..data.common import TrajectoryDataset
from ..utils.io import write_csv

FoldRunner = Callable[[str, Tuple[List[np.ndarray], List[str]], Tuple[List[np.ndarray], List[str]]], Dict[str, float]]


@dataclass
class SceneFoldResult:
    scene: str
    metrics: Dict[str, float]


def cross_scene_generalization(
    dataset: TrajectoryDataset,
    run_fold: FoldRunner,
    *,
    csv_path: str | Path,
) -> List[SceneFoldResult]:
    folds = dataset.leave_one_scene_out()
    results: List[SceneFoldResult] = []
    for scene, (train_split, val_split) in folds.items():
        metrics = run_fold(scene, train_split, val_split)
        results.append(SceneFoldResult(scene=scene, metrics=metrics))
    metric_names = sorted({name for res in results for name in res.metrics.keys()})
    means = {name: float(np.mean([res.metrics.get(name, np.nan) for res in results])) for name in metric_names}
    rows = []
    for res in results:
        row = {"scene": res.scene}
        for name in metric_names:
            value = res.metrics.get(name, np.nan)
            row[name] = value
            row[f"delta_{name}"] = value - means[name]
        rows.append(row)
    fieldnames = ["scene"] + metric_names + [f"delta_{m}" for m in metric_names]
    write_csv(csv_path, rows, fieldnames=fieldnames)
    return results


__all__ = ["cross_scene_generalization", "SceneFoldResult"]
