from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def ci95_bootstrap(values: Iterable[float], n_boot: int = 1000) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    boot_means = []
    for _ in range(n_boot):
        samples = np.random.choice(arr, size=arr.size, replace=True)
        boot_means.append(np.mean(samples))
    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return float(lower), float(upper)


__all__ = ["mean_std", "ci95_bootstrap"]
