from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from .stats import ci95_bootstrap, mean_std


def bootstrap_metric(values: Iterable[float], n_boot: int = 1000) -> Mapping[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}

    mean, _ = mean_std(arr)
    ci_low, ci_high = ci95_bootstrap(arr, n_boot=n_boot)
    return {
        "mean": mean,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


__all__ = ["bootstrap_metric"]
