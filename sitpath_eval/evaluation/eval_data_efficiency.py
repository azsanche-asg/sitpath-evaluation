from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Sequence

import matplotlib.pyplot as plt

from ..utils.io import write_csv
from ..utils.viz import plot_ade_vs_fraction

FractionRunner = Callable[[float], Dict[str, float]]


def evaluate_data_efficiency(
    fractions: Sequence[float],
    run_fraction: FractionRunner,
    *,
    csv_path: str | Path,
    fig_path: str | Path,
    metric: str = "ADE",
) -> Dict[float, Dict[str, float]]:
    results: Dict[float, Dict[str, float]] = {}
    rows = []
    ade_values = []
    for frac in fractions:
        metrics = run_fraction(frac)
        results[frac] = metrics
        rows.append({"fraction": frac, **metrics})
        ade_values.append(metrics.get(metric, float("nan")))
    if rows:
        extra_keys = sorted(set(rows[0].keys()) - {"fraction"})
        fieldnames = ["fraction", *extra_keys]
    else:
        fieldnames = ["fraction"]
    write_csv(csv_path, rows, fieldnames=fieldnames)

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    plot_ade_vs_fraction(fractions, ade_values, label=metric)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return results


__all__ = ["evaluate_data_efficiency"]
