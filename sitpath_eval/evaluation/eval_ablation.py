from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List

import matplotlib.pyplot as plt

from ..utils.io import write_csv
from ..utils.viz import plot_bar_with_ci

AblationRunner = Callable[[Dict[str, object]], Dict[str, float]]


def _metric_triplet(prefix: str, metrics: Dict[str, float]) -> tuple[float, float, float]:
    mean = metrics.get(prefix)
    if mean is None:
        mean = metrics.get(f"{prefix}_mean", float("nan"))
    ci_low = metrics.get(f"{prefix}_ci95_low", mean)
    ci_high = metrics.get(f"{prefix}_ci95_high", mean)
    return float(mean), float(ci_low), float(ci_high)


def evaluate_ablation(
    configs: Iterable[Dict[str, object]],
    run_config: AblationRunner,
    *,
    csv_path: str | Path,
    fig_prefix: str | Path,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for cfg in configs:
        metrics = run_config(cfg)
        row = {**cfg, **metrics}
        results.append(row)
    fieldnames = sorted({key for row in results for key in row.keys()})
    write_csv(csv_path, results, fieldnames=fieldnames)

    labels = [str(row.get("name") or row.get("tag") or idx) for idx, row in enumerate(results)]
    ade_means, ade_ci_lows, ade_ci_highs = zip(*[_metric_triplet("ADE", row) for row in results])
    fde_means, fde_ci_lows, fde_ci_highs = zip(*[_metric_triplet("FDE", row) for row in results])

    fig_prefix = Path(fig_prefix)
    fig_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 3))
    plot_bar_with_ci(labels, ade_means, ade_ci_lows, ade_ci_highs)
    plt.ylabel("ADE (m)")
    plt.tight_layout()
    plt.savefig(fig_prefix.with_name(fig_prefix.stem + "_ade.png"))
    plt.close()

    plt.figure(figsize=(6, 3))
    plot_bar_with_ci(labels, fde_means, fde_ci_lows, fde_ci_highs)
    plt.ylabel("FDE (m)")
    plt.tight_layout()
    plt.savefig(fig_prefix.with_name(fig_prefix.stem + "_fde.png"))
    plt.close()

    return results


__all__ = ["evaluate_ablation"]
