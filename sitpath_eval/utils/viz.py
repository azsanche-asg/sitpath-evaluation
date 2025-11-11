from __future__ import annotations

import math
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib import cm, patches


def _get_axes(ax=None):
    return ax if ax is not None else plt.gca()


def plot_ade_vs_fraction(fractions: Sequence[float], ade_values: Sequence[float], *, label: str | None = None, ax=None, **plot_kwargs):
    ax = _get_axes(ax)
    if "marker" not in plot_kwargs:
        plot_kwargs["marker"] = "o"
    ax.plot(fractions, ade_values, label=label, **plot_kwargs)
    ax.set_xlabel("Train fraction")
    ax.set_ylabel("ADE (m)")
    ax.set_title("ADE vs. train data fraction")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_bar_with_ci(labels: Sequence[str], means: Sequence[float], ci_lows: Sequence[float], ci_highs: Sequence[float], *, color: str = "#4c72b0", ax=None):
    ax = _get_axes(ax)
    x = range(len(labels))
    errors = [
        [max(0.0, mean - low) for mean, low in zip(means, ci_lows)],
        [max(0.0, high - mean) for mean, high in zip(means, ci_highs)],
    ]
    bars = ax.bar(x, means, color=color, alpha=0.8)
    ax.errorbar(x, means, yerr=errors, fmt="none", ecolor="black", capsize=4, linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Metric")
    ax.set_title("Model comparison with 95% CI")
    return ax


def plot_token_overlay(
    agent_pos: Sequence[float],
    token_radii: Sequence[float],
    token_angles: Sequence[Sequence[float]],
    *,
    token_labels: Sequence[str] | None = None,
    ax=None,
    cmap_name: str = "viridis",
):
    ax = _get_axes(ax)
    cmap = cm.get_cmap(cmap_name, max(1, len(token_radii)))
    for idx, (radius, angle_range) in enumerate(zip(token_radii, token_angles)):
        start, end = angle_range
        start_deg = math.degrees(start)
        end_deg = math.degrees(end)
        wedge = patches.Wedge(
            center=(agent_pos[0], agent_pos[1]),
            r=radius,
            theta1=start_deg,
            theta2=end_deg,
            width=0.25 * radius,
            facecolor=cmap(idx),
            alpha=0.4,
            edgecolor="none",
        )
        ax.add_patch(wedge)
        if token_labels:
            mid = 0.5 * (start + end)
            text_r = radius * 0.7
            text_x = agent_pos[0] + text_r * math.cos(mid)
            text_y = agent_pos[1] + text_r * math.sin(mid)
            ax.text(text_x, text_y, token_labels[idx], ha="center", va="center", fontsize=9)

    ax.scatter([agent_pos[0]], [agent_pos[1]], c="black", s=35, label="agent")
    max_radius = max(token_radii) if token_radii else 1.0
    buffer = max_radius * 1.1
    ax.set_xlim(agent_pos[0] - buffer, agent_pos[0] + buffer)
    ax.set_ylim(agent_pos[1] - buffer, agent_pos[1] + buffer)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Token overlay")
    ax.legend(loc="upper right")
    return ax


__all__ = [
    "plot_ade_vs_fraction",
    "plot_bar_with_ci",
    "plot_token_overlay",
]
