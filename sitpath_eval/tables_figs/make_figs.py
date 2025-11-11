from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.viz import plot_ade_vs_fraction, plot_token_overlay

FIG_ROOT = Path("figs")
SOURCE_ROOT = Path("figs_sources")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def make_data_efficiency_fig(source: Path, out_path: Path) -> None:
    if not source.exists():
        return
    df = pd.read_csv(source)
    plt.figure(figsize=(4, 3))
    plot_ade_vs_fraction(df["fraction"], df["ADE"], label="ADE")
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()


def make_zipf_fig(source: Path, out_path: Path) -> None:
    if not source.exists():
        return
    with source.open("r", encoding="utf-8") as fh:
        freqs = json.load(fh)
    counts = np.array(sorted(freqs.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(counts) + 1)
    plt.figure(figsize=(4, 3))
    plt.loglog(ranks, counts)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Token Zipf curve")
    _ensure_dir(out_path)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def make_mi_bar(source: Path, out_path: Path) -> None:
    if not source.exists():
        return
    df = pd.read_csv(source)
    plt.figure(figsize=(4, 3))
    plt.bar(df["model"], df["MI"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Mutual information (bits)")
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()


def make_linear_probe_fig(source: Path, out_path: Path) -> None:
    if not source.exists():
        return
    df = pd.read_csv(source)
    plt.figure(figsize=(4, 3))
    plt.bar(df["dataset"], df["accuracy"], color="#4c72b0")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()


def make_overlay_fig(source: Path, out_path: Path, token_labels: Optional[list[str]] = None) -> None:
    if not source.exists():
        return
    data = np.load(source)
    obs = data["obs"]
    pred = data["pred"]
    target = data["target"]
    plt.figure(figsize=(4, 4))
    idx = 0
    plt.plot(obs[idx, :, 0], obs[idx, :, 1], "ko-", label="obs")
    plt.plot(target[idx, :, 0], target[idx, :, 1], "g--", label="gt")
    plt.plot(pred[idx, :, 0], pred[idx, :, 1], "r-", label="pred")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()

    if "agent_pos" in data and "token_radii" in data and "token_angles" in data:
        overlay_path = out_path.with_name(out_path.stem + "_tokens.png")
        plt.figure(figsize=(4, 4))
        plot_token_overlay(data["agent_pos"], data["token_radii"], data["token_angles"], token_labels=token_labels)
        plt.tight_layout()
        plt.savefig(overlay_path)
        plt.close()


def make_controllability_fig(source: Path, out_path: Path) -> None:
    if not source.exists():
        return
    df = pd.read_csv(source)
    plt.figure(figsize=(4, 3))
    plt.bar(df["strategy"], df["constraint_rate"], label="constraint", alpha=0.7)
    plt.plot(df["strategy"], df["goal_retention"], "o-", label="goal deviation")
    plt.ylabel("Rate / meters")
    plt.legend()
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path)
    plt.close()


def build_all_figs(root: Path = SOURCE_ROOT, out_dir: Path = FIG_ROOT) -> None:
    make_data_efficiency_fig(root / "data_efficiency.csv", out_dir / "ade_vs_fraction.png")
    make_zipf_fig(root / "token_freq.json", out_dir / "zipf.png")
    make_mi_bar(root / "mutual_information.csv", out_dir / "mi_bar.png")
    make_linear_probe_fig(root / "linear_probe.csv", out_dir / "linear_probe.png")
    make_overlay_fig(root / "qualitative.npz", out_dir / "qualitative.png")
    make_controllability_fig(root / "controllability.csv", out_dir / "controllability.png")


if __name__ == "__main__":
    build_all_figs()
