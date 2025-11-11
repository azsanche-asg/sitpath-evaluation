#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from glob import glob
from pathlib import Path
from typing import Dict, List

import pandas as pd

from sitpath_eval.evaluation import evaluate_ablation


def run_config(cfg_path: Path, save_dir: Path, split: str) -> Dict[str, float]:
    cfg_str = str(cfg_path)
    result_csv = save_dir / f"{cfg_path.stem}_{split}.csv"
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["python", "scripts/train.py", "--cfg", cfg_str], check=True)
    subprocess.run(["python", "scripts/evaluate.py", "--cfg", cfg_str, "--split", split, "--save_csv", str(result_csv)], check=True)
    df = pd.read_csv(result_csv)
    metrics = {}
    for _, row in df.iterrows():
        metric = row["metric"]
        metrics[metric] = row["mean"]
        metrics[f"{metric}_ci95_low"] = row.get("ci95_low", row["mean"])
        metrics[f"{metric}_ci95_high"] = row.get("ci95_high", row["mean"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation sweep")
    parser.add_argument("--cfgs_glob", required=True, type=str)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--save_csv", type=str, default="tables/ablations.csv")
    parser.add_argument("--fig_prefix", type=str, default="figs/ablations")
    args = parser.parse_args()

    cfg_paths = sorted(Path(p) for p in glob(args.cfgs_glob))
    if not cfg_paths:
        raise ValueError("No configs matched the glob pattern")

    configs = [{"cfg_path": str(path), "name": path.stem} for path in cfg_paths]
    save_dir = Path("tables/ablations_raw")

    def runner(cfg_dict):
        return run_config(Path(cfg_dict["cfg_path"]), save_dir, args.split)

    evaluate_ablation(configs, runner, csv_path=args.save_csv, fig_prefix=args.fig_prefix)


if __name__ == "__main__":
    main()
