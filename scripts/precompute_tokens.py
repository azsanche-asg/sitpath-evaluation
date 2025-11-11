#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from sitpath_eval.config import load_config
from sitpath_eval.data import ETHUCYDataset, NuScenesMiniDataset, SDDMiniDataset
from sitpath_eval.tokenizer import SitPathTokenizer
from sitpath_eval.utils.io import safe_mkdirs, save_pickle

DATASET_BUILDERS = {
    "eth_ucy": ETHUCYDataset,
    "sdd_mini": SDDMiniDataset,
    "nuscenes_mini": NuScenesMiniDataset,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute SitPath tokens for a dataset")
    parser.add_argument("--cfg", type=str, help="Optional YAML config to pull tokenizer settings from")
    parser.add_argument("--dataset", type=str, help="Dataset name if cfg not provided", default=None)
    parser.add_argument("--max-neighbors", type=int, default=5, help="Number of neighbors per trajectory window")
    parser.add_argument("--output", type=str, help="Override cache directory")
    return parser.parse_args()


def build_dataset(cfg: Dict) -> Tuple[str, object]:
    dataset_cfg = cfg.get("dataset", {})
    name = dataset_cfg.get("name")
    if not name:
        raise ValueError("Dataset name must be provided via --dataset or cfg")
    builder = DATASET_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unsupported dataset '{name}'")
    root = Path("data") / name
    kwargs = {
        "root": root,
        "obs_len": dataset_cfg.get("obs_len", 8),
        "pred_len": dataset_cfg.get("pred_len", 12),
        "fps": dataset_cfg.get("fps", 2.5),
    }
    return name, builder(**kwargs)


def load_tokenizer(cfg: Dict) -> SitPathTokenizer:
    tokenizer_cfg = cfg.get("tokenizer", {})
    return SitPathTokenizer(
        M=tokenizer_cfg.get("M", 16),
        R=tokenizer_cfg.get("R", 5.0),
        B=tokenizer_cfg.get("B", 4),
        K_tau=tokenizer_cfg.get("K_tau", 3),
        collapse=tokenizer_cfg.get("collapse", True),
        stall_eps=tokenizer_cfg.get("stall_eps", 0.05),
    )


def get_records(dataset, tokenizer: SitPathTokenizer, splits: List[str], max_neighbors: int) -> List[Dict]:
    records: List[Dict] = []
    for split in splits:
        trajectories, scene_ids = dataset.split(split)
        for idx, (traj, scene_id) in enumerate(zip(trajectories, scene_ids)):
            neighbors = []
            for other_idx, other in enumerate(trajectories):
                if other_idx == idx:
                    continue
                neighbors.append(other)
                if len(neighbors) >= max_neighbors:
                    break
            token_result = tokenizer.tokenize(traj, neighbors)
            obs_len = dataset.obs_len
            pred_len = dataset.pred_len
            obs_tokens = token_result.ids[: max(0, obs_len - 1)]
            fut_tokens = token_result.ids[max(0, obs_len - 1) : max(0, obs_len - 1) + pred_len]
            records.append(
                {
                    "split": split,
                    "scene": scene_id,
                    "obs_tokens": obs_tokens,
                    "fut_tokens": fut_tokens,
                    "obs": traj[:obs_len],
                    "fut": traj[obs_len : obs_len + pred_len],
                }
            )
    return records


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg) if args.cfg else load_config(None)
    if args.dataset:
        cfg.setdefault("dataset", {})["name"] = args.dataset
    dataset_name, dataset = build_dataset(cfg)
    tokenizer = load_tokenizer(cfg)
    cache_dir = Path(args.output) if args.output else Path(cfg.get("dataset", {}).get("cache_dir", f"cache/{dataset_name}"))
    safe_mkdirs(cache_dir)
    records = get_records(dataset, tokenizer, ["train", "val", "test"], args.max_neighbors)
    save_path = cache_dir / "tokens.pkl"
    save_pickle(save_path, records)
    tokenizer.save_vocab(cache_dir / "vocab.json")
    print(f"Saved {len(records)} tokenized trajectories to {save_path}")


if __name__ == "__main__":
    main()
