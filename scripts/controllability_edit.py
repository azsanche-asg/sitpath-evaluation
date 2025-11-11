#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sitpath_eval.config import load_config
from sitpath_eval.data import ETHUCYDataset, NuScenesMiniDataset, SDDMiniDataset
from sitpath_eval.evaluation import evaluate_controllability
from sitpath_eval.models import CoordGRU, CoordTransformer

DATASET_BUILDERS = {
    "eth_ucy": ETHUCYDataset,
    "sdd_mini": SDDMiniDataset,
    "nuscenes_mini": NuScenesMiniDataset,
}


class CoordDataset(Dataset):
    def __init__(self, trajectories: List, obs_len: int, pred_len: int) -> None:
        self.obs = [torch.tensor(traj[:obs_len], dtype=torch.float32) for traj in trajectories]
        self.fut = [torch.tensor(traj[obs_len : obs_len + pred_len], dtype=torch.float32) for traj in trajectories]

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"obs": self.obs[idx], "fut": self.fut[idx]}


def load_dataset(cfg: Dict) -> Tuple[str, object]:
    dataset_cfg = cfg.get("dataset", {})
    name = dataset_cfg.get("name")
    if not name:
        raise ValueError("dataset.name missing from config")
    builder = DATASET_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unsupported dataset '{name}'")
    dataset = builder(
        root=Path("data") / name,
        obs_len=dataset_cfg.get("obs_len", 8),
        pred_len=dataset_cfg.get("pred_len", 12),
        fps=dataset_cfg.get("fps", 2.5),
    )
    return name, dataset


def create_model(model_cfg: Dict, dataset_cfg: Dict) -> nn.Module:
    obs_len = dataset_cfg.get("obs_len", 8)
    pred_len = dataset_cfg.get("pred_len", 12)
    name = model_cfg.get("name")
    if name == "coord_transformer":
        return CoordTransformer(obs_len=obs_len, pred_len=pred_len)
    if name == "coord_gru":
        return CoordGRU(obs_len=obs_len, pred_len=pred_len)
    raise NotImplementedError("Controllability script currently supports coordinate models only.")


def build_dataloader(dataset_obj, split: str, batch_size: int = 64) -> DataLoader:
    trajectories, _ = dataset_obj.split(split)
    ds = CoordDataset(trajectories, dataset_obj.obs_len, dataset_obj.pred_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def run_inference(model: nn.Module, dataloader: DataLoader, device: torch.device):
    obs_list, fut_list, pred_list = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            obs = batch["obs"].to(device)
            fut = batch["fut"].to(device)
            pred = model(obs)
            obs_list.append(obs.cpu().numpy())
            fut_list.append(fut.cpu().numpy())
            pred_list.append(pred.cpu().numpy())
    return (
        np.concatenate(obs_list, axis=0),
        np.concatenate(fut_list, axis=0),
        np.concatenate(pred_list, axis=0),
    )


def apply_edit(strategy: str, obs: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    edited = pred.copy()
    heading = obs[:, -1] - obs[:, -2]
    norms = np.linalg.norm(heading, axis=-1, keepdims=True) + 1e-6
    heading_unit = heading / norms
    if strategy == "avoid_front":
        steps = min(k, pred.shape[1])
        for t in range(steps):
            edited[:, t] -= 0.5 * heading_unit
    elif strategy == "keep_right":
        perp = np.stack([-heading_unit[:, 1], heading_unit[:, 0]], axis=-1)
        edited -= 0.3 * perp[:, None, :]
    elif strategy == "tempo_slow":
        diffs = np.diff(edited, axis=1, prepend=obs[:, -1:, :])
        slowed = diffs * 0.8
        edited = np.cumsum(slowed, axis=1) + obs[:, -1:, :]
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    return edited


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controllability edits")
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--edit", choices=["avoid_front", "keep_right", "tempo_slow", "all"], default="all")
    parser.add_argument("--k", type=int, default=5, help="Horizon for avoid-front edits")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--save_csv", required=True, type=str)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = load_config(args.cfg)
    dataset_name, dataset_obj = load_dataset(config)
    model = create_model(config.get("model", {}), config.get("dataset", {}))
    training_cfg = config.get("training", {})
    log_dir = Path(training_cfg.get("log_dir", f"outputs/{dataset_name}/{config['model']['name']}"))
    seed = args.seed if args.seed is not None else training_cfg.get("seeds", [0])[0]
    ckpt_path = log_dir / f"seed_{seed}" / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = build_dataloader(dataset_obj, args.split, batch_size=training_cfg.get("batch_size", 64))
    obs, target, orig_pred = run_inference(model, dataloader, device)

    def runner(strategy: str):
        if args.edit != "all" and strategy != args.edit:
            # Return identity edit for unrelated strategies
            edited = orig_pred
        else:
            edited = apply_edit(strategy, obs, orig_pred, args.k)
        return {"obs": obs, "target": target, "orig_pred": orig_pred, "edited_pred": edited}

    evaluate_controllability(runner, csv_path=args.save_csv, avoid_front_k=args.k)


if __name__ == "__main__":
    main()
