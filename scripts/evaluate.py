#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sitpath_eval.config import load_config
from sitpath_eval.data import ETHUCYDataset, NuScenesMiniDataset, SDDMiniDataset
from sitpath_eval.evaluation import evaluate_model
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


def create_model(model_cfg: Dict, dataset_cfg: Dict) -> Tuple[nn.Module, str]:
    obs_len = dataset_cfg.get("obs_len", 8)
    pred_len = dataset_cfg.get("pred_len", 12)
    name = model_cfg.get("name")
    if name == "coord_transformer":
        return CoordTransformer(obs_len=obs_len, pred_len=pred_len), "coordinates"
    if name == "coord_gru":
        return CoordGRU(obs_len=obs_len, pred_len=pred_len), "coordinates"
    raise NotImplementedError("Evaluation script currently supports coordinate models only.")


def build_dataloader(dataset_obj, split: str, batch_size: int = 32) -> DataLoader:
    trajectories, _ = dataset_obj.split(split)
    ds = CoordDataset(trajectories, dataset_obj.obs_len, dataset_obj.pred_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def coord_infer_fn(model: nn.Module, batch: Dict, device: torch.device):
    obs = batch["obs"].to(device)
    fut = batch["fut"].to(device)
    pred = model(obs)
    return {"pred": pred, "target": fut}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SitPath models")
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--save_csv", required=True, type=str)
    parser.add_argument("--seed", type=int, help="Seed directory to load checkpoints from")
    args = parser.parse_args()

    config = load_config(args.cfg)
    dataset_name, dataset_obj = load_dataset(config)
    model, representation = create_model(config.get("model", {}), config.get("dataset", {}))
    if representation != "coordinates":
        raise NotImplementedError("Only coordinate-based evaluation is implemented.")
    training_cfg = config.get("training", {})
    log_dir = Path(training_cfg.get("log_dir", f"outputs/{dataset_name}/{config['model']['name']}"))
    seed = args.seed if args.seed is not None else training_cfg.get("seeds", [0])[0]
    ckpt_path = log_dir / f"seed_{seed}" / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = build_dataloader(dataset_obj, args.split, batch_size=training_cfg.get("batch_size", 64))
    evaluate_model(model, dataloader, device, infer_fn=coord_infer_fn, csv_path=args.save_csv)


if __name__ == "__main__":
    main()
