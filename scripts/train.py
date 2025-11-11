#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sitpath_eval.config import parse_config_with_cli
from sitpath_eval.data import ETHUCYDataset, NuScenesMiniDataset, SDDMiniDataset
from sitpath_eval.models import CoordGRU, CoordTransformer, SitPathGRU, SitPathTransformer
from sitpath_eval.training import EarlyStopping, token_cross_entropy, train_one_epoch, validate
from sitpath_eval.training.schedulers import cosine_with_warmup
from sitpath_eval.utils.io import load_pickle, safe_mkdirs, save_json
from sitpath_eval.utils.seed import set_all_seeds

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


class TokenDataset(Dataset):
    def __init__(self, records: List[Dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        return {
            "tokens": torch.tensor(rec["obs_tokens"], dtype=torch.long),
            "targets": torch.tensor(rec["fut_tokens"], dtype=torch.long),
            "fut": torch.tensor(rec["fut"], dtype=torch.float32),
        }


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


def load_token_records(cache_dir: Path, split: str) -> List[Dict]:
    tokens_path = cache_dir / "tokens.pkl"
    if not tokens_path.exists():
        raise FileNotFoundError(f"Missing token cache at {tokens_path}. Run scripts/precompute_tokens.py first.")
    records = load_pickle(tokens_path)
    return [rec for rec in records if rec.get("split") == split]


def create_model(model_cfg: Dict, dataset_cfg: Dict, cache_dir: Path) -> Tuple[nn.Module, str, str]:
    obs_len = dataset_cfg.get("obs_len", 8)
    pred_len = dataset_cfg.get("pred_len", 12)
    name = model_cfg.get("name")
    if name == "coord_transformer":
        return CoordTransformer(obs_len=obs_len, pred_len=pred_len), "coordinates", "coord"
    if name == "coord_gru":
        return CoordGRU(obs_len=obs_len, pred_len=pred_len), "coordinates", "coord"
    if name == "sitpath_transformer":
        vocab = json.load((cache_dir / "vocab.json").open("r", encoding="utf-8"))
        return SitPathTransformer(vocab_size=len(vocab), obs_len=obs_len - 1, pred_len=pred_len), "sitpath_tokens", "token"
    if name == "sitpath_gru":
        vocab = json.load((cache_dir / "vocab.json").open("r", encoding="utf-8"))
        mode = model_cfg.get("output", "token")
        return SitPathGRU(vocab_size=len(vocab), obs_len=obs_len - 1, pred_len=pred_len, mode=mode), "sitpath_tokens", ("token" if mode == "token" else "coord")
    raise ValueError(f"Unsupported model '{name}'")


def build_dataloader(dataset_obj, split: str, batch_size: int, representation: str, cache_dir: Path) -> DataLoader:
    if representation == "coordinates":
        trajectories, _ = dataset_obj.split(split)
        ds = CoordDataset(trajectories, dataset_obj.obs_len, dataset_obj.pred_len)
    elif representation == "sitpath_tokens":
        records = load_token_records(cache_dir, split)
        ds = TokenDataset(records)
    else:
        raise NotImplementedError(f"Representation '{representation}' not implemented")
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))


def coord_forward_fn(model: nn.Module, batch: Dict, device: torch.device):
    obs = batch["obs"].to(device)
    fut = batch["fut"].to(device)
    pred = model(obs)
    return pred, fut


def coord_loss_fn(pred, target, batch):
    return nn.functional.mse_loss(pred, target)


def token_forward_fn(model: nn.Module, batch: Dict, device: torch.device):
    tokens = batch["tokens"].to(device)
    logits = model(tokens)
    return logits, None


def token_loss_fn(pred, target, batch):
    return token_cross_entropy(pred, batch["targets"].to(pred.device))


def main() -> None:
    config, _ = parse_config_with_cli(description="Train SitPath models")
    dataset_name, dataset_obj = load_dataset(config)
    training_cfg = config.get("training", {})
    batch_size = training_cfg.get("batch_size", 64)
    epochs = training_cfg.get("epochs", 20)
    log_dir = Path(training_cfg.get("log_dir", f"outputs/{dataset_name}/{config['model']['name']}"))
    cache_dir = Path(config.get("dataset", {}).get("cache_dir", f"cache/{dataset_name}"))

    model_cfg = config.get("model", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = training_cfg.get("seeds", [0])
    optimizer_cfg = training_cfg.get("optimizer", {"lr": 1e-3, "weight_decay": 0.0})

    summary = {}

    for seed in seeds:
        set_all_seeds(seed)
        run_dir = log_dir / f"seed_{seed}"
        safe_mkdirs(run_dir)
        model, representation, target_type = create_model(model_cfg, config.get("dataset", {}), cache_dir)
        model.to(device)
        train_loader = build_dataloader(dataset_obj, "train", batch_size, representation, cache_dir)
        val_loader = build_dataloader(dataset_obj, "val", batch_size, representation, cache_dir)
        optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_cfg.get("lr", 1e-3), weight_decay=optimizer_cfg.get("weight_decay", 0.0))
        scheduler_cfg = training_cfg.get("scheduler")
        scheduler = None
        if scheduler_cfg and scheduler_cfg.get("type") == "cosine":
            total_steps = epochs * max(1, len(train_loader))
            scheduler = cosine_with_warmup(optimizer, total_steps, scheduler_cfg.get("warmup_steps", 0))
        forward_fn = coord_forward_fn if representation == "coordinates" else token_forward_fn
        loss_fn = coord_loss_fn if target_type == "coord" else token_loss_fn
        early_stop = EarlyStopping(patience=training_cfg.get("early_stop_patience", 5))
        best_metric = float("inf")
        history = []
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                forward_fn=forward_fn,
                loss_fn=loss_fn,
                scheduler=scheduler,
            )
            val_metrics = validate(model, val_loader, device, forward_fn=forward_fn, loss_fn=loss_fn)
            val_loss = val_metrics.get("loss", 0.0)
            val_ade = val_metrics.get("ADE", val_loss)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_ADE": val_ade})
            if val_ade < best_metric:
                best_metric = val_ade
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                }, run_dir / "model.pt")
            if early_stop.step(val_ade):
                break
        save_json(run_dir / "metrics.json", history)
        summary[str(seed)] = history

    save_json(log_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
