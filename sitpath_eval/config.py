from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": {
        "name": None,
        "obs_len": 8,
        "pred_len": 12,
        "fps": 2.5,
        "cache_dir": "cache/default/",
    },
    "model": {
        "name": "sitpath_transformer",
        "type": "transformer",
        "matched_capacity": True,
    },
    "training": {
        "seeds": [0],
        "batch_size": 32,
        "epochs": 20,
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-3,
            "weight_decay": 0.0,
        },
        "early_stop_metric": "ADE",
        "log_dir": "outputs/default/",
    },
    "eval": {
        "sampling": {"K": 20},
        "metrics": ["ADE", "FDE"],
    },
    "tokenizer": {
        "M": 16,
        "R": 5.0,
        "B": 4,
        "K_tau": 3,
        "collapse": True,
        "stall_eps": 0.05,
    },
}


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _default_config() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_CONFIG)


def load_config(path: Optional[str | Path], overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    config = _default_config()
    if path:
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
            if not isinstance(data, Mapping):
                raise ValueError(f"Config at {cfg_path} must be a mapping")
            _deep_update(config, data)
    if overrides:
        _deep_update(config, overrides)
    return config


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--dataset-name", type=str, help="Override dataset name")
    parser.add_argument("--obs-len", type=int, help="Observation horizon override")
    parser.add_argument("--pred-len", type=int, help="Prediction horizon override")
    parser.add_argument("--fps", type=float, help="Frames per second override")
    parser.add_argument("--cache-dir", type=str, help="Dataset cache directory override")
    parser.add_argument("--batch-size", type=int, help="Training batch size override")
    parser.add_argument("--epochs", type=int, help="Training epochs override")
    parser.add_argument("--lr", type=float, help="Learning rate override")
    parser.add_argument("--log-dir", type=str, help="Training log/output directory override")
    parser.add_argument("--seed", type=int, help="Single seed override (replaces list)")
    parser.add_argument("--seeds", type=int, nargs="+", help="Multiple seeds override")
    parser.add_argument("--token-M", type=int, dest="token_M", help="Tokenizer M override")
    parser.add_argument("--token-R", type=float, dest="token_R", help="Tokenizer R override")
    parser.add_argument("--token-B", type=int, dest="token_B", help="Tokenizer B override")
    parser.add_argument("--token-Ktau", type=int, dest="token_Ktau", help="Tokenizer K_tau override")
    parser.add_argument(
        "--no-token-collapse",
        action="store_false",
        dest="token_collapse",
        help="Disable tokenizer collapse",
    )
    parser.add_argument(
        "--token-collapse",
        action="store_true",
        dest="token_collapse",
        help="Force tokenizer collapse",
    )
    parser.set_defaults(token_collapse=None)
    parser.add_argument("--stall-eps", type=float, help="Tokenizer stall epsilon override")


def _apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.dataset_name:
        config.setdefault("dataset", {})["name"] = args.dataset_name
    dataset = config.setdefault("dataset", {})
    training = config.setdefault("training", {})
    tokenizer = config.setdefault("tokenizer", {})

    if args.obs_len is not None:
        dataset["obs_len"] = args.obs_len
    if args.pred_len is not None:
        dataset["pred_len"] = args.pred_len
    if args.fps is not None:
        dataset["fps"] = args.fps
    if args.cache_dir:
        dataset["cache_dir"] = args.cache_dir

    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.lr is not None:
        optimizer = training.setdefault("optimizer", {})
        optimizer["lr"] = args.lr
    if args.log_dir:
        training["log_dir"] = args.log_dir
    if args.seeds:
        training["seeds"] = list(args.seeds)
    elif args.seed is not None:
        training["seeds"] = [args.seed]

    if args.token_M is not None:
        tokenizer["M"] = args.token_M
    if args.token_R is not None:
        tokenizer["R"] = args.token_R
    if args.token_B is not None:
        tokenizer["B"] = args.token_B
    if args.token_Ktau is not None:
        tokenizer["K_tau"] = args.token_Ktau
    if args.token_collapse is not None:
        tokenizer["collapse"] = bool(args.token_collapse)
    if args.stall_eps is not None:
        tokenizer["stall_eps"] = args.stall_eps

    return config


def parse_config_with_cli(argv: Optional[Iterable[str]] = None, description: Optional[str] = None) -> tuple[Dict[str, Any], argparse.Namespace]:
    parser = argparse.ArgumentParser(description=description)
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    config = load_config(args.cfg)
    config = _apply_cli_overrides(config, args)
    return config, args


__all__ = [
    "DEFAULT_CONFIG",
    "load_config",
    "add_config_arguments",
    "parse_config_with_cli",
]
