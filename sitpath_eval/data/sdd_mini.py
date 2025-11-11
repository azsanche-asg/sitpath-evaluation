from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .common import TrajectoryDataset, sliding_windows, to_float32

SPLITS = ("train", "val", "test")


def _discover_scene_files(root: Path) -> Tuple[Dict[str, Sequence[str]], Dict[str, Path]]:
    split_map: Dict[str, List[str]] = {split: [] for split in SPLITS}
    scene_files: Dict[str, Path] = {}
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for scene_dir in sorted(split_dir.iterdir()):
            if scene_dir.is_dir():
                csv_path = scene_dir / "trajectories.csv"
                if not csv_path.exists():
                    continue
                scene_name = f"{split}/{scene_dir.name}"
                split_map[split].append(scene_name)
                scene_files[scene_name] = csv_path
            elif scene_dir.suffix == ".csv":
                scene_name = f"{split}/{scene_dir.stem}"
                split_map[split].append(scene_name)
                scene_files[scene_name] = scene_dir
    return split_map, scene_files


class SDDMiniDataset(TrajectoryDataset):
    def __init__(
        self,
        root: str | Path = "data/sdd_mini",
        obs_len: int = 8,
        pred_len: int = 12,
        fps: float = 2.5,
        source_fps: float = 5.0,
        normalize: bool = False,
    ) -> None:
        root_path = Path(root)
        split_map, scene_files = _discover_scene_files(root_path)
        valid_split_map = {k: tuple(v) for k, v in split_map.items() if v}
        if not valid_split_map:
            raise FileNotFoundError(f"No scenes discovered under {root_path}")
        self.scene_files = scene_files
        self.source_fps = source_fps
        super().__init__(
            root=root_path,
            obs_len=obs_len,
            pred_len=pred_len,
            fps=fps,
            normalize=normalize,
            split_map=valid_split_map,
        )

    def _load_all_scenes(self) -> Dict[str, Tuple[List[np.ndarray], List[str]]]:
        cache: Dict[str, Tuple[List[np.ndarray], List[str]]] = {}
        for scene_name, csv_path in self.scene_files.items():
            trajectories, ids = self._load_scene_csv(scene_name, csv_path)
            cache[scene_name] = (trajectories, ids)
        return cache

    def _load_scene_csv(self, scene_name: str, csv_path: Path) -> Tuple[List[np.ndarray], List[str]]:
        df = pd.read_csv(csv_path)
        group_col = self._infer_group_col(df)
        order_col = self._infer_order_col(df)
        coords_cols = [col for col in ("x", "y", "px", "py", "center_x", "center_y") if col in df.columns]
        if len(coords_cols) < 2:
            raise ValueError(f"CSV {csv_path} missing XY columns")
        x_col, y_col = coords_cols[:2]
        trajectories: List[np.ndarray] = []
        scene_ids: List[str] = []
        stride = max(1, int(round(self.source_fps / self.fps)))
        for track_id, track_df in df.groupby(group_col):
            track_df = track_df.sort_values(order_col)
            coords = track_df[[x_col, y_col]].to_numpy(dtype=float)
            coords = coords[::stride]
            if len(coords) < self.total_len:
                continue
            for idx, window in enumerate(sliding_windows(coords, self.total_len)):
                trajectories.append(to_float32(window))
                scene_ids.append(f"{scene_name}:{track_id}:{idx}")
        return trajectories, scene_ids

    @staticmethod
    def _infer_group_col(df: pd.DataFrame) -> str:
        for col in ("track_id", "agent_id", "object_id", "instance_id"):
            if col in df.columns:
                return col
        raise ValueError("Unable to infer track id column")

    @staticmethod
    def _infer_order_col(df: pd.DataFrame) -> str:
        for col in ("frame_id", "timestamp", "frame", "time"):
            if col in df.columns:
                return col
        raise ValueError("Unable to infer temporal ordering column")


__all__ = ["SDDMiniDataset"]
