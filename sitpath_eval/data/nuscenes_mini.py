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
        for csv_file in sorted(split_dir.glob("*.csv")):
            scene_name = f"{split}/{csv_file.stem}"
            split_map[split].append(scene_name)
            scene_files[scene_name] = csv_file
        for scene_dir in sorted(split_dir.iterdir()):
            if scene_dir.is_dir():
                csv_path = scene_dir / "trajectories.csv"
                if csv_path.exists():
                    scene_name = f"{split}/{scene_dir.name}"
                    split_map[split].append(scene_name)
                    scene_files[scene_name] = csv_path
    return split_map, scene_files


class NuScenesMiniDataset(TrajectoryDataset):
    def __init__(
        self,
        root: str | Path = "data/nuscenes_mini",
        obs_len: int = 8,
        pred_len: int = 12,
        fps: float = 2.5,
        source_fps: float = 2.0,
        normalize: bool = False,
    ) -> None:
        root_path = Path(root)
        split_map, scene_files = _discover_scene_files(root_path)
        valid_split_map = {k: tuple(v) for k, v in split_map.items() if v}
        if not valid_split_map:
            raise FileNotFoundError(f"No nuScenes-mini CSVs found under {root_path}")
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
        coord_cols = self._infer_coord_cols(df)
        stride = max(1, int(round(self.source_fps / self.fps)))
        trajectories: List[np.ndarray] = []
        scene_ids: List[str] = []
        for track_id, track_df in df.groupby(group_col):
            track_df = track_df.sort_values(order_col)
            coords = track_df[list(coord_cols)].to_numpy(dtype=float)
            coords = coords[::stride]
            if len(coords) < self.total_len:
                continue
            for idx, window in enumerate(sliding_windows(coords, self.total_len)):
                trajectories.append(to_float32(window))
                scene_ids.append(f"{scene_name}:{track_id}:{idx}")
        return trajectories, scene_ids

    @staticmethod
    def _infer_group_col(df: pd.DataFrame) -> str:
        for col in ("instance_token", "track_id", "agent_id", "object_id"):
            if col in df.columns:
                return col
        raise ValueError("Could not infer track id column for nuScenes CSV")

    @staticmethod
    def _infer_order_col(df: pd.DataFrame) -> str:
        for col in ("timestamp", "frame", "frame_id"):
            if col in df.columns:
                return col
        raise ValueError("Could not infer ordering column for nuScenes CSV")

    @staticmethod
    def _infer_coord_cols(df: pd.DataFrame) -> Tuple[str, str]:
        for pair in (("center_x", "center_y"), ("x", "y"), ("pos_x", "pos_y")):
            if pair[0] in df.columns and pair[1] in df.columns:
                return pair
        raise ValueError("Could not infer coordinate columns for nuScenes CSV")


__all__ = ["NuScenesMiniDataset"]
