from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .common import TrajectoryDataset, sliding_windows, to_float32

SCENES = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
DEFAULT_SPLITS = {
    "train": ["ETH", "HOTEL", "UNIV", "ZARA1"],
    "val": ["ZARA2"],
    "test": SCENES,
}


class ETHUCYDataset(TrajectoryDataset):
    def __init__(
        self,
        root: str | Path = "data/eth_ucy",
        obs_len: int = 8,
        pred_len: int = 12,
        fps: float = 2.5,
        *,
        split_map: Dict[str, Sequence[str]] | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            obs_len=obs_len,
            pred_len=pred_len,
            fps=fps,
            normalize=normalize,
            split_map=split_map or DEFAULT_SPLITS,
        )

    def _load_all_scenes(self) -> Dict[str, Tuple[List[np.ndarray], List[str]]]:
        cache: Dict[str, Tuple[List[np.ndarray], List[str]]] = {}
        for scene in SCENES:
            trajectories, ids = self._load_scene(scene)
            cache[scene] = (trajectories, ids)
        return cache

    def _load_scene(self, scene: str) -> Tuple[List[np.ndarray], List[str]]:
        scene_dir = self._resolve_scene_dir(scene)
        csv_path = scene_dir / "trajectories.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing ETH/UCY CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        required_cols = {"track_id", "frame_id", "x", "y"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV {csv_path} missing columns {required_cols}")
        trajectories: List[np.ndarray] = []
        scene_ids: List[str] = []
        for track_id, track_df in df.groupby("track_id"):
            trace = track_df.sort_values("frame_id")[ ["x", "y"] ].to_numpy(dtype=float)
            if len(trace) < self.total_len:
                continue
            for idx, window in enumerate(sliding_windows(trace, self.total_len, step=1)):
                trajectories.append(to_float32(window))
                scene_ids.append(f"{scene}:{track_id}:{idx}")
        return trajectories, scene_ids

    def _resolve_scene_dir(self, scene: str) -> Path:
        candidates = [self.root / scene, self.root / scene.lower(), self.root / scene.upper()]
        for directory in candidates:
            if directory.exists():
                return directory
        raise FileNotFoundError(f"Scene directory not found for {scene} under {self.root}")


__all__ = ["ETHUCYDataset"]
