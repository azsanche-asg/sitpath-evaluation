from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TrajectoryBatch:
    obs: np.ndarray
    fut: np.ndarray
    meta: List[Dict]


class TrajectoryDataset(abc.ABC):
    def __init__(
        self,
        root: str | Path,
        obs_len: int,
        pred_len: int,
        fps: float,
        *,
        normalize: bool = False,
        split_map: Optional[Dict[str, Sequence[str]]] = None,
    ) -> None:
        self.root = Path(root)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.fps = fps
        self.normalize = normalize
        self._norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.split_map = split_map or {}

    @property
    def total_len(self) -> int:
        return self.obs_len + self.pred_len

    def split(self, split_name: str) -> Tuple[List[np.ndarray], List[str]]:
        """Gather scenes corresponding to the split name."""
        scenes = self.split_map.get(split_name)
        if not scenes:
            scenes = self.scene_ids
        return self._gather_scenes(scenes)

    def leave_one_scene_out(self) -> Dict[str, Tuple[Tuple[List[np.ndarray], List[str]], Tuple[List[np.ndarray], List[str]]]]:
        splits: Dict[
            str,
            Tuple[Tuple[List[np.ndarray], List[str]], Tuple[List[np.ndarray], List[str]]],
        ] = {}
        scenes = self.scene_ids
        for held_out in scenes:
            train_trajs: List[np.ndarray] = []
            train_ids: List[str] = []
            val_trajs: List[np.ndarray] = []
            val_ids: List[str] = []
            for scene in scenes:
                trajs, ids = self._scene_cache[scene]
                if scene == held_out:
                    val_trajs.extend(trajs)
                    val_ids.extend(ids)
                else:
                    train_trajs.extend(trajs)
                    train_ids.extend(ids)
            splits[held_out] = ((train_trajs, train_ids), (val_trajs, val_ids))
        return splits

    @property
    def scene_ids(self) -> Sequence[str]:
        return list(self._scene_cache.keys())

    @property
    def _scene_cache(self) -> Dict[str, Tuple[List[np.ndarray], List[str]]]:
        if not hasattr(self, "__scene_cache"):
            cache = self._load_all_scenes()
            setattr(self, "__scene_cache", cache)
        return getattr(self, "__scene_cache")

    @abc.abstractmethod
    def _load_all_scenes(self) -> Dict[str, Tuple[List[np.ndarray], List[str]]]:
        """Load trajectories grouped per scene."""

    def _gather_scenes(self, scene_names: Sequence[str]) -> Tuple[List[np.ndarray], List[str]]:
        trajectories: List[np.ndarray] = []
        scene_ids: List[str] = []
        for scene in scene_names:
            trajs, ids = self._scene_cache[scene]
            trajectories.extend(trajs)
            scene_ids.extend(ids)
        return trajectories, scene_ids

    def iterator(self, split_name: str) -> Iterator[TrajectoryBatch]:
        trajectories, scene_ids = self.split(split_name)
        for traj, scene in zip(trajectories, scene_ids):
            obs, fut = traj[: self.obs_len], traj[self.obs_len : self.total_len]
            if self.normalize:
                obs, fut = self._normalize_traj(obs, fut)
            meta = {"scene": scene, "length": len(traj)}
            yield TrajectoryBatch(obs=obs, fut=fut, meta=[meta])

    def batch_iterator(self, split_name: str, batch_size: int) -> Iterator[TrajectoryBatch]:
        buffer_obs: List[np.ndarray] = []
        buffer_fut: List[np.ndarray] = []
        buffer_meta: List[Dict] = []
        for batch in self.iterator(split_name):
            buffer_obs.append(batch.obs)
            buffer_fut.append(batch.fut)
            buffer_meta.extend(batch.meta)
            if len(buffer_obs) == batch_size:
                yield self._stack_batch(buffer_obs, buffer_fut, buffer_meta)
                buffer_obs, buffer_fut, buffer_meta = [], [], []
        if buffer_obs:
            yield self._stack_batch(buffer_obs, buffer_fut, buffer_meta)

    def _stack_batch(
        self,
        obs_list: Sequence[np.ndarray],
        fut_list: Sequence[np.ndarray],
        meta_list: Sequence[Dict],
    ) -> TrajectoryBatch:
        return TrajectoryBatch(
            obs=np.stack(obs_list, axis=0),
            fut=np.stack(fut_list, axis=0),
            meta=list(meta_list),
        )

    def _normalize_traj(self, obs: np.ndarray, fut: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._norm_stats is None:
            concat = np.concatenate([obs, fut], axis=0)
            mean = np.mean(concat, axis=0)
            std = np.std(concat, axis=0) + 1e-6
            self._norm_stats = (mean, std)
        mean, std = self._norm_stats
        return (obs - mean) / std, (fut - mean) / std


def sliding_windows(coords: np.ndarray, window: int, step: int = 1) -> List[np.ndarray]:
    windows = []
    for start in range(0, len(coords) - window + 1, step):
        windows.append(coords[start : start + window])
    return windows


def to_float32(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32, copy=False)


__all__ = [
    "TrajectoryDataset",
    "TrajectoryBatch",
    "sliding_windows",
    "to_float32",
]
