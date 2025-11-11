from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.io import safe_mkdirs


@dataclass
class Token:
    occupancy: np.ndarray  # shape [M], radial bins per sector (0=empty)
    tempo_bin: int
    stall: int


@dataclass
class TokenResult:
    tokens: List[Token]
    ids: List[int]
    vocab: Dict[str, int]


class SitPathTokenizer:
    """Symbolic tokenizer for trajectory + neighborhood context."""

    def __init__(
        self,
        M: int = 16,
        R: float = 5.0,
        B: int = 4,
        K_tau: int = 3,
        collapse: bool = True,
        stall_eps: float = 0.05,
        *,
        max_speed: Optional[float] = None,
    ) -> None:
        self.M = M
        self.R = R
        self.B = B
        self.K_tau = K_tau
        self.collapse = collapse
        self.stall_eps = stall_eps
        self.sector_angle = 2 * math.pi / max(1, M)
        self.max_speed = max_speed if max_speed is not None else R  # proxy upper-bound

        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self._next_id = 0
        self._cache: Dict[str, int] = {}

    def tokenize(self, traj: np.ndarray, neighbors: Optional[Iterable[np.ndarray]] = None) -> TokenResult:
        traj = np.asarray(traj, dtype=float)
        if traj.ndim != 2 or traj.shape[1] != 2:
            raise ValueError("Trajectory must be shaped [T,2]")
        neighbor_list: List[np.ndarray] = []
        if neighbors is not None:
            for nbr in neighbors:
                arr = np.asarray(nbr, dtype=float)
                if arr.shape[0] < traj.shape[0]:
                    # pad by repeating last seen coordinate
                    pad = np.repeat(arr[-1:, :], traj.shape[0] - arr.shape[0], axis=0)
                    arr = np.concatenate([arr, pad], axis=0)
                neighbor_list.append(arr)

        tokens: List[Token] = []
        ids: List[int] = []
        for t in range(1, len(traj)):
            token = self._tokenize_step(traj, neighbor_list, t)
            token_id = self._encode_token(token)
            tokens.append(token)
            ids.append(token_id)
        return TokenResult(tokens=tokens, ids=ids, vocab=self.vocab)

    def _tokenize_step(self, traj: np.ndarray, neighbors: Sequence[np.ndarray], t: int) -> Token:
        origin = traj[t]
        prev = traj[t - 1]
        occupancy = np.zeros(self.M, dtype=np.int32)
        best_dist = np.full(self.M, np.inf)
        rel_positions = [nbr[t] - origin for nbr in neighbors if len(nbr) > t]
        for rel in rel_positions:
            dist = float(np.linalg.norm(rel))
            if dist <= 0 or dist > self.R:
                continue
            angle = math.atan2(rel[1], rel[0]) + math.pi  # shift to [0, 2pi)
            sector = int(angle / self.sector_angle) % self.M
            radial_bin = self._radial_bin(dist)
            if self.collapse:
                if dist < best_dist[sector]:
                    best_dist[sector] = dist
                    occupancy[sector] = radial_bin
            else:
                occupancy[sector] = max(occupancy[sector], radial_bin)
        tempo_bin = self._tempo_bin(origin - prev)
        stall_flag = int(np.linalg.norm(origin - prev) < self.stall_eps)
        return Token(occupancy=occupancy, tempo_bin=tempo_bin, stall=stall_flag)

    def _radial_bin(self, distance: float) -> int:
        bin_width = self.R / max(1, self.B)
        bin_idx = int(math.ceil(distance / bin_width))
        return max(1, min(self.B, bin_idx))

    def _tempo_bin(self, velocity: np.ndarray) -> int:
        speed = float(np.linalg.norm(velocity))
        normalized = min(0.9999, speed / max(self.max_speed, 1e-6))
        bin_idx = int(normalized * self.K_tau)
        return min(self.K_tau - 1, max(0, bin_idx))

    def _encode_token(self, token: Token) -> int:
        key = self._token_key(token)
        if key in self._cache:
            return self._cache[key]
        idx = self.vocab.setdefault(key, self._next_id)
        if idx == self._next_id:
            self.reverse_vocab[idx] = key
            self._next_id += 1
        self._cache[key] = idx
        return idx

    def _token_key(self, token: Token) -> str:
        occ_str = ",".join(map(str, token.occupancy.tolist()))
        return f"{occ_str}|tau={token.tempo_bin}|stall={token.stall}"

    # Vocabulary management -------------------------------------------------
    def save_vocab(self, path: str | Path) -> None:
        path = Path(path)
        safe_mkdirs(path.parent)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.vocab, fh, indent=2)

    def load_vocab(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.vocab = {str(k): int(v) for k, v in data.items()}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self._next_id = max(self.reverse_vocab.keys(), default=-1) + 1
        self._cache = {}

    def encode_sequence(self, tokens: Sequence[Token]) -> List[int]:
        return [self._encode_token(token) for token in tokens]

    def decode_ids(self, ids: Sequence[int]) -> List[str]:
        return [self.reverse_vocab.get(idx, "<unk>") for idx in ids]


__all__ = ["SitPathTokenizer", "Token", "TokenResult"]
