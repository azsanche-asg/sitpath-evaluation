from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class DecodeState:
    position: np.ndarray


class SitPathInverseDecoder:
    def __init__(self, M: int, R: float, B: int) -> None:
        self.M = M
        self.R = R
        self.B = B
        self.sector_angle = 2 * math.pi / M

    def decode_step(self, state: DecodeState, token: Dict) -> np.ndarray:
        occupancy = np.asarray(token["occupancy"], dtype=float)
        tempo = token["tempo_bin"]
        stall = token.get("stall", 0)
        if stall:
            return state.position.copy()
        sector = int(np.argmax(occupancy))
        radial_bin = max(1, occupancy[sector])
        radius = (radial_bin / self.B) * self.R
        angle = (sector + 0.5) * self.sector_angle - math.pi
        displacement = np.array([math.cos(angle), math.sin(angle)]) * radius * 0.1 * (tempo + 1)
        state.position = state.position + displacement
        return state.position.copy()


__all__ = ["SitPathInverseDecoder", "DecodeState"]
