from __future__ import annotations

import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(optimizer: Optimizer, total_steps: int, warmup_steps: int = 0, min_lr_scale: float = 0.0) -> LambdaLR:
    warmup_steps = max(0, warmup_steps)
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda)


__all__ = ["cosine_with_warmup"]
