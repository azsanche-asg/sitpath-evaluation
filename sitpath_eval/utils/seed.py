from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int, *, deterministic: bool = True) -> None:
    """Seed python, numpy, and torch for reproducible runs."""
    if seed is None:
        raise ValueError("Seed value must not be None")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = ["set_all_seeds"]
