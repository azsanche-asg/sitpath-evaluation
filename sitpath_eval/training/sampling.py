from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn


SamplerFn = Callable[[nn.Module, dict], Tensor]


def k_sample_predictions(model: torch.nn.Module, batch: dict, sampler_fn: SamplerFn, k: int) -> Tensor:
    """Collect K stochastic predictions for minADE/diversity."""
    samples = []
    for _ in range(k):
        samples.append(sampler_fn(model, batch))
    return torch.stack(samples, dim=0)


def greedy_sampler(model: nn.Module, batch: dict) -> Tensor:
    device = next(model.parameters()).device
    inputs = batch.get("obs") or batch.get("tokens")
    if inputs is None:
        raise ValueError("Batch must contain 'obs' or 'tokens' for greedy sampling")
    inputs = inputs.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    return outputs


def diversity_score(samples: Tensor) -> Tensor:
    \"\"\"Average pairwise endpoint distance per trajectory.\"\"\"\n    endpoints = samples[:, :, -1].permute(1, 0, 2)  # [B,K,2]\n    B, K, _ = endpoints.shape\n    if K < 2:\n        return torch.zeros(B, device=endpoints.device)\n    diffs = endpoints.unsqueeze(2) - endpoints.unsqueeze(1)\n    dists = torch.norm(diffs, dim=-1)\n    tri = torch.triu_indices(K, K, offset=1)\n    pairwise = dists[:, tri[0], tri[1]]\n    return pairwise.mean(dim=-1)\n*** End Patch


__all__ = ["k_sample_predictions", "greedy_sampler", "diversity_score"]
