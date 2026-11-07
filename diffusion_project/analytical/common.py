"""Stable empirical posteriors and continuous cosine probability-flow sampling."""

import math
import torch
from torch.utils.data import DataLoader


def images_from(dataset, batch_size=64):
    if torch.is_tensor(dataset):
        for part in dataset.split(batch_size):
            yield part
    else:
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            yield batch[0] if isinstance(batch, (tuple, list)) else batch


def cosine(t):
    t = torch.as_tensor(t)
    theta = t * math.pi / 2
    # Exact endpoint prevents tiny negative cos(pi/2) in float32.
    a = torch.where(t == 1, torch.zeros_like(t), torch.cos(theta))
    b = torch.sin(theta)
    return a, b, -math.pi / 2 * b, math.pi / 2 * a


def posterior_mean(query, candidates, values, a, b, chunk_size=2048):
    """query [Q,D], candidates [N,D], values [N,C]. Chunk-invariant softmax.

    Squared query norms cancel in the posterior. Products retain the candidate
    norm term, and running maxima keep weights stable at small noise.
    """
    if len(candidates) == 0 or b <= 0:
        raise ValueError("Need candidates and positive noise")
    maximum = torch.full(
        (len(query),), -torch.inf, device=query.device, dtype=query.dtype
    )
    numerator = torch.zeros(
        len(query), values.shape[-1], device=query.device, dtype=query.dtype
    )
    denominator = torch.zeros_like(maximum)
    for start in range(0, len(candidates), chunk_size):
        c = candidates[start : start + chunk_size].to(query)
        v = values[start : start + chunk_size].to(query)
        logits = (a * (query @ c.T) - 0.5 * a * a * c.square().sum(-1)[None]) / (b * b)
        new_max = torch.maximum(maximum, logits.max(-1).values)
        scale = torch.exp(maximum - new_max)
        weights = torch.exp(logits - new_max[:, None])
        numerator = numerator * scale[:, None] + weights @ v
        denominator = denominator * scale + weights.sum(-1)
        maximum = new_max
    return numerator / denominator[:, None]


class EmpiricalModel:
    def score(self, x, t, **kwargs):
        a, b, _, _ = cosine(torch.as_tensor(t, device=x.device, dtype=x.dtype))
        return (a * self.denoise(x, t, **kwargs) - x) / (b * b)

    def velocity(self, x, t, **kwargs):
        a, b, ad, bd = cosine(torch.as_tensor(t, device=x.device, dtype=x.dtype))
        clean = self.denoise(x, t, **kwargs)
        return ad * clean + bd * (x - a * clean) / b


@torch.no_grad()
def sample(model, seeds, times, parameters, method="euler", batch_size=8, device="cpu"):
    """Times ascend from tmin to 1; parameters are aligned to those exact times.

    Euler and DDIM use the same calibrated times. Neither applies an additional
    terminal clean projection.
    """
    times = torch.as_tensor(times, dtype=torch.float64)
    if (
        len(times) != len(parameters)
        or len(times) < 2
        or not torch.all(times[1:] > times[:-1])
    ):
        raise ValueError("Need ascending times and one parameter per time")
    if not 0 < times[0] < times[-1] <= 1:
        raise ValueError("Require 0 < tmin < tmax <= 1")
    if method not in ("euler", "ddim"):
        raise ValueError("method must be euler or ddim")
    results = []
    for seed_batch in seeds.split(batch_size):
        x = seed_batch.to(device).clone()
        for i in range(len(times) - 1, 0, -1):
            t, previous = times[i].to(x), times[i - 1].to(x)
            param = (
                parameters[i].to(x.device)
                if torch.is_tensor(parameters[i])
                else parameters[i]
            )
            a, b, ad, bd = cosine(t)
            clean = model.denoise(x, t, parameter=param)
            eps = (x - a * clean) / b
            if method == "euler":
                x += (previous - t) * (ad * clean + bd * eps)
            else:
                ap, bp, _, _ = cosine(previous)
                x = ap * clean + bp * eps
        results.append(x.cpu())
    return torch.cat(results)
