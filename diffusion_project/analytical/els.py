"""Boundary-broken ELS, reimplemented from Kambm/convolutional_diffusion.

Interior patches share candidates across interior positions. Edge patches share
along the edge at the same distance from it. Corners retain their absolute
position. For k >= image size, use same-position local patches.
"""

from collections import defaultdict
import torch
from torch.nn import functional as F
from .common import EmpiricalModel, images_from, cosine


def boundary_groups(h, w, k):
    r = k // 2
    groups = defaultdict(list)
    for y in range(h):
        for x in range(w):
            if k >= min(h, w):
                key = (y, x)
            else:
                key = (
                    max(0, r - y),
                    max(0, y + r - h + 1),
                    max(0, r - x),
                    max(0, x + r - w + 1),
                )
            groups[key].append(y * w + x)
    return list(groups.values())


class BoundaryBrokenELS(EmpiricalModel):
    def __init__(
        self,
        dataset,
        source_batch_size=8,
        candidate_batch_size=1024,
        query_batch_size=128,
    ):
        self.dataset = dataset
        self.source_batch_size = source_batch_size
        self.candidate_batch_size = candidate_batch_size
        self.query_batch_size = query_batch_size

    @torch.no_grad()
    def denoise(self, x, t, parameter):
        k = int(parameter)
        if k < 1 or k % 2 == 0:
            raise ValueError("ELS patch size must be a positive odd integer")
        batch, c, h, w = x.shape
        a, b, _, _ = cosine(torch.as_tensor(t, device=x.device, dtype=x.dtype))
        if b <= 0:
            raise ValueError("Noise must be positive")
        if k >= min(h, w):
            # Same-position fallback without materializing enormous k-by-k patch banks.
            from .wiener import WienerLS

            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            y, z = yy.flatten(), xx.flatten()
            mask = ((y[:, None] - y[None]).abs() <= k // 2) & (
                (z[:, None] - z[None]).abs() <= k // 2
            )
            return WienerLS(
                self.dataset, self.source_batch_size, self.query_batch_size
            ).denoise(x, t, mask)
        groups = boundary_groups(h, w, k)
        query = F.unfold(x, k, padding=k // 2).transpose(1, 2)
        maximum = torch.full((batch, h * w), -torch.inf, device=x.device, dtype=x.dtype)
        numerator = torch.zeros(batch, h * w, c, device=x.device, dtype=x.dtype)
        denominator = torch.zeros_like(maximum)
        for images in images_from(self.dataset, self.source_batch_size):
            images = images.to(x)
            patches = F.unfold(images, k, padding=k // 2).transpose(1, 2)
            centers = images.flatten(2).transpose(1, 2)
            for group in groups:
                ids = torch.tensor(group, device=x.device)
                candidates = patches[:, ids].reshape(-1, c * k * k)
                values = centers[:, ids].reshape(-1, c)
                for qids in ids.split(self.query_batch_size):
                    q = query[:, qids]
                    mx = maximum[:, qids]
                    num = numerator[:, qids]
                    den = denominator[:, qids]
                    for start in range(0, len(candidates), self.candidate_batch_size):
                        cand = candidates[start : start + self.candidate_batch_size]
                        vals = values[start : start + self.candidate_batch_size]
                        logits = (
                            a * (q @ cand.T) - 0.5 * a * a * cand.square().sum(-1)
                        ) / (b * b)
                        updated = torch.maximum(mx, logits.max(-1).values)
                        scale = torch.exp(mx - updated)
                        weights = torch.exp(logits - updated[:, :, None])
                        num = num * scale[:, :, None] + weights @ vals
                        den = den * scale + weights.sum(-1)
                        mx = updated
                    maximum[:, qids] = mx
                    numerator[:, qids] = num
                    denominator[:, qids] = den
        return (numerator / denominator[:, :, None]).transpose(1, 2).reshape_as(x)
