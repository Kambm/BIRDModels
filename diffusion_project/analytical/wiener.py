"""Wiener-selected local empirical denoiser (not a linear Gaussian baseline)."""

import torch
from .common import EmpiricalModel, images_from, cosine


@torch.no_grad()
def dataset_statistics(dataset, batch_size=64, device="cpu", max_samples=None):
    """Population covariance; normalize by the actual number of samples consumed."""
    total = second = None
    count = 0
    shape = None
    for images in images_from(dataset, batch_size):
        if max_samples is not None:
            images = images[: max_samples - count]
        if not len(images):
            break
        shape = tuple(images.shape[1:])
        x = images.to(device=device, dtype=torch.float64).flatten(1)
        if total is None:
            total = torch.zeros(x.shape[1], device=device, dtype=x.dtype)
            second = torch.zeros(x.shape[1], x.shape[1], device=device, dtype=x.dtype)
        total += x.sum(0)
        second += x.T @ x
        count += len(x)
        if max_samples is not None and count >= max_samples:
            break
    if not count:
        raise ValueError("Empty statistics dataset")
    mean = total / count
    return (
        mean.reshape(shape).float().cpu(),
        (second / count - mean[:, None] * mean[None]).float().cpu(),
    )


@torch.no_grad()
def wiener_order(covariance, shape, t, device="cpu"):
    """Spatial importance sums absolute filter coefficients over both channel axes.

    Drop the positive scalar a: it does not affect ranking and vanishes at t=1.
    The t=1 ordering is the limiting covariance ordering, avoiding a zero filter.
    """
    c, h, w = shape
    cov = covariance.to(device)
    a, b, _, _ = cosine(torch.as_tensor(t, device=device, dtype=cov.dtype))
    matrix = a * a * cov + b * b * torch.eye(len(cov), device=device, dtype=cov.dtype)
    filt = torch.linalg.solve(matrix, cov.T).T
    importance = filt.reshape(c, h * w, c, h * w).abs().sum((0, 2))
    return importance.argsort(dim=-1, descending=True, stable=True).cpu()


def mask_from_counts(order, counts):
    counts = torch.as_tensor(counts, device=order.device).long().expand(order.shape[0])
    if counts.min() < 1 or counts.max() > order.shape[1]:
        raise ValueError("Mask counts must be between 1 and image spatial size")
    mask = torch.zeros_like(order, dtype=torch.bool)
    ranks = torch.arange(order.shape[1], device=order.device)[None].expand_as(order)
    mask.scatter_(1, order, ranks < counts[:, None])
    return mask


class WienerLS(EmpiricalModel):
    """Posterior over training images, with one calibrated spatial mask per target.

    Both channels and spatial positions participate in the distance. The posterior
    returns the source pixel at the same absolute location.
    To use square, boundary-broken equivariant patches instead, select 'els' in the
    unified analytical configuration (BoundaryBrokenELS).
    """

    def __init__(self, dataset, source_batch_size=32, target_batch_size=64):
        self.dataset = dataset
        self.source_batch_size = source_batch_size
        self.target_batch_size = target_batch_size

    @torch.no_grad()
    def denoise(self, x, t, parameter):
        batch, channels, h, w = x.shape
        a, b, _, _ = cosine(torch.as_tensor(t, device=x.device, dtype=x.dtype))
        if b <= 0:
            raise ValueError("Noise must be positive")
        mask = parameter.reshape(h * w, h * w).to(x)
        output = torch.empty(batch, channels, h * w, device=x.device, dtype=x.dtype)
        for start in range(0, h * w, self.target_batch_size):
            stop = min(start + self.target_batch_size, h * w)
            m = mask[start:stop]
            maximum = torch.full(
                (batch, stop - start), -torch.inf, device=x.device, dtype=x.dtype
            )
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros(
                batch, stop - start, channels, device=x.device, dtype=x.dtype
            )
            for images in images_from(self.dataset, self.source_batch_size):
                images = images.to(x)
                # [batch,source,spatial]; chunking bounds this allocation.
                distance = (x[:, None] - a * images[None]).square().sum(2).flatten(2)
                logits = -torch.einsum("bns,ps->bnp", distance, m) / (2 * b * b)
                new_max = torch.maximum(maximum, logits.max(1).values)
                scale = torch.exp(maximum - new_max)
                weights = torch.exp(logits - new_max[:, None])
                values = images.flatten(2)[:, :, start:stop]
                numerator = numerator * scale[:, :, None] + torch.einsum(
                    "bnp,ncp->bpc", weights, values
                )
                denominator = denominator * scale + weights.sum(1)
                maximum = new_max
            output[:, :, start:stop] = (numerator / denominator[:, :, None]).transpose(
                1, 2
            )
        return output.reshape_as(x)
