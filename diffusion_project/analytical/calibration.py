"""Validation-based calibration with fixed corruption per time and accepted-state tracking."""

import torch
from .common import cosine, images_from
from .wiener import wiener_order, mask_from_counts


@torch.no_grad()
def denoising_error(
    model, clean, noisy, t, parameter, batch_size=8, device="cpu", per_pixel=False
):
    total = None
    for start in range(0, len(clean), batch_size):
        truth = clean[start : start + batch_size].to(device)
        prediction = model.denoise(
            noisy[start : start + batch_size].to(device), t, parameter=parameter
        )
        squared = (prediction - truth).square()
        part = squared.sum((0, 1)) if per_pixel else squared.sum()
        total = part if total is None else total + part
    return total / (len(clean) * clean.shape[1]) if per_pixel else total / clean.numel()


@torch.no_grad()
def calibrate_step(
    model,
    kind,
    clean,
    noisy,
    t,
    previous=None,
    covariance=None,
    minimum=10,
    min_increment=4,
    fraction_increment=0.05,
    max_k=None,
    batch_size=8,
    device="cpu",
):
    """Greedy continuation. Once a candidate worsens, retain the last accepted state.

    Wiener: one spatial count per output position. ELS: one odd patch side length.
    Same clean/noisy examples are reused for every candidate at this time.
    """
    shape = clean.shape[1:]
    if kind == "wiener":
        order = wiener_order(covariance, shape, t, device=device).to(device)
        count = order.shape[0]
        maximum = count if max_k is None else min(int(max_k), count)
        current = (
            torch.full((count,), min(minimum, maximum), device=device, dtype=torch.long)
            if previous is None
            else previous.to(device).long().clamp(1, maximum)
        )
        param = mask_from_counts(order, current)
        best = denoising_error(
            model, clean, noisy, t, param, batch_size, device, True
        ).flatten()
        active = current < maximum
        while active.any():
            increments = torch.maximum(
                torch.full_like(current, min_increment),
                (current * fraction_increment).long(),
            )
            proposal = torch.where(
                active, (current + increments).clamp(max=maximum), current
            )
            trial = mask_from_counts(order, proposal)
            error = denoising_error(
                model, clean, noisy, t, trial, batch_size, device, True
            ).flatten()
            accepted = active & (error <= best) & (proposal > current)
            current = torch.where(accepted, proposal, current)
            best = torch.where(accepted, error, best)
            active = accepted & (current < maximum)
        return dict(
            parameter=mask_from_counts(order, current).cpu(),
            counts=current.cpu(),
            error=best.cpu(),
        )
    if kind != "els":
        raise ValueError("kind must be wiener or els")
    limit = int(max_k or (2 * shape[-1] - 1))
    current = int(previous) if previous is not None else int(minimum)
    if current % 2 == 0 or current < 1 or current > limit:
        raise ValueError("ELS minimum/previous must be positive odd and <= max_k")
    best = denoising_error(model, clean, noisy, t, current, batch_size, device)
    for candidate in range(current + 2, limit + 1, 2):
        error = denoising_error(model, clean, noisy, t, candidate, batch_size, device)
        if error > best:
            break
        current, best = candidate, error
    return dict(parameter=current, counts=current, error=best.cpu())
