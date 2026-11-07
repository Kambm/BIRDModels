"""Discrete cosine schedule and deterministic v-prediction DDIM."""

import math
import torch


def cosine_beta_schedule(T, s=0.008, max_beta=0.999):
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64) / T
    alphas_cumprod = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-8, max_beta).float()


class DiffusionSchedule:
    def __init__(self, T, device):
        betas = cosine_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0
        )

        self.T = T
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    def gather(self, arr, t, x_shape):
        out = arr.gather(0, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0, t, noise):
        alpha = self.gather(self.sqrt_alphas_cumprod, t, x0.shape)
        sigma = self.gather(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return alpha * x0 + sigma * noise

    def v_target(self, x0, t, noise):
        alpha = self.gather(self.sqrt_alphas_cumprod, t, x0.shape)
        sigma = self.gather(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return alpha * noise - sigma * x0

    def predict_x0_eps_from_v(self, xt, t, v):
        alpha = self.gather(self.sqrt_alphas_cumprod, t, xt.shape)
        sigma = self.gather(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        x0 = alpha * xt - sigma * v
        eps = sigma * xt + alpha * v
        return x0, eps


@torch.no_grad()
def ddim_sample(model, fixed_noise, schedule, sample_steps=100):
    if not 1 <= sample_steps <= schedule.T:
        raise ValueError("sample_steps must be between 1 and diffusion_steps")
    model.eval()
    device = next(model.parameters()).device
    x = fixed_noise.to(device).clone()
    ids = torch.linspace(schedule.T - 1, 0, sample_steps, device=device).long()
    for i, scalar in enumerate(ids):
        t = torch.full((len(x),), int(scalar), device=device, dtype=torch.long)
        x0, eps = schedule.predict_x0_eps_from_v(x, t, model(x, t))
        x0 = x0.clamp(-1, 1)
        if i == len(ids) - 1:
            x = x0
        else:
            prev = torch.full_like(t, int(ids[i + 1]))
            x = (
                schedule.gather(schedule.sqrt_alphas_cumprod, prev, x.shape) * x0
                + schedule.gather(schedule.sqrt_one_minus_alphas_cumprod, prev, x.shape)
                * eps
            )
    return x.cpu()
