"""Multiscale attention UNet with timestep conditioning."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class TimestepEmbedder(nn.Module):
    def __init__(self, model_channels, time_dim=None, freq_dim=None):
        super().__init__()
        self.freq_dim = freq_dim or model_channels
        time_dim = time_dim or 4 * model_channels
        self.mlp = nn.Sequential(
            nn.Linear(self.freq_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.freq_dim))


def norm_layer(channels):
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise ValueError(
        f"could not find a valid GroupNorm group count for {channels} channels"
    )


class TimestepSequential(nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.0):
        super().__init__()
        self.in_layers = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.time_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2 * out_channels),
        )
        self.out_norm = norm_layer(out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

        nn.init.zeros_(self.out_layers[-1].weight)
        nn.init.zeros_(self.out_layers[-1].bias)

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = (
            self.time_layers(emb).view(emb.shape[0], -1, 1, 1).chunk(2, dim=1)
        )
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        h = h.reshape(B, C, H, W)

        return x + self.proj_out(h)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class AttentionUNet(nn.Module):
    def __init__(
        self,
        image_size=64,
        in_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(16,),
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        assert image_size % (2 ** (len(channel_mult) - 1)) == 0

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.time_dim = 4 * model_channels

        self.time_embed = TimestepEmbedder(model_channels, time_dim=self.time_dim)
        self.input_conv = nn.Conv2d(
            in_channels, model_channels, kernel_size=3, padding=1
        )

        self.down_blocks = nn.ModuleList()
        self.input_block_channels = [model_channels]

        ch = model_channels
        resolution = image_size

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, self.time_dim, dropout=dropout)]
                ch = out_ch
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepSequential(*layers))
                self.input_block_channels.append(ch)

            if level != len(channel_mult) - 1:
                self.down_blocks.append(TimestepSequential(Downsample(ch)))
                self.input_block_channels.append(ch)
                resolution //= 2

        self.middle_block = TimestepSequential(
            ResBlock(ch, ch, self.time_dim, dropout=dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(ch, ch, self.time_dim, dropout=dropout),
        )

        self.up_blocks = nn.ModuleList()
        input_block_channels = list(self.input_block_channels)

        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult

            for block_idx in range(num_res_blocks + 1):
                skip_ch = input_block_channels.pop()
                layers = [
                    ResBlock(ch + skip_ch, out_ch, self.time_dim, dropout=dropout)
                ]
                ch = out_ch

                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))

                if level != 0 and block_idx == num_res_blocks:
                    layers.append(Upsample(ch))
                    resolution *= 2

                self.up_blocks.append(TimestepSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        emb = self.time_embed(t)

        h = self.input_conv(x)
        skips = [h]

        for block in self.down_blocks:
            h = block(h, emb)
            skips.append(h)

        h = self.middle_block(h, emb)

        for block in self.up_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = block(h, emb)

        return self.out(h)
