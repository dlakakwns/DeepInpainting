from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ConvAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_safe_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = ConvAct(channels, channels, dropout=dropout)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_safe_groups(channels), channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class GlobalPoolMCQLNet(nn.Module):
    """Small CNN used by the method-only MCQL imputer.

    Input:  [B, 3, H, W]
    Output: [B, K, H, W]
    """

    def __init__(self, n_bins: int = 32, hidden_channels: int = 64, num_blocks: int = 7, dropout: float = 0.05):
        super().__init__()
        hidden = int(hidden_channels)
        self.stem = ConvAct(3, hidden, dropout=dropout)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout=dropout) for _ in range(int(num_blocks))])
        self.global_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            ConvAct(hidden * 2, hidden, dropout=dropout),
            nn.Conv2d(hidden, int(n_bins), kernel_size=1),
        )

    def forward(self, x):
        h = self.blocks(self.stem(x))
        g = F.adaptive_avg_pool2d(h, 1).flatten(1)
        g = self.global_proj(g).view(h.shape[0], h.shape[1], 1, 1).expand_as(h)
        return self.head(torch.cat([h, g], dim=1))


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
