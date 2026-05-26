from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_safe_group_count(out_channels), out_channels),
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
            nn.GroupNorm(_safe_group_count(channels), channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class DeepInpaintingNet(nn.Module):
    """CNN backbone for DeepInpainting.

    Input shape:
        [batch, 3, height, width]

    Output shape:
        [batch, n_bins, height, width]
    """

    def __init__(self, n_bins: int = 32, hidden_channels: int = 64, num_blocks: int = 7, dropout: float = 0.05):
        super().__init__()
        hidden = int(hidden_channels)
        self.stem = ConvAct(3, hidden, dropout=dropout)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout=dropout) for _ in range(int(num_blocks))])
        self.global_projection = nn.Sequential(
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
        local_features = self.blocks(self.stem(x))
        global_features = F.adaptive_avg_pool2d(local_features, 1).flatten(1)
        global_features = self.global_projection(global_features).view(
            local_features.shape[0], local_features.shape[1], 1, 1
        ).expand_as(local_features)
        return self.head(torch.cat([local_features, global_features], dim=1))


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
