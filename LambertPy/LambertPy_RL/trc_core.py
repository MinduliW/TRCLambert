"""Minimal TRC building blocks for RL policy."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class NetConfig:
    d_z: int = 128
    d_h: int = 256
    n_heads: int = 4
    n_blocks: int = 2
    dropout: float = 0.0


def make_mlp(d_in: int, d_h: int, d_out: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_in, d_h),
        nn.GELU(),
        nn.Linear(d_h, d_h),
        nn.GELU(),
        nn.Linear(d_h, d_out),
    )


class ReasoningModule(nn.Module):
    """
    Cross-latent fusion block.

    Accepts query latent and any number of context latents of same shape, then
    applies residual MLP updates.
    """

    def __init__(self, d_z: int, d_h: int, n_heads: int, n_blocks: int, dropout: float):
        super().__init__()
        del n_heads  # kept for API compatibility
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_z),
                    nn.Linear(d_z, d_h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_h, d_z),
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, q: torch.Tensor, *ctx: torch.Tensor) -> torch.Tensor:
        x = q
        if len(ctx) > 0:
            c = torch.stack(ctx, dim=0).mean(dim=0)
            x = x + c
        for b in self.blocks:
            x = x + b(x)
        return x
