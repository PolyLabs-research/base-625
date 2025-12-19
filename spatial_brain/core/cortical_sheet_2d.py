"""2D cortical sheet dynamics."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class CorticalSheet2D(nn.Module):
    """A 2D synthetic cortical sheet storing content vectors per (y, x) column.

    The state has shape (batch, height, width, content_dim). Beta acts as a spatial
    stencil controlling how much each location updates; gamma scales expression.
    """

    def __init__(
        self,
        height: int,
        width: int,
        content_dim: int,
        device: Optional[torch.device] = None,
        lateral: bool = True,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.content_dim = content_dim

        self.local = nn.Linear(content_dim, content_dim)
        self.nonlinearity = nn.Tanh()

        self.lateral = lateral
        if lateral:
            self.lateral_conv = nn.Conv2d(
                in_channels=content_dim,
                out_channels=content_dim,
                kernel_size=3,
                padding=1,
                groups=content_dim,
            )
        self.to(device=device)

    def step(
        self,
        prev_content: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Advance the 2D cortical sheet by one time step.

        Args:
            prev_content: Tensor of shape (batch, H, W, D).
            beta: Tensor of shape (batch, H, W) controlling inhibition (higher closes gates).
            gamma: Tensor of shape (batch, H, W) controlling gain/expression (anti-correlated with beta).
            external_input: Optional tensor (batch, H, W, D) providing sensory drive.
        """
        if external_input is None:
            external = torch.zeros_like(prev_content)
        else:
            external = external_input

        local_update = self.local(prev_content)

        if self.lateral:
            lateral_in = prev_content.permute(0, 3, 1, 2)  # (B, D, H, W)
            lateral_out = self.lateral_conv(lateral_in).permute(0, 2, 3, 1)
        else:
            lateral_out = torch.zeros_like(prev_content)

        proposed = self.nonlinearity(local_update + lateral_out + external)

        beta_gate = torch.sigmoid(-beta).unsqueeze(-1)  # high beta -> small update
        updated = prev_content + beta_gate * (proposed - prev_content)

        expressed = gamma.unsqueeze(-1) * updated
        return expressed

    def forward(
        self,
        prev_content: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias to `step` for nn.Module compatibility."""
        return self.step(prev_content, beta, gamma, external_input)
