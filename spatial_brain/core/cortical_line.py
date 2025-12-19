"""Cortical line content dynamics."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class CorticalLine(nn.Module):
    """A 1D cortical sheet of columns that hold continuous content vectors.

    The model maintains a content state of shape (batch, num_columns, content_dim).
    Beta acts as a top-down stencil: high beta closes gates and slows updates.
    Gamma is anti-correlated with beta and controls expression/gain.
    """

    def __init__(
        self,
        num_columns: int,
        content_dim: int,
        device: Optional[torch.device] = None,
        lateral: bool = True,
    ) -> None:
        super().__init__()
        self.num_columns = num_columns
        self.content_dim = content_dim
        self.device = device

        self.local = nn.Linear(content_dim, content_dim)
        self.nonlinearity = nn.Tanh()

        # Depthwise 1D conv gives simple lateral diffusion across neighboring columns.
        self.lateral = lateral
        if lateral:
            self.lateral_conv = nn.Conv1d(
                in_channels=content_dim,
                out_channels=content_dim,
                kernel_size=3,
                padding=1,
                groups=content_dim,
            )
        self.to(device=device)

    def forward(
        self,
        prev_state: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        external_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Advance the cortical state by one step.

        Args:
            prev_state: Tensor of shape (batch, num_columns, content_dim).
            beta: Tensor of shape (batch, num_columns) controlling inhibition (higher = more closed).
            gamma: Tensor of shape (batch, num_columns) controlling expression/gain (anti-correlated with beta).
            external_input: Optional tensor (batch, num_columns, content_dim) for sensory drive.

        Returns:
            Tensor with the updated content state, same shape as prev_state.
        """
        if external_input is None:
            external = torch.zeros_like(prev_state)
        else:
            external = external_input

        # Local update at each column.
        local_update = self.local(prev_state)

        # Optional lateral diffusion across neighboring columns.
        if self.lateral:
            # Conv expects (batch, channels, width).
            lateral_in = prev_state.permute(0, 2, 1)
            lateral_out = self.lateral_conv(lateral_in).permute(0, 2, 1)
        else:
            lateral_out = torch.zeros_like(prev_state)

        proposed = self.nonlinearity(local_update + lateral_out + external)

        # Beta shrinks the step size into the proposed state.
        beta_gate = torch.sigmoid(-beta).unsqueeze(-1)  # high beta -> small gate
        updated = prev_state + beta_gate * (proposed - prev_state)

        # Gamma scales final expression.
        expressed = gamma.unsqueeze(-1) * updated
        return expressed
