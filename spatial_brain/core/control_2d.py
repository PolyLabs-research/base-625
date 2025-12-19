"""Top-down control producing beta/gamma stencils for a 2D cortical sheet."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ControlNetwork2D(nn.Module):
    """Control network that emits spatial beta/gamma fields over a 2D grid."""

    def __init__(
        self,
        height: int,
        width: int,
        context_dim: int,
        max_steps: int,
        hidden_dim: int = 128,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.context_dim = context_dim
        self.max_steps = max_steps

        self.time_embed = nn.Embedding(max_steps, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, height * width),
        )
        self.to(device=device)

    def forward(self, context: torch.Tensor, t: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce beta and gamma fields for a given context and time step.

        Args:
            context: (batch, context_dim)
            t: scalar int or tensor of shape (batch,)

        Returns:
            beta: (batch, height, width)
            gamma: (batch, height, width)
        """
        if isinstance(t, int):
            t_tensor = torch.tensor([t], device=context.device).expand(context.shape[0])
        else:
            t_tensor = t.to(context.device)
        time_feat = self.time_embed(t_tensor)
        concat = torch.cat([context, time_feat], dim=-1)
        beta_flat = self.mlp(concat)
        beta = beta_flat.view(-1, self.height, self.width)
        gamma = torch.sigmoid(-beta)  # Anti-correlated with beta.
        return beta, gamma
