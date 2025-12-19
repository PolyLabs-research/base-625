"""Top-down control producing beta/gamma stencils for a 1D cortical line."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ControlNetwork1D(nn.Module):
    """Synthetic analogue of PFC beta stencils."""

    def __init__(
        self,
        num_columns: int,
        context_dim: int,
        max_steps: int,
        hidden_dim: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_columns = num_columns
        self.context_dim = context_dim
        self.max_steps = max_steps

        self.time_embed = nn.Embedding(max_steps, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_columns),
        )
        self.to(device=device)

    def forward(self, context: torch.Tensor, t: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce beta and gamma fields for a given context and time step."""
        if isinstance(t, int):
            t_tensor = torch.tensor([t], device=context.device)
            t_tensor = t_tensor.expand(context.shape[0])
        else:
            t_tensor = t.to(context.device)
        time_feat = self.time_embed(t_tensor)
        concat = torch.cat([context, time_feat], dim=-1)
        beta = self.mlp(concat)
        gamma = torch.sigmoid(-beta)  # Anti-correlated with beta; opens when beta closes.
        return beta, gamma
