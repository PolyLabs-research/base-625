"""Brain-controlled agent for 2D navigation using the spatial sheet."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from spatial_brain.core.control_2d import ControlNetwork2D
from spatial_brain.core.cortical_sheet_2d import CorticalSheet2D


class BrainAgent2D(nn.Module):
    """Wrap CorticalSheet2D + ControlNetwork2D into a navigation policy."""

    def __init__(
        self,
        sheet: CorticalSheet2D,
        control: ControlNetwork2D,
        action_dim: int = 4,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.sheet = sheet
        self.control = control
        self.action_dim = action_dim
        self.device = torch.device(device)

        content_dim = getattr(sheet, "content_dim", 32)
        # Simple 1x1 conv encodes 3-channel grid input into content_dim features.
        self.obs_encoder = nn.Conv2d(in_channels=3, out_channels=content_dim, kernel_size=1)
        self.policy_head = nn.Linear(content_dim, action_dim)
        self.value_head = nn.Linear(content_dim, 1)
        self.to(self.device)

        self.state: torch.Tensor | None = None

    def reset_state(self, batch_size: int = 1) -> None:
        """Clear internal sheet state for a new episode."""
        self.state = torch.zeros(
            (batch_size, self.sheet.height, self.sheet.width, self.sheet.content_dim),
            device=self.device,
        )

    def forward_step(
        self,
        obs_grid: torch.Tensor,
        step_idx: int,
        extra_context: Optional[torch.Tensor] = None,
        sample: bool = False,
    ) -> Dict[str, Any]:
        """Process one timestep and return action + brain fields."""
        if self.state is None or self.state.shape[0] != obs_grid.shape[0]:
            self.reset_state(batch_size=obs_grid.shape[0])

        bsz = obs_grid.shape[0]
        obs_grid = obs_grid.to(self.device)

        # Build context for control net.
        context_dim = getattr(self.control, "context_dim", None)
        if extra_context is not None:
            context = extra_context.to(self.device)
        elif context_dim is not None:
            context = torch.zeros((bsz, context_dim), device=self.device)
        else:
            context = torch.zeros((bsz, 1), device=self.device)

        beta, gamma = self.control(context, step_idx)

        # Encode observation grid into external input matching sheet shape.
        external_input = self._encode_obs(obs_grid)

        # Update cortical sheet.
        self.state = self.sheet(self.state, beta, gamma, external_input=external_input)

        # Pool content for policy logits.
        pooled = self.state.mean(dim=(1, 2))  # (B, content_dim)
        logits = self.policy_head(pooled)
        dist = torch.distributions.Categorical(logits=logits)
        if sample:
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=-1)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        value = self.value_head(pooled).squeeze(-1)

        content_norm = torch.norm(self.state, dim=-1)  # (B, H, W)

        return {
            "action": action,
            "logits": logits,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "beta": beta,
            "gamma": gamma,
            "content_norm": content_norm,
            "state": self.state,
        }

    def select_action(self, obs: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
        """Convenience wrapper to consume env observation dict."""
        grid = obs["grid"]
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        return self.forward_step(grid, step_idx)

    # Internal helpers
    def _encode_obs(self, obs_grid: torch.Tensor) -> torch.Tensor:
        """Encode (B, C, H, W) grid into (B, Hs, Ws, D) external input."""
        bsz, _, h_obs, w_obs = obs_grid.shape
        encoded = self.obs_encoder(obs_grid)  # (B, D, h_obs, w_obs)
        encoded = encoded.permute(0, 2, 3, 1)  # (B, h_obs, w_obs, D)

        if h_obs == self.sheet.height and w_obs == self.sheet.width:
            return encoded

        # Center-pad or crop to match sheet size.
        target = torch.zeros(
            (bsz, self.sheet.height, self.sheet.width, self.sheet.content_dim),
            device=self.device,
        )
        start_y = max((self.sheet.height - h_obs) // 2, 0)
        start_x = max((self.sheet.width - w_obs) // 2, 0)
        end_y = min(start_y + h_obs, self.sheet.height)
        end_x = min(start_x + w_obs, self.sheet.width)
        slice_h = end_y - start_y
        slice_w = end_x - start_x
        target[:, start_y:end_y, start_x:end_x, :] = encoded[:, :slice_h, :slice_w, :]
        return target
