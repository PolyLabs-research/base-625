"""Rollout utilities for 2D navigation with the spatial brain."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from spatial_brain.nav2d.brain_agent import BrainAgent2D
from spatial_brain.nav2d.env import GridWorldEnv


def run_episode(
    env: GridWorldEnv,
    agent: BrainAgent2D,
    max_steps: Optional[int] = None,
    record_brain: bool = True,
) -> Dict[str, Any]:
    """Run a single episode and return a structured log."""
    obs = env.reset()
    agent.reset_state(batch_size=1)
    max_steps = max_steps or env.max_steps

    log_steps: list[Dict[str, Any]] = []
    total_reward = 0.0
    success = False

    for _ in range(max_steps):
        grid_batch = obs["grid"].unsqueeze(0) if obs["grid"].dim() == 3 else obs["grid"]
        result = agent.select_action(obs, obs["step"])
        action = int(result["action"].item())

        next_obs, reward, done, info = env.step(action)
        total_reward += reward

        beta = None
        gamma = None
        content_norm = None
        if record_brain:
            beta_val = result.get("beta")
            gamma_val = result.get("gamma")
            norm_val = result.get("content_norm")
            if beta_val is not None:
                beta = beta_val.detach().cpu().numpy()
            if gamma_val is not None:
                gamma = gamma_val.detach().cpu().numpy()
            if norm_val is not None:
                content_norm = norm_val.detach().cpu().numpy()

        log_steps.append(
            {
                "grid": env.render_array(),
                "action": action,
                "reward": reward,
                "done": done,
                "dist_to_goal": env.distance_to_goal(),
                "beta": beta,
                "gamma": gamma,
                "content_norm": content_norm,
                "info": info,
            }
        )

        obs = next_obs
        if done:
            success = env.agent_pos == env.goal_pos
            break

    episode_log: Dict[str, Any] = {
        "episode_steps": len(log_steps),
        "total_reward": total_reward,
        "success": success,
        "log": log_steps,
        "env_config": {
            "width": env.width,
            "height": env.height,
            "obstacle_density": env.obstacle_density,
            "max_steps": env.max_steps,
            "seed": env.last_seed,
        },
    }
    return episode_log


def save_episode_log(log: Dict[str, Any], path: Union[str, Path]) -> None:
    """Persist an episode log using torch serialization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(log, path)


def load_episode_log(path: Union[str, Path]) -> Dict[str, Any]:
    """Load an episode log saved by save_episode_log."""
    return torch.load(path, map_location="cpu")
