"""Visualization helpers for 2D navigation experiments."""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

from spatial_brain.nav2d.env import GridCodes


def plot_grid_state(grid: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Render a grid with obstacles, agent, and goal."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    cmap = plt.cm.get_cmap("tab20")
    colors = {
        GridCodes.EMPTY: "#f7f7f7",
        GridCodes.OBSTACLE: "#4a4a4a",
        GridCodes.AGENT: cmap(2),
        GridCodes.GOAL: cmap(3),
    }
    color_grid = np.zeros((*grid.shape, 3))
    for code, color in colors.items():
        mask = grid == code
        color_grid[mask] = mcolors.to_rgb(color) if isinstance(color, str) else color[:3]

    ax.imshow(color_grid, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Environment")
    return ax


def plot_brain_fields(
    beta: np.ndarray,
    gamma: np.ndarray,
    content_norm: np.ndarray,
    ax_beta: Optional[plt.Axes] = None,
    ax_gamma: Optional[plt.Axes] = None,
    ax_norm: Optional[plt.Axes] = None,
) -> tuple[plt.Axes, plt.Axes, plt.Axes]:
    """Plot beta, gamma, and content norm heatmaps."""
    if ax_beta is None or ax_gamma is None or ax_norm is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        ax_beta, ax_gamma, ax_norm = axes
    im_beta = ax_beta.imshow(beta, origin="lower")
    ax_beta.set_title("Beta")
    plt.colorbar(im_beta, ax=ax_beta, fraction=0.046, pad=0.04)

    im_gamma = ax_gamma.imshow(gamma, origin="lower")
    ax_gamma.set_title("Gamma")
    plt.colorbar(im_gamma, ax=ax_gamma, fraction=0.046, pad=0.04)

    im_norm = ax_norm.imshow(content_norm, origin="lower")
    ax_norm.set_title("Content norm")
    plt.colorbar(im_norm, ax=ax_norm, fraction=0.046, pad=0.04)
    return ax_beta, ax_gamma, ax_norm


def episode_summary(log: Dict[str, Any]) -> Dict[str, float | int | bool]:
    """Compute simple episode statistics."""
    steps = log.get("log", [])
    dists = [s["dist_to_goal"] for s in steps]
    return {
        "success": bool(log.get("success", False)),
        "episode_steps": int(log.get("episode_steps", len(steps))),
        "total_reward": float(log.get("total_reward", 0.0)),
        "mean_distance": float(np.mean(dists)) if dists else float("nan"),
        "final_distance": float(dists[-1]) if dists else float("nan"),
    }
