"""Simple 2D grid-world for navigation experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class GridCodes:
    EMPTY: int = 0
    OBSTACLE: int = 1
    AGENT: int = 2
    GOAL: int = 3


class GridWorldEnv:
    """Deterministic grid world with obstacles, agent, and goal."""

    def __init__(
        self,
        width: int = 9,
        height: int = 9,
        obstacle_density: float = 0.1,
        max_steps: int = 100,
        device: torch.device | str = "cpu",
    ) -> None:
        self.width = width
        self.height = height
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps
        self.device = torch.device(device)

        self.agent_pos: Tuple[int, int] | None = None
        self.goal_pos: Tuple[int, int] | None = None
        self.obstacles: set[Tuple[int, int]] = set()
        self.step_count = 0
        self.rng = np.random.default_rng()
        self.last_seed: Optional[int] = None

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Sample a new layout and return the initial observation."""
        if seed is not None:
            self.last_seed = seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self._sample_layout()
        return self._build_obs()

    def _sample_layout(self) -> None:
        """Place agent, goal, and obstacles."""
        self.obstacles.clear()
        free_cells = {(x, y) for x in range(self.width) for y in range(self.height)}

        # Choose agent and goal positions first.
        self.agent_pos = self._choice(list(free_cells))
        free_cells.remove(self.agent_pos)
        self.goal_pos = self._choice(list(free_cells))
        free_cells.remove(self.goal_pos)

        # Sample obstacles excluding border to keep start/goal reachable more often.
        interior = [(x, y) for x in range(1, self.width - 1) for y in range(1, self.height - 1)]
        interior = [cell for cell in interior if cell in free_cells]
        num_obstacles = int(len(interior) * self.obstacle_density)
        if num_obstacles > 0:
            obstacle_cells = self.rng.choice(len(interior), size=num_obstacles, replace=False)
            for idx in obstacle_cells:
                pos = interior[int(idx)]
                if pos not in (self.agent_pos, self.goal_pos):
                    self.obstacles.add(pos)

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Advance one step with action in {0:up,1:right,2:down,3:left}."""
        assert self.agent_pos is not None and self.goal_pos is not None, "Call reset() first."
        ax, ay = self.agent_pos
        proposed = {
            0: (ax, ay - 1),  # up
            1: (ax + 1, ay),  # right
            2: (ax, ay + 1),  # down
            3: (ax - 1, ay),  # left
        }.get(int(action), (ax, ay))

        reward = -0.01
        bumped = False

        if self._is_blocked(proposed):
            proposed = (ax, ay)
            bumped = True
        self.agent_pos = proposed

        done = False
        if self.agent_pos == self.goal_pos:
            reward += 1.0
            done = True
        if bumped:
            reward -= 0.1

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        obs = self._build_obs()
        info: Dict[str, Any] = {"bumped": bumped}
        return obs, reward, done, info

    def render_array(self) -> np.ndarray:
        """Return integer-coded grid of shape (H, W)."""
        grid = np.full((self.height, self.width), GridCodes.EMPTY, dtype=np.int32)
        for x, y in self.obstacles:
            grid[y, x] = GridCodes.OBSTACLE
        if self.agent_pos:
            ax, ay = self.agent_pos
            grid[ay, ax] = GridCodes.AGENT
        if self.goal_pos:
            gx, gy = self.goal_pos
            grid[gy, gx] = GridCodes.GOAL
        return grid

    def distance_to_goal(self) -> float:
        """Manhattan distance from agent to goal."""
        if self.agent_pos is None or self.goal_pos is None:
            return float("inf")
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        return float(abs(ax - gx) + abs(ay - gy))

    # Helpers
    def _choice(self, seq: list[Tuple[int, int]]) -> Tuple[int, int]:
        idx = self.rng.integers(low=0, high=len(seq))
        return seq[int(idx)]

    def _is_blocked(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        if pos in self.obstacles:
            return True
        return False

    def _build_obs(self) -> Dict[str, Any]:
        grid_tensor = self._grid_to_tensor()
        obs = {
            "grid": grid_tensor,
            "step": self.step_count,
            "dist_to_goal": self.distance_to_goal(),
            "meta": {
                "width": self.width,
                "height": self.height,
                "obstacle_density": self.obstacle_density,
                "max_steps": self.max_steps,
                "seed": self.last_seed,
            },
        }
        return obs

    def _grid_to_tensor(self) -> torch.Tensor:
        """Return (C, H, W) float tensor with agent/goal/obstacle channels."""
        grid = np.zeros((3, self.height, self.width), dtype=np.float32)
        if self.agent_pos:
            ax, ay = self.agent_pos
            grid[0, ay, ax] = 1.0
        if self.goal_pos:
            gx, gy = self.goal_pos
            grid[1, gy, gx] = 1.0
        for x, y in self.obstacles:
            grid[2, y, x] = 1.0
        return torch.from_numpy(grid).to(self.device)

