"""Simple reinforcement learning loop for 2D navigation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from spatial_brain.core.utils import create_device, set_seed
from spatial_brain.nav2d.brain_agent import BrainAgent2D
from spatial_brain.nav2d.env import GridWorldEnv


@dataclass
class Trajectory:
    """Container for a single episode."""

    rewards: List[float]
    log_probs: List[torch.Tensor]
    values: List[torch.Tensor]
    entropies: List[torch.Tensor]
    actions: List[int]
    distances: List[float]
    success: bool

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))

    @property
    def length(self) -> int:
        return len(self.rewards)


@dataclass
class RLConfig:
    """Training hyperparameters and environment settings."""

    width: int = 9
    height: int = 9
    obstacle_density: float = 0.1
    max_steps: int = 50
    num_episodes: int = 500
    gamma: float = 0.99
    lr: float = 1e-3
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    seed: int = 1
    log_every: int = 10
    save_every: int = 100
    checkpoint_dir: str = "spatial_brain/episodes/nav2d/checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_agent(height: int, width: int, max_steps: int, device: torch.device) -> BrainAgent2D:
    """Construct a fresh BrainAgent2D matching the environment shape."""
    from spatial_brain.core.control_2d import ControlNetwork2D
    from spatial_brain.core.cortical_sheet_2d import CorticalSheet2D

    content_dim = 32
    sheet = CorticalSheet2D(height=height, width=width, content_dim=content_dim, device=device)
    control = ControlNetwork2D(
        height=height,
        width=width,
        context_dim=4,
        max_steps=max_steps + 1,
        device=device,
    )
    return BrainAgent2D(sheet=sheet, control=control, action_dim=4, device=device)


def discounted_returns(rewards: List[float], gamma: float, device: torch.device) -> torch.Tensor:
    """Compute discounted returns for a list of rewards."""
    returns = torch.zeros(len(rewards), device=device)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        returns[i] = running
    return returns


def collect_episode(env: GridWorldEnv, agent: BrainAgent2D, max_steps: int) -> Trajectory:
    """Roll out a single episode with stochastic policy actions."""
    obs = env.reset()
    agent.reset_state(batch_size=1)

    rewards: List[float] = []
    log_probs: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    actions: List[int] = []
    dists: List[float] = []

    success = False
    for _ in range(max_steps):
        grid = obs["grid"]
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        step_idx = int(obs.get("step", 0))
        out = agent.forward_step(grid, step_idx, sample=True)
        action = int(out["action"].item())
        log_prob = out["log_prob"]
        value = out["value"]
        entropy = out["entropy"]

        next_obs, reward, done, _ = env.step(action)

        rewards.append(float(reward))
        log_probs.append(log_prob.squeeze())
        values.append(value.squeeze())
        entropies.append(entropy.squeeze())
        actions.append(action)
        dists.append(env.distance_to_goal())

        obs = next_obs
        if done:
            success = env.agent_pos == env.goal_pos
            break

    return Trajectory(
        rewards=rewards,
        log_probs=log_probs,
        values=values,
        entropies=entropies,
        actions=actions,
        distances=dists,
        success=success,
    )


def update_policy(
    agent: BrainAgent2D,
    optimizer: torch.optim.Optimizer,
    trajectory: Trajectory,
    config: RLConfig,
) -> Dict[str, float]:
    """Compute actor-critic losses and apply a gradient step."""
    device = next(agent.parameters()).device
    returns = discounted_returns(trajectory.rewards, config.gamma, device)
    log_probs = torch.stack(trajectory.log_probs)
    values = torch.stack(trajectory.values)
    entropies = torch.stack(trajectory.entropies)

    advantages = returns - values.detach()
    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_bonus = entropies.mean()

    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy_bonus.item()),
    }


def train(config: RLConfig) -> Dict[str, Any]:
    """Main training loop returning history and final agent."""
    set_seed(config.seed)
    device = create_device(prefer_cuda=(config.device == "cuda"))
    env = GridWorldEnv(
        width=config.width,
        height=config.height,
        obstacle_density=config.obstacle_density,
        max_steps=config.max_steps,
        device=device,
    )
    agent = build_agent(config.height, config.width, config.max_steps, device=device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.lr)

    history: List[Dict[str, Any]] = []
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(1, config.num_episodes + 1):
        traj = collect_episode(env, agent, config.max_steps)
        losses = update_policy(agent, optimizer, traj, config)

        metrics = {
            "episode": episode,
            "reward": traj.total_reward,
            "length": traj.length,
            "success": traj.success,
            "mean_distance": float(sum(traj.distances) / len(traj.distances)) if traj.distances else 0.0,
        }
        metrics.update(losses)
        history.append(metrics)

        if episode % config.log_every == 0:
            success_rate = sum(m["success"] for m in history[-config.log_every :]) / float(
                min(len(history), config.log_every)
            )
            print(
                f"[ep {episode:04d}] reward={traj.total_reward:.3f} "
                f"len={traj.length} success={traj.success} "
                f"success_rate(last {config.log_every})={success_rate:.2f} "
                f"loss={losses['loss']:.4f}"
            )

        if config.save_every and episode % config.save_every == 0:
            ckpt_path = checkpoint_dir / f"nav2d_ep{episode:05d}.pt"
            torch.save(
                {
                    "agent_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.__dict__,
                    "episode": episode,
                    "history": history,
                },
                ckpt_path,
            )

    return {"agent": agent, "optimizer": optimizer, "history": history}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BrainAgent2D with actor-critic on GridWorld.")
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--obstacle-density", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="spatial_brain/episodes/nav2d/checkpoints",
        help="Where to save checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RLConfig(
        width=args.width,
        height=args.height,
        obstacle_density=args.obstacle_density,
        max_steps=args.max_steps,
        num_episodes=args.num_episodes,
        gamma=args.gamma,
        lr=args.lr,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    train(config)


if __name__ == "__main__":
    main()
