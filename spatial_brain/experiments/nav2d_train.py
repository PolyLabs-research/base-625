"""CLI entry point for training a BrainAgent2D on the GridWorld task."""

from __future__ import annotations

import argparse
from pathlib import Path

from spatial_brain.nav2d.train_rl import RLConfig, train


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train BrainAgent2D with actor-critic on GridWorld.")
    parser.add_argument("--width", type=int, default=9, help="Width of the grid world.")
    parser.add_argument("--height", type=int, default=9, help="Height of the grid world.")
    parser.add_argument(
        "--obstacles",
        type=float,
        default=0.15,
        help="Obstacle density in the environment.",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Total number of episodes to train.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for returns.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="spatial_brain/checkpoints/nav2d/",
        help="Directory to save model checkpoints.",
    )
    # Arguments that are in RLConfig but not explicitly requested in the prompt are added here
    # for completeness, using the defaults from RLConfig.
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode.")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Coefficient for the value loss.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Coefficient for the entropy bonus.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--log-every", type=int, default=10, help="Log progress every N episodes.")
    parser.add_argument("--save-every", type=int, default=100, help="Save a checkpoint every N episodes.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="Device to use for training (cuda or cpu).",
    )
    return parser.parse_args()


def main() -> None:
    """Set up and run the RL training loop."""
    args = parse_args()

    # Ensure the checkpoint directory exists
    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    config = RLConfig(
        width=args.width,
        height=args.height,
        obstacle_density=args.obstacles,
        max_steps=args.max_steps,
        num_episodes=args.episodes,
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

    print("Starting training with configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")

    train(config)

    print("Training finished.")


if __name__ == "__main__":
    main()
