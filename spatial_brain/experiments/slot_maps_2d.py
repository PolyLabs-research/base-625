"""Estimate average beta maps per working-memory slot on a 2D cortical sheet."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from spatial_brain.core.tasks import ThreeSlotWMTask
from spatial_brain.core.training_loop import run_single_trial_2d, train_three_slot_wm_2d
from spatial_brain.core.utils import create_device
from spatial_brain.viz.plot_fields import plot_field_2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute average beta slot maps for 2D WM model.")
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--content-dim", type=int, default=32)
    parser.add_argument("--num-items", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta-smooth-lambda", type=float, default=0.0)
    parser.add_argument("--num-trials", type=int, default=200)
    parser.add_argument("--plots-dir", type=str, default="spatial_brain/plots/slot_maps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = create_device()

    result = train_three_slot_wm_2d(
        height=args.height,
        width=args.width,
        content_dim=args.content_dim,
        num_items=args.num_items,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta_smooth_lambda=args.beta_smooth_lambda,
        device=device,
    )
    cortical = result["cortical_sheet"]
    control = result["control_net"]
    task = result["task"]

    slot_sums = [torch.zeros((args.height, args.width), device=device) for _ in range(3)]
    for _ in range(args.num_trials):
        beta_seq, gamma_seq, content_seq = run_single_trial_2d(cortical, control, task, device)
        slot_sums[0] += beta_seq[0]
        slot_sums[1] += beta_seq[1]
        slot_sums[2] += beta_seq[2]

    slot_avgs = [s / max(args.num_trials, 1) for s in slot_sums]

    plots_dir = Path(args.plots_dir)
    os.makedirs(plots_dir, exist_ok=True)
    plot_field_2d(slot_avgs[0], "Avg beta slot 1", out_path=str(plots_dir / "avg_beta_slot1.png"))
    plot_field_2d(slot_avgs[1], "Avg beta slot 2", out_path=str(plots_dir / "avg_beta_slot2.png"))
    plot_field_2d(slot_avgs[2], "Avg beta slot 3", out_path=str(plots_dir / "avg_beta_slot3.png"))
    print(f"Saved slot maps to {plots_dir}")


if __name__ == "__main__":
    main()
