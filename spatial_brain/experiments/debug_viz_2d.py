"""Debug visualization for 2D cortical sheet beta/gamma and content dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from spatial_brain.core.training_loop import TrainingConfig2D, run_single_trial_2d, train_three_slot_wm_2d
from spatial_brain.core.utils import get_sheet_config
from spatial_brain.viz.plot_fields import (
    plot_beta_gamma_over_time_2d,
    plot_content_norm_over_time_2d,
    plot_training_curves,
    plot_avg_beta_heatmap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug visualizations for 2D spatial brain.")
    parser.add_argument(
        "--sheet-preset",
        type=str,
        default="small",
        choices=["small", "medium", "glyph"],
        help="Preset for cortical sheet size.",
    )
    parser.add_argument("--height", type=int, default=None, help="Override preset height.")
    parser.add_argument("--width", type=int, default=None, help="Override preset width.")
    parser.add_argument("--content-dim", type=int, default=None, help="Override preset content dim.")
    parser.add_argument("--num-items", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--beta-smooth-lambda",
        type=float,
        default=0.02,
        help="Weight of spatial smoothness regularisation on the beta field.",
    )
    parser.add_argument("--num-trials", type=int, default=64, help="Trials for beta slot averaging")
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="spatial_brain/plots",
        help="Directory to save PNG plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plots_dir = Path(args.plots_dir)

    preset_cfg = get_sheet_config(args.sheet_preset)
    height = args.height if args.height is not None else preset_cfg.height
    width = args.width if args.width is not None else preset_cfg.width
    content_dim = args.content_dim if args.content_dim is not None else preset_cfg.content_dim

    config = TrainingConfig2D(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        height=height,
        width=width,
        content_dim=content_dim,
        num_items=args.num_items,
        sequence_length=3,
        seed=args.seed,
        beta_smooth_lambda=args.beta_smooth_lambda,
    )

    result = train_three_slot_wm_2d(config)
    cortical = result["cortical_sheet"]
    control = result["control_net"]
    task = result["task"]

    # Plot training curves
    plot_training_curves(
        result["train_loss"],
        result["train_acc"],
        result.get("val_loss", []),
        result.get("val_acc", []),
        out_path=str(plots_dir / "training_curves.png"),
    )

    # Run a single trial with batch=1 to capture fields.
    device = next(cortical.parameters()).device
    beta_seq, gamma_seq, state_seq = run_single_trial_2d(cortical, control, task, device)

    plots_dir.mkdir(parents=True, exist_ok=True)
    phase_names = {
        0: "encode slot 1",
        1: "encode slot 2",
        2: "encode slot 3",
    }
    beta_titles = []
    gamma_titles = []
    time_steps = beta_seq.shape[0]
    for t in range(time_steps):
        phase_label = phase_names.get(t, "delay/recall")
        beta_titles.append(f"Beta t={t} ({phase_label})")
        gamma_titles.append(f"Gamma t={t} ({phase_label})")

    plot_beta_gamma_over_time_2d(
        beta_seq,
        gamma_seq,
        beta_titles=beta_titles,
        gamma_titles=gamma_titles,
        out_path=str(plots_dir / "beta_gamma.png"),
    )
    plot_content_norm_over_time_2d(state_seq, out_path=str(plots_dir / "content_norm.png"))

    # Slot-consistency analysis: average beta at encoding steps for many trials.
    slot_sums = [torch.zeros((config.height, config.width), device=device) for _ in range(config.sequence_length)]
    for _ in range(args.num_trials):
        beta_seq, _, _ = run_single_trial_2d(cortical, control, task, device)
        for t in range(config.sequence_length):
            slot_sums[t] += beta_seq[t]
    slot_avgs = [s / max(args.num_trials, 1) for s in slot_sums]
    for idx, avg in enumerate(slot_avgs, start=1):
        plot_avg_beta_heatmap(avg, f"Avg beta slot {idx}", out_path=str(plots_dir / f"avg_beta_slot{idx}.png"))
    print(f"Saved diagnostic plots to {plots_dir}")


if __name__ == "__main__":
    main()
