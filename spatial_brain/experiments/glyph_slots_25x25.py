"""Glyph-focused 3-slot WM experiment on a 25x25 cortical sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from spatial_brain.core.training_loop import (
    InputEncoderFn,
    TrainingConfig2D,
    run_single_trial_2d,
    train_three_slot_wm_2d,
)
from spatial_brain.core.utils import create_device, get_sheet_config
from spatial_brain.integration.base625_integration import build_glyph_bank
from spatial_brain.viz.plot_fields import plot_field_2d, plot_field_sequence_2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise glyph slots on a 25x25 cortical sheet.")
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--beta-smooth-lambda",
        type=float,
        default=0.02,
        help="Weight of spatial smoothness regularisation on the beta field.",
    )
    parser.add_argument("--num-items", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=200)
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        choices=["easy", "hard"],
        help="Glyph task difficulty: easy=3 glyphs, hard=num-items glyphs.",
    )
    parser.add_argument("--plots-dir", type=str, default="spatial_brain/plots/glyph_25x25")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = create_device()
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    sheet_cfg = get_sheet_config("glyph")
    height, width, content_dim = sheet_cfg.height, sheet_cfg.width, sheet_cfg.content_dim
    num_items = 3 if args.difficulty == "easy" else args.num_items

    glyph_bank = build_glyph_bank(num_items, height, width, difficulty=args.difficulty)

    def make_glyph_encoder(bank_np: np.ndarray, encoder_device: torch.device) -> InputEncoderFn:
        bank = torch.from_numpy(bank_np).to(encoder_device)

        def encoder(item_ids_t: torch.Tensor, t: int) -> torch.Tensor:
            patterns = bank[item_ids_t]  # (B, H, W)
            expanded = patterns.unsqueeze(-1).expand(-1, height, width, content_dim)
            return expanded

        return encoder

    glyph_encoder = make_glyph_encoder(glyph_bank, device)

    train_config = TrainingConfig2D(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        height=height,
        width=width,
        content_dim=content_dim,
        num_items=num_items,
        sequence_length=3,
        seed=args.seed,
        prefer_cuda=device.type == "cuda",
        beta_smooth_lambda=args.beta_smooth_lambda,
    )

    result = train_three_slot_wm_2d(
        train_config,
        device=device,
        input_encoder=glyph_encoder,
    )
    cortical = result["cortical_sheet"]
    control = result["control_net"]
    task = result["task"]

    preview_count = min(3, num_items)
    preview_titles = [f"Glyph {idx}" for idx in range(preview_count)]
    plot_field_sequence_2d(
        glyph_bank[:preview_count],
        n_cols=preview_count,
        titles=preview_titles,
        title_prefix="Glyph",
        out_path=str(plots_dir / "glyph_examples.png"),
    )

    slot_sums = [torch.zeros((height, width), device=device) for _ in range(task.sequence_length)]
    for _ in range(args.num_trials):
        beta_seq, _, _ = run_single_trial_2d(
            cortical,
            control,
            task,
            device,
            input_encoder=glyph_encoder,
        )
        for slot_idx in range(task.sequence_length):
            slot_sums[slot_idx] += beta_seq[slot_idx].to(device)

    slot_avgs = [s / max(args.num_trials, 1) for s in slot_sums]
    plot_field_2d(
        slot_avgs[0],
        "Glyph avg beta slot 1",
        out_path=str(plots_dir / "glyph_avg_beta_slot1.png"),
    )
    plot_field_2d(
        slot_avgs[1],
        "Glyph avg beta slot 2",
        out_path=str(plots_dir / "glyph_avg_beta_slot2.png"),
    )
    plot_field_2d(
        slot_avgs[2],
        "Glyph avg beta slot 3",
        out_path=str(plots_dir / "glyph_avg_beta_slot3.png"),
    )

    print(f"Saved 25x25 glyph slot plots to: {plots_dir}")


if __name__ == "__main__":
    main()
