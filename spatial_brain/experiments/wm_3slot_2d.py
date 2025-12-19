"""Runnable experiment for 3-slot working memory on a 2D cortical sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from spatial_brain.core.training_loop import (
    TrainingConfig2D,
    _compute_readout_2d,
    train_three_slot_wm_2d,
)
from spatial_brain.core.utils import SheetConfig, create_device, get_sheet_config


def evaluate(
    cortical,
    control,
    task,
    readout,
    slot_attention,
    config: TrainingConfig2D,
) -> tuple[float, float]:
    """Run a small evaluation batch."""
    device = next(cortical.parameters()).device
    batch = task.sample_batch(config.batch_size)
    state = torch.zeros(
        (config.batch_size, config.height, config.width, config.content_dim), device=device
    )

    with torch.no_grad():
        for t in range(batch.inputs.shape[0]):
            beta, gamma = control(batch.contexts[t].to(device), t)
            external_input = batch.inputs[t].to(device).view(
                config.batch_size, config.height, config.width, config.content_dim
            )
            state = cortical(state, beta, gamma, external_input=external_input)
        logits = _compute_readout_2d(state, slot_attention, readout)
        targets = batch.targets.to(device)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.num_items), targets.view(-1), reduction="mean"
        )
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
    return loss.item(), acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3-slot WM on a 2D cortical sheet.")
    parser.add_argument(
        "--sheet-preset",
        type=str,
        default="small",
        choices=["small", "medium", "glyph"],
        help="Preset for cortical sheet size.",
    )
    parser.add_argument("--height", type=int, default=None, help="Override preset height.")
    parser.add_argument("--width", type=int, default=None, help="Override preset width.")
    parser.add_argument(
        "--content-dim",
        type=int,
        default=None,
        help="Override preset content dimension.",
    )
    parser.add_argument("--num-items", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--beta-smooth-lambda",
        type=float,
        default=0.02,
        help="Weight of spatial smoothness regularisation on the beta field.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-path", type=str, default=None, help="Optional checkpoint path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset_cfg: SheetConfig = get_sheet_config(args.sheet_preset)
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
    device = create_device()
    result = train_three_slot_wm_2d(
        height=height,
        width=width,
        content_dim=content_dim,
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
    readout = result["readout"]
    slot_attention = result["slot_attention"]

    eval_loss, eval_acc = evaluate(cortical, control, task, readout, slot_attention, config)
    print(f"Final eval loss: {eval_loss:.4f}")
    print(f"Final eval accuracy: {eval_acc:.3f}")

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cortical_state_dict": cortical.state_dict(),
                "control_state_dict": control.state_dict(),
                "config": config.__dict__,
            },
            save_path,
        )
        print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
