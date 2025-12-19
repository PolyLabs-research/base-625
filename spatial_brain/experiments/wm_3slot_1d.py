"""Runnable experiment for 3-slot working memory on a 1D cortical line."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from spatial_brain.core.training_loop import TrainingConfig, _compute_readout, train_three_slot_wm


def evaluate(cortical, control, task, readout, slot_attention, config: TrainingConfig) -> tuple[float, float]:
    """Run a small evaluation batch."""
    device = next(cortical.parameters()).device
    batch = task.sample_batch(config.batch_size)
    state = torch.zeros(
        (config.batch_size, config.num_columns, config.content_dim), device=device
    )

    # We do not train during evaluation; this is only a quick diagnostic.
    with torch.no_grad():
        for t in range(batch.inputs.shape[0]):
            beta, gamma = control(batch.contexts[t].to(device), t)
            state = cortical(state, beta, gamma, external_input=batch.inputs[t].to(device))
        logits = _compute_readout(state, slot_attention, readout)
        targets = batch.targets.to(device)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.num_items), targets.view(-1), reduction="mean"
        )
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
    return loss.item(), acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3-slot WM on a 1D cortical line.")
    parser.add_argument("--num-columns", type=int, default=32)
    parser.add_argument("--content-dim", type=int, default=32)
    parser.add_argument("--num-items", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-path", type=str, default=None, help="Optional checkpoint path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_columns=args.num_columns,
        content_dim=args.content_dim,
        num_items=args.num_items,
        sequence_length=3,
        seed=args.seed,
    )

    result = train_three_slot_wm(config)
    cortical = result["cortical_line"]
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
