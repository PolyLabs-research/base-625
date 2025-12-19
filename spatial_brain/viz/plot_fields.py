"""Visualization helpers for beta/gamma fields and content activity."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_beta_gamma_heatmaps(beta: torch.Tensor, gamma: torch.Tensor) -> None:
    """Plot beta and gamma fields over time as 1D heatmaps.

    Args:
        beta: Tensor of shape (time, batch, num_columns).
        gamma: Tensor of shape (time, batch, num_columns).
    """
    beta_np = beta.detach().cpu().numpy()
    gamma_np = gamma.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(beta_np[:, 0, :], aspect="auto", origin="lower")
    axes[0].set_title("Beta field (1D)")
    axes[0].set_xlabel("Columns")
    axes[0].set_ylabel("Time")

    axes[1].imshow(gamma_np[:, 0, :], aspect="auto", origin="lower")
    axes[1].set_title("Gamma field (1D)")
    axes[1].set_xlabel("Columns")
    axes[1].set_ylabel("Time")
    fig.tight_layout()
    plt.show()


def plot_trial_diagnostics(
    beta_seq: Sequence[torch.Tensor],
    gamma_seq: Sequence[torch.Tensor],
    state_seq: Sequence[torch.Tensor],
) -> None:
    """Plot beta/gamma fields and content norms across time for a single trial."""
    beta = torch.stack(beta_seq)  # (time, batch, num_columns)
    gamma = torch.stack(gamma_seq)
    plot_beta_gamma_heatmaps(beta, gamma)

    # Content norm per column over time.
    norms = torch.stack([s.norm(dim=-1) for s in state_seq])  # (time, batch, num_columns)
    norms_np = norms.detach().cpu().numpy()
    plt.figure(figsize=(5, 4))
    plt.imshow(norms_np[:, 0, :], aspect="auto", origin="lower")
    plt.title("Content activation norm")
    plt.xlabel("Columns")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.show()


def plot_beta_gamma_frames_2d(beta: torch.Tensor, gamma: torch.Tensor, time_index: int = 0) -> None:
    """Plot beta/gamma snapshots for a 2D sheet at a given time index."""
    beta_np = beta.detach().cpu().numpy()
    gamma_np = gamma.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(beta_np[time_index, 0])
    axes[0].set_title(f"Beta t={time_index}")
    axes[1].imshow(gamma_np[time_index, 0])
    axes[1].set_title(f"Gamma t={time_index}")
    fig.tight_layout()
    plt.show()


def plot_beta_gamma_heatmaps_2d(beta: torch.Tensor, gamma: torch.Tensor) -> None:
    """Plot beta/gamma over time by flattening spatial dims, for quick diagnostics."""
    t, b, h, w = beta.shape
    beta_flat = beta.view(t, b, h * w)
    gamma_flat = gamma.view(t, b, h * w)
    plot_beta_gamma_heatmaps(beta_flat, gamma_flat)


def plot_trial_diagnostics_2d(
    beta_seq: Sequence[torch.Tensor],
    gamma_seq: Sequence[torch.Tensor],
    state_seq: Sequence[torch.Tensor],
) -> None:
    """Plot 2D trial diagnostics: beta/gamma maps and content norms."""
    beta = torch.stack(beta_seq)  # (time, batch, H, W)
    gamma = torch.stack(gamma_seq)
    plot_beta_gamma_frames_2d(beta, gamma, time_index=0)

    norms = torch.stack([s.norm(dim=-1) for s in state_seq])  # (time, batch, H, W)
    norms_np = norms.detach().cpu().numpy()
    plt.figure(figsize=(6, 4))
    plt.imshow(norms_np[0, 0], origin="lower")
    plt.title("Content norm at t=0")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


def _maybe_save(fig: plt.Figure, out_path: str | None) -> None:
    if out_path:
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        fig.tight_layout()
        plt.show()


def plot_field_sequence_2d(
    fields: "np.ndarray | torch.Tensor | Sequence[torch.Tensor]",
    n_cols: int = 5,
    titles: list[str] | None = None,
    title_prefix: str = "Field",
    out_path: str | None = None,
) -> None:
    """Plot a sequence of 2D fields over time."""

    if isinstance(fields, (list, tuple)):
        if not fields:
            raise ValueError("fields sequence is empty")
        if hasattr(fields[0], "detach"):
            data = torch.stack(fields)
        else:
            data = np.stack(fields)
    else:
        data = fields
    if hasattr(data, "detach"):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = np.asarray(data)

    if data_np.ndim == 4:
        data_np = data_np[:, 0]
    if data_np.ndim != 3:
        raise ValueError(f"Expected data with shape (time, H, W); got {data_np.shape}")

    time_steps = data_np.shape[0]
    if time_steps == 0:
        raise ValueError("fields must contain at least one timestep")
    cols = min(max(n_cols, 1), time_steps)
    rows = math.ceil(time_steps / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(rows, cols)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx < time_steps:
                ax.imshow(data_np[idx], origin="lower")
                if titles is not None and idx < len(titles) and titles[idx]:
                    ax.set_title(titles[idx])
                else:
                    prefix = title_prefix or "t"
                    ax.set_title(f"{prefix} t={idx}")
            else:
                ax.axis("off")
            idx += 1

    _maybe_save(fig, out_path)


def plot_training_curves(
    train_loss: list[float],
    train_acc: list[float],
    val_loss: list[float] | None = None,
    val_acc: list[float] | None = None,
    out_path: str | None = None,
) -> None:
    """Plot loss and accuracy vs epoch; optionally save to a PNG."""
    epochs = range(1, len(train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train loss")
    if val_loss:
        axes[0].plot(epochs[: len(val_loss)], val_loss, label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train acc")
    if val_acc:
        axes[1].plot(epochs[: len(val_acc)], val_acc, label="val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    _maybe_save(fig, out_path)


def plot_beta_gamma_over_time_2d(
    beta_seq: Sequence[torch.Tensor] | torch.Tensor,
    gamma_seq: Sequence[torch.Tensor] | torch.Tensor,
    beta_titles: list[str] | None = None,
    gamma_titles: list[str] | None = None,
    out_path: str | None = None,
) -> None:
    """Plot 2D beta/gamma fields for each time step of a single trial."""
    beta = torch.stack(beta_seq) if isinstance(beta_seq, (list, tuple)) else beta_seq
    gamma = torch.stack(gamma_seq) if isinstance(gamma_seq, (list, tuple)) else gamma_seq
    time_steps = beta.shape[0]
    fig, axes = plt.subplots(2, time_steps, figsize=(3 * time_steps, 6))
    if time_steps == 1:
        axes = axes.reshape(2, 1)
    for t in range(time_steps):
        beta_slice = beta[t, 0] if beta.dim() == 4 else beta[t]
        gamma_slice = gamma[t, 0] if gamma.dim() == 4 else gamma[t]
        axes[0, t].imshow(beta_slice.detach().cpu().numpy(), origin="lower")
        if beta_titles is not None and t < len(beta_titles):
            axes[0, t].set_title(beta_titles[t])
        else:
            axes[0, t].set_title(f"Beta t={t}")
        axes[1, t].imshow(gamma_slice.detach().cpu().numpy(), origin="lower")
        if gamma_titles is not None and t < len(gamma_titles):
            axes[1, t].set_title(gamma_titles[t])
        else:
            axes[1, t].set_title(f"Gamma t={t}")
    _maybe_save(fig, out_path)


def plot_content_norm_over_time_2d(
    state_seq: Sequence[torch.Tensor] | torch.Tensor,
    out_path: str | None = None,
) -> None:
    """Plot content norm per location over time for a single trial."""
    states = torch.stack(state_seq) if isinstance(state_seq, (list, tuple)) else state_seq
    time_steps = states.shape[0]
    norms = torch.norm(states, dim=-1)  # (time, batch, H, W) or (time, H, W)
    fig, axes = plt.subplots(1, time_steps, figsize=(3 * time_steps, 3))
    if time_steps == 1:
        axes = [axes]
    for t in range(time_steps):
        norm_slice = norms[t, 0] if norms.dim() == 4 else norms[t]
        axes[t].imshow(norm_slice.detach().cpu().numpy(), origin="lower")
        axes[t].set_title(f"Norm t={t}")
    _maybe_save(fig, out_path)


def plot_avg_beta_heatmap(beta: torch.Tensor, title: str, out_path: str | None = None) -> None:
    """Plot a single beta heatmap (e.g., averaged over trials)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(beta.detach().cpu().numpy(), origin="lower")
    ax.set_title(title)
    _maybe_save(fig, out_path)


def plot_field_2d(field: torch.Tensor | "np.ndarray", title: str, out_path: str | None = None) -> None:
    """Plot a single 2D field (H, W) with a colorbar."""
    if hasattr(field, "detach"):
        data = field.detach().cpu().numpy()
    else:
        data = field
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(data, origin="lower")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    _maybe_save(fig, out_path)
