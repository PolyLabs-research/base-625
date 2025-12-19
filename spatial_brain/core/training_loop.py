"""Training utilities for spatial brain experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn

from spatial_brain.core.control_1d import ControlNetwork1D
from spatial_brain.core.cortical_line import CorticalLine
from spatial_brain.core.control_2d import ControlNetwork2D
from spatial_brain.core.cortical_sheet_2d import CorticalSheet2D
from spatial_brain.core.tasks import ThreeSlotBatch, ThreeSlotWMTask
from spatial_brain.core.utils import create_device, get_logger, set_seed


class TrainingOutput(dict):
    """Dictionary-like training output that also supports tuple unpacking."""

    def __iter__(self):
        # Enables backward-compat tuple unpacking of models.
        ordered_keys = []
        if "cortical_line" in self:
            ordered_keys.append("cortical_line")
        if "cortical_sheet" in self:
            ordered_keys.append("cortical_sheet")
        ordered_keys.extend(["control_net", "task", "readout", "slot_attention"])
        for k in ordered_keys:
            if k in self:
                yield self[k]
@dataclass
class TrainingConfig:
    """Configuration for training the three-slot WM model."""

    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    num_columns: int = 32
    content_dim: int = 32
    num_items: int = 10
    sequence_length: int = 3
    seed: int = 1
    prefer_cuda: bool = True


@dataclass
class TrainingConfig2D:
    """Configuration for training the three-slot WM model on a 2D sheet."""

    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    height: int = 16
    width: int = 16
    content_dim: int = 32
    num_items: int = 10
    sequence_length: int = 3
    seed: int = 1
    prefer_cuda: bool = True
    beta_smooth_lambda: float = 0.0


InputEncoderFn = Callable[[torch.Tensor, int], torch.Tensor]


def _compute_readout(
    final_state: torch.Tensor,
    slot_attention: torch.Tensor,
    readout: nn.Linear,
) -> torch.Tensor:
    """Pool spatial state into slot logits."""
    # slot_attention: (slots, num_columns)
    weights = torch.softmax(slot_attention, dim=1)
    # Expand to (batch, slots, num_columns, 1)
    weighted = weights.unsqueeze(0).unsqueeze(-1) * final_state.unsqueeze(1)
    pooled = weighted.sum(dim=2)  # (batch, slots, content_dim)
    logits = readout(pooled)  # (batch, slots, num_items)
    return logits


def _compute_readout_2d(
    final_state: torch.Tensor,
    slot_attention: torch.Tensor,
    readout: nn.Linear,
) -> torch.Tensor:
    """Pool spatial state from a 2D sheet into slot logits."""
    # slot_attention: (slots, H, W)
    weights = torch.softmax(slot_attention.view(slot_attention.shape[0], -1), dim=1)
    weights = weights.view_as(slot_attention)
    # (batch, slots, H, W, 1)
    weighted = weights.unsqueeze(0).unsqueeze(-1) * final_state.unsqueeze(1)
    pooled = weighted.sum(dim=3).sum(dim=2)  # (batch, slots, content_dim)
    logits = readout(pooled)
    return logits


def train_three_slot_wm(
    config: TrainingConfig,
) -> Dict[str, object]:
    """Train a spatial WM model with beta/gamma control fields.

    Returns a dictionary with trained modules and history.
    """
    set_seed(config.seed)
    device = create_device(prefer_cuda=config.prefer_cuda)
    logger = get_logger("spatial_brain.train")

    task = ThreeSlotWMTask(
        num_items=config.num_items,
        sequence_length=config.sequence_length,
        num_columns=config.num_columns,
        content_dim=config.content_dim,
    )
    task.item_embeddings.to(device)

    cortical = CorticalLine(
        num_columns=config.num_columns,
        content_dim=config.content_dim,
        device=device,
    )
    control = ControlNetwork1D(
        num_columns=config.num_columns,
        context_dim=task.context_dim,
        max_steps=config.sequence_length + 1,
        device=device,
    )

    readout = nn.Linear(config.content_dim, config.num_items, device=device)
    slot_attention = nn.Parameter(torch.randn(config.sequence_length, config.num_columns, device=device))

    optimizer = torch.optim.Adam(
        [
            {"params": cortical.parameters()},
            {"params": control.parameters()},
            {"params": readout.parameters()},
            {"params": [slot_attention]},
            {"params": task.item_embeddings.parameters()},
        ],
        lr=config.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    train_losses: list[float] = []
    train_accs: list[float] = []

    for epoch in range(config.num_epochs):
        batch = task.sample_batch(config.batch_size)
        state = torch.zeros(
            (config.batch_size, config.num_columns, config.content_dim), device=device
        )
        loss_accum = 0.0
        correct = 0
        total = 0

        for t in range(batch.inputs.shape[0]):
            context_t = batch.contexts[t].to(device)
            external_input = batch.inputs[t].to(device)
            beta, gamma = control(context_t, t)
            state = cortical(state, beta, gamma, external_input=external_input)

        logits = _compute_readout(state, slot_attention, readout)
        targets = batch.targets.to(device)
        # Compute loss per slot then mean.
        loss = 0.0
        for s in range(config.sequence_length):
            loss += criterion(logits[:, s, :], targets[:, s])
            preds = logits[:, s, :].argmax(dim=-1)
            correct += (preds == targets[:, s]).sum().item()
            total += targets.shape[0]
        loss = loss / config.sequence_length
        loss_accum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = correct / total if total > 0 else 0.0
        logger.info("Epoch %d | loss=%.4f | acc=%.3f", epoch + 1, loss_accum, acc)
        train_losses.append(loss_accum)
        train_accs.append(acc)

    return TrainingOutput({
        "cortical_line": cortical,
        "control_net": control,
        "task": task,
        "readout": readout,
        "slot_attention": slot_attention,
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": [],
        "val_acc": [],
    })


def train_three_slot_wm_2d(
    height: int | TrainingConfig2D,
    width: int | None = None,
    content_dim: int | None = None,
    num_items: int | None = None,
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    beta_smooth_lambda: float = 0.0,
    device: torch.device | None = None,
    input_encoder: Optional[InputEncoderFn] = None,
) -> Dict[str, Any]:
    """Train a spatial WM model on a 2D cortical sheet.

    Accepts either a TrainingConfig2D as the first argument (backward compatible)
    or explicit parameters matching the CLI specification.
    """

    # Backward-compatible config handling
    if isinstance(height, TrainingConfig2D):
        cfg = height
        h, w = cfg.height, cfg.width
        content = cfg.content_dim
        items = cfg.num_items
        epochs = cfg.num_epochs
        bsz = cfg.batch_size
        lr = cfg.learning_rate
        seed = cfg.seed
        # Respect explicit override; otherwise use config value.
        if beta_smooth_lambda != 0.0:
            lambda_smooth = beta_smooth_lambda
        else:
            lambda_smooth = cfg.beta_smooth_lambda
        prefer_cuda = cfg.prefer_cuda
    else:
        cfg = None
        h = height
        w = width if width is not None else 1
        content = content_dim if content_dim is not None else 32
        items = num_items if num_items is not None else 10
        epochs = num_epochs
        bsz = batch_size
        lr = learning_rate
        seed = 1
        lambda_smooth = beta_smooth_lambda
        prefer_cuda = True

    set_seed(seed)
    if device is None:
        device = create_device(prefer_cuda=prefer_cuda)
    logger = get_logger("spatial_brain.train2d")

    num_columns = h * w
    sequence_length = 3 if cfg is None else cfg.sequence_length
    task = ThreeSlotWMTask(
        num_items=items,
        sequence_length=sequence_length,
        num_columns=num_columns,
        content_dim=content,
    )
    task.item_embeddings.to(device)

    cortical = CorticalSheet2D(
        height=h,
        width=w,
        content_dim=content,
        device=device,
    )
    control = ControlNetwork2D(
        height=h,
        width=w,
        context_dim=task.context_dim,
        max_steps=sequence_length + 1,
        device=device,
    )

    readout = nn.Linear(content, items, device=device)
    slot_attention = nn.Parameter(torch.randn(sequence_length, h, w, device=device))

    optimizer = torch.optim.Adam(
        [
            {"params": cortical.parameters()},
            {"params": control.parameters()},
            {"params": readout.parameters()},
            {"params": [slot_attention]},
            {"params": task.item_embeddings.parameters()},
        ],
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()
    train_losses: list[float] = []
    train_accs: list[float] = []
    val_losses: list[float] = []
    val_accs: list[float] = []
    smooth_hist: list[float] = []

    for epoch in range(epochs):
        batch = task.sample_batch(bsz)
        state = torch.zeros((bsz, h, w, content), device=device)
        correct = 0
        total = 0
        beta_seq = []
        gamma_seq = []

        for t in range(batch.inputs.shape[0]):
            context_t = batch.contexts[t].to(device)
            if input_encoder is not None and t < batch.item_indices.shape[1]:
                item_ids_t = batch.item_indices[:, t].to(device)
                external_input = input_encoder(item_ids_t, t).to(device)
            else:
                external_input = batch.inputs[t].to(device).view(bsz, h, w, content)
            beta, gamma = control(context_t, t)
            beta_seq.append(beta)
            gamma_seq.append(gamma)
            state = cortical(state, beta, gamma, external_input=external_input)

        logits = _compute_readout_2d(state, slot_attention, readout)
        targets = batch.targets.to(device)
        ce_loss = torch.tensor(0.0, device=device)
        for s in range(sequence_length):
            ce_loss = ce_loss + criterion(logits[:, s, :], targets[:, s])
            preds = logits[:, s, :].argmax(dim=-1)
            correct += (preds == targets[:, s]).sum().item()
            total += targets.shape[0]
        ce_loss = ce_loss / sequence_length

        if lambda_smooth > 0.0:
            smooth_terms = []
            for beta in beta_seq:
                dx = beta[:, 1:, :] - beta[:, :-1, :]
                dy = beta[:, :, 1:] - beta[:, :, :-1]
                smooth_terms.append((dx.pow(2).mean() + dy.pow(2).mean()))
            beta_smooth_loss = torch.stack(smooth_terms).mean()
        else:
            beta_smooth_loss = torch.tensor(0.0, device=device)

        total_loss = ce_loss + lambda_smooth * beta_smooth_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        acc = correct / total if total > 0 else 0.0
        logger.info(
            "Epoch %d | loss=%.4f | acc=%.3f | smooth=%.4f",
            epoch + 1,
            total_loss.item(),
            acc,
            beta_smooth_loss.item(),
        )
        train_losses.append(total_loss.item())
        train_accs.append(acc)
        if lambda_smooth > 0.0:
            smooth_hist.append(beta_smooth_loss.item())

    # Optional single validation point using final batch/state.
    if beta_seq and gamma_seq:
        with torch.no_grad():
            logits = _compute_readout_2d(state, slot_attention, readout)
            targets = batch.targets.to(device)
            val_loss = criterion(logits.view(-1, items), targets.view(-1)).item()
            preds = logits.argmax(dim=-1)
            val_acc = (preds == targets).float().mean().item()
            val_losses.append(val_loss)
            val_accs.append(val_acc)

    return TrainingOutput({
        "cortical_sheet": cortical,
        "control_net": control,
        "task": task,
        "readout": readout,
        "slot_attention": slot_attention,
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs,
        "train_beta_smooth": smooth_hist,
    })


def run_single_trial_2d(
    cortical_sheet: torch.nn.Module,
    control_net: torch.nn.Module,
    task: ThreeSlotWMTask,
    device: torch.device,
    input_encoder: Optional[InputEncoderFn] = None,
    external_input_override: Callable[[int, ThreeSlotBatch], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a single trial (batch_size=1) and return beta/gamma/content sequences."""

    cortical_sheet.eval()
    control_net.eval()
    task.item_embeddings.to(device)
    with torch.no_grad():
        batch = task.sample_batch(1)
        h = cortical_sheet.height
        w = cortical_sheet.width
        d = cortical_sheet.content_dim
        state = torch.zeros((1, h, w, d), device=device)
        beta_seq = []
        gamma_seq = []
        content_seq = []
        for t in range(batch.inputs.shape[0]):
            context_t = batch.contexts[t].to(device)
            if input_encoder is not None and t < batch.item_indices.shape[1]:
                item_ids_t = batch.item_indices[:, t].to(device)
                external_input = input_encoder(item_ids_t, t).to(device)
            else:
                external_input = batch.inputs[t].to(device).view(1, h, w, d)
            if external_input_override is not None:
                override = external_input_override(t, batch)
                if override is not None:
                    external_input = override.to(device)
            beta, gamma = control_net(context_t, t)
            state = cortical_sheet(state, beta, gamma, external_input=external_input)
            beta_seq.append(beta[0])
            gamma_seq.append(gamma[0])
            content_seq.append(state[0])

    beta_stack = torch.stack(beta_seq)
    gamma_stack = torch.stack(gamma_seq)
    content_stack = torch.stack(content_seq)
    return beta_stack, gamma_stack, content_stack
