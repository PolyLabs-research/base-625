"""Utility helpers for spatial brain experiments."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_device(prefer_cuda: bool = True) -> torch.device:
    """Return a torch.device, preferring CUDA when available."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_tensor(name: str, tensor: torch.Tensor) -> str:
    """Return a short string describing a tensor."""
    return f"{name}: shape={tuple(tensor.shape)}, device={tensor.device}, dtype={tensor.dtype}"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Create a simple logger with a standard format."""
    logger = logging.getLogger(name or __name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@dataclass(frozen=True)
class SheetConfig:
    """Preset describing a cortical sheet."""

    height: int
    width: int
    content_dim: int


def get_sheet_config(preset: str) -> SheetConfig:
    """Return a standard sheet size/content configuration for a preset."""
    preset = preset.lower()
    if preset == "small":
        return SheetConfig(height=8, width=8, content_dim=32)
    if preset == "medium":
        return SheetConfig(height=16, width=16, content_dim=64)
    if preset == "glyph":
        return SheetConfig(height=25, width=25, content_dim=64)
    raise ValueError(f"Unknown sheet preset: {preset}")
