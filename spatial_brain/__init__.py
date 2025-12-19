"""Spatial Computation playground package."""

from spatial_brain.core.control_1d import ControlNetwork1D
from spatial_brain.core.control_2d import ControlNetwork2D
from spatial_brain.core.cortical_line import CorticalLine
from spatial_brain.core.cortical_sheet_2d import CorticalSheet2D
from spatial_brain.core.tasks import ThreeSlotWMTask
from spatial_brain.core.training_loop import (
    TrainingConfig,
    TrainingConfig2D,
    train_three_slot_wm as train_wm_3slot_1d,
    train_three_slot_wm_2d as train_wm_3slot_2d,
    run_single_trial_2d,
)

__all__ = [
    "CorticalLine",
    "CorticalSheet2D",
    "ControlNetwork1D",
    "ControlNetwork2D",
    "ThreeSlotWMTask",
    "TrainingConfig",
    "TrainingConfig2D",
    "train_wm_3slot_1d",
    "train_wm_3slot_2d",
    "run_single_trial_2d",
]
