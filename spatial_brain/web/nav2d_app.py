"""Streamlit app for 2D navigation with the spatial brain."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import streamlit as st
import torch

# Ensure project root is importable when launched via `streamlit run`.
ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_brain.core.control_2d import ControlNetwork2D
from spatial_brain.core.cortical_sheet_2d import CorticalSheet2D
from spatial_brain.nav2d.brain_agent import BrainAgent2D
from spatial_brain.nav2d.env import GridWorldEnv
from spatial_brain.nav2d.rollout import run_episode, save_episode_log
from spatial_brain.viz.nav2d_viz import episode_summary, plot_brain_fields, plot_grid_state


class RandomPolicy:
    """Uniform random policy stub."""

    def __init__(self, action_dim: int = 4) -> None:
        self.action_dim = action_dim

    def reset_state(self, batch_size: int = 1) -> None:
        return None

    def select_action(self, obs: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
        action = torch.randint(low=0, high=self.action_dim, size=(1,))
        return {"action": action, "beta": None, "gamma": None, "content_norm": None}


def build_brain_agent(height: int, width: int, max_steps: int, device: torch.device) -> BrainAgent2D:
    content_dim = 32
    sheet = CorticalSheet2D(height=height, width=width, content_dim=content_dim, device=device)
    control = ControlNetwork2D(
        height=height,
        width=width,
        context_dim=4,
        max_steps=max_steps + 1,
        device=device,
    )
    agent = BrainAgent2D(sheet=sheet, control=control, action_dim=4, device=device)
    return agent


def maybe_load_checkpoint(agent: BrainAgent2D, path_str: str) -> bool:
    """Try to load a checkpoint into the agent."""
    if not path_str:
        return False
    path = Path(path_str).expanduser()
    if not path.exists():
        st.error(f"Checkpoint not found: {path}")
        return False
    try:
        ckpt = torch.load(path, map_location=agent.device)
        state_dict = None
        if isinstance(ckpt, dict):
            if "agent_state_dict" in ckpt:
                state_dict = ckpt["agent_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        if state_dict is not None:
            agent.load_state_dict(state_dict, strict=False)
            return True
    except Exception as exc:  # pragma: no cover - best effort load
        st.error(f"Failed to load checkpoint: {exc}")
    return False


def render_episode(log: Dict[str, Any]) -> None:
    """Display episode playback controls and plots."""
    steps = log.get("log", [])
    if not steps:
        st.info("No steps recorded.")
        return
    step_idx = st.slider("Step", min_value=0, max_value=len(steps) - 1, value=len(steps) - 1)
    step = steps[step_idx]

    col_env, col_brain = st.columns(2)
    with col_env:
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_grid_state(step["grid"], ax=ax)
        st.pyplot(fig)
    with col_brain:
        beta = step.get("beta")
        gamma = step.get("gamma")
        content_norm = step.get("content_norm")
        if beta is not None and gamma is not None and content_norm is not None:
            beta_2d = beta[0] if beta.ndim == 3 else beta
            gamma_2d = gamma[0] if gamma.ndim == 3 else gamma
            norm_2d = content_norm[0] if content_norm.ndim == 3 else content_norm
            fig_fields, axes = plt.subplots(1, 3, figsize=(12, 4))
            plot_brain_fields(beta_2d, gamma_2d, norm_2d, *axes)
            st.pyplot(fig_fields)
        else:
            st.info("No brain fields recorded for this run.")

    stats = episode_summary(log)
    st.subheader("Episode stats")
    st.json(stats)


def main() -> None:
    st.set_page_config(page_title="Spatial Brain 2D Navigation Sandbox", layout="wide")
    st.title("Spatial Brain 2D Navigation Sandbox")

    # Sidebar controls
    st.sidebar.header("Environment")
    width = st.sidebar.number_input("Width", min_value=5, max_value=25, value=9, step=1)
    height = st.sidebar.number_input("Height", min_value=5, max_value=25, value=9, step=1)
    obstacle_density = st.sidebar.slider("Obstacle density", 0.0, 0.4, value=0.1, step=0.05)
    max_steps = st.sidebar.number_input("Max steps", min_value=10, max_value=200, value=50, step=5)

    st.sidebar.header("Policy")
    model_choice = st.sidebar.radio(
        "Agent",
        options=["Random policy (no brain)", "New untrained brain", "Load brain checkpoint"],
    )
    checkpoint_path = ""
    if model_choice == "Load brain checkpoint":
        checkpoint_path = st.sidebar.text_input("Checkpoint path (.pt)")

    save_logs = st.sidebar.checkbox("Save episode log", value=False)
    run_button = st.sidebar.button("Run single episode")

    if "last_episode_log" not in st.session_state:
        st.session_state["last_episode_log"] = None

    if run_button:
        env = GridWorldEnv(
            width=int(width),
            height=int(height),
            obstacle_density=float(obstacle_density),
            max_steps=int(max_steps),
            device="cpu",
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_choice == "Random policy (no brain)":
            agent: Any = RandomPolicy()
            record_brain = False
        else:
            agent = build_brain_agent(height=int(height), width=int(width), max_steps=int(max_steps), device=device)
            record_brain = True
            if model_choice == "Load brain checkpoint" and checkpoint_path:
                maybe_load_checkpoint(agent, checkpoint_path)

        episode_log = run_episode(env, agent, record_brain=record_brain)
        st.session_state["last_episode_log"] = episode_log

        if save_logs:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path("spatial_brain/episodes/nav2d")
            out_path = out_dir / f"episode_{ts}.pt"
            save_episode_log(episode_log, out_path)
            st.sidebar.success(f"Saved log to {out_path}")

    last_log = st.session_state.get("last_episode_log")
    if last_log:
        render_episode(last_log)
    else:
        st.info("Run an episode to see the visualization.")


if __name__ == "__main__":
    main()
