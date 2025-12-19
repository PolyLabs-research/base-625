# Technical Overview

This repository contains two main streams of work:
- **Base-625 glyph toolkit:** a 25×25 Boolean grid numeral/symbol system (0–625 states) with renderers, GUIs, dataset builders, and validators.
- **Spatial Brain playground:** exploratory neural-field simulations (1D/2D cortical sheets with beta/gamma control fields, WM tasks, and a small navigation RL toy) with an eye toward hosting the base-625 glyphs.

Use this document as a handoff guide for agents picking up the project.

## Environment & Dependencies
- Python 3.12+ recommended.
- Core packages: `pillow`, `numpy`. GUI scripts rely on Tkinter (bundled with most Python builds).
- Spatial Brain extras (`spatial_brain`): `torch`, `matplotlib`, `streamlit` (see `requirements_spatial_brain.txt`).
- Hardware: CUDA optional; training scripts auto-detect GPU via `torch.cuda.is_available()`.
- Snapshot the machine before starting a session: `python3 collect_system_info.py` writes `system_summary_<timestamp>.txt` in repo root.

## Repository Layout (key paths)
- `base625_grid.py` – grid logic, symbol glyphs, expression encoder CLI.
- `symbol_painter.py` – Tkinter glyph editor; outputs JSON/CSV decode tables + PNGs under `out/symbols/`.
- `gen_bit_dataset.py` – builds NumPy/CSV/JSONL datasets of digits (0–625) and custom symbols.
- `export_0_625.py` – exports images + decode tables for every value.
- `validate_bits_dataset.py` – consistency checks for generated datasets (shape, label coverage, collisions, montages, optional quick MLP).
- `base626_calculator.py` – GUI calculator rendering expressions as base-626 panels using saved glyphs.
- `number_viewer.py` – quick GUI to render a single number’s digit panels.
- `collect_system_info.py` – environment snapshot utility.
- `spatial_brain/` – spatial computing playground (WM tasks, glyph-mode experiments, 2D navigation RL, web viz).
- Other exploratory physics/neuro scripts: `orch_or_sim.py`, `2D Orch-OR Simulation.py`, `Quantum Maze Solver.py`, `Quantum Wedge Sensor.py`, notebooks/plots in root.
- Generated artifacts live in `out/` and `data/` (not versioned by default).

## Base-625 Toolkit
- **Render digits/symbols:** `python3 base625_grid.py encode-digit 42 -o out/d42.png --order row` or `encode-number` for count-mode (0..625). `encode-expr` stitches a whole expression strip; `symbol-sheet` dumps all operator glyphs.
- **Glyph authoring:** `python3 symbol_painter.py`. Draw, optionally auto-center, and Save. Outputs:
  - PNG per glyph under `out/symbols/`
  - `out/symbols/decode_table_symbols.json` / `.csv` (pretty symbol, label, paths, hashes)
- **Dataset generation:** `python3 gen_bit_dataset.py --out data/bits --include digits symbols --order row --copies 3 --noise-flip 0.01 --csv`. Outputs `X.npy`, `y.npy`, `dataset.npz`, `class_map.json` (+ CSV/JSONL if requested). Digits 0–624 use digit encoding; 625 uses count-mode full grid.
- **Dataset validation:** `python3 validate_bits_dataset.py --data data/bits --symbols out/symbols/decode_table_symbols.json --montages --quick-train`. Checks binary shapes, class coverage, digit bit-sums, symbol parity with PNGs, collisions, and optional small MLP accuracy; writes reports to `data/bits/validation_output/`.
- **Bulk export:** `python3 export_0_625.py --out out/digits_0_625_row --order row --cell-px 20 --mode count` (or `--mode digit`). Writes decode tables in JSON/CSV plus per-value PNGs.
- **Interactive viewers:**
  - `python3 base626_calculator.py` renders expressions using saved glyphs (expects `out/symbols/decode_table_symbols.json`). Supports row/snake/random42 fill orders and resizable cell/gap.
  - `python3 number_viewer.py` shows a single number as base-626 digit panels.
- **Conventions & gotchas:** 25×25 grid → 625 cells. In digit mode valid digits are 0–624; value 625 is represented via count-mode. Operator decoding in `base625_grid.py` is exact-boolean-match only. Regenerate decode table after glyph edits so downstream tools stay in sync.

## Spatial Brain Playground
- **Concept:** cortical content vs control separation. Content vectors live on spatial substrates; beta fields gate updates, gamma modulates expression. WM tasks carve spatial “slots”.
- **Core modules:** `core/` has 1D/2D control networks (`control_1d.py`, `control_2d.py`), cortical substrates (`cortical_line.py`, `cortical_sheet_2d.py`), WM task generator (`tasks.py`), training loops (`training_loop.py`), and utilities (`utils.py` for seeding/logging/device selection).
- **1D WM experiment:** `python -m spatial_brain.experiments.wm_3slot_1d --num-epochs 5 --batch-size 16 ...` trains beta/gamma stencils over a line. Optional checkpoint via `--save-path`.
- **2D WM experiment:** `python -m spatial_brain.experiments.wm_3slot_2d --height 16 --width 16 --beta-smooth-lambda 0.01 --num-epochs 10`. Supports sheet presets (see README_spatial_brain.md) and beta smoothness regularization.
- **Glyph-mode WM:** `python -m spatial_brain.experiments.glyph_slots_25x25 --num-epochs 200 --beta-smooth-lambda 0.02` uses a 25×25 sheet preset and synthetic/base-625 glyph banks from `integration/base625_integration.py`.
- **Debugging/viz:** `experiments/debug_viz_2d.py`, `slot_maps_2d.py`, and `viz/plot_fields.py` provide plots of beta/gamma/content trajectories; saved figures are not versioned.
- **Nav2D RL toy:** `python -m spatial_brain.experiments.nav2d_train` wraps `nav2d/train_rl.py` (policy/value losses over grid worlds in `nav2d/env.py`, agent in `nav2d/brain_agent.py`). Checkpoints under `spatial_brain/checkpoints` or `spatial_brain/episodes/nav2d/checkpoints`. Web/Streamlit visualizer at `spatial_brain/web/nav2d_app.py` (requires `streamlit run spatial_brain/web/nav2d_app.py`).
- **Integration hooks:** `integration/base625_integration.py` contains helpers for locating base-625 assets and constructing glyph banks; `README_spatial_brain.md` outlines vision to embed real 25×25 glyphs into spatial tasks.

## Outputs & Conventions
- Generated assets: `out/` (digits strips, operator glyphs, PNG strips, meta JSON) and `data/` (datasets). These are safe to regenerate; only the latest nav2d checkpoint is typically kept.
- Plots such as `frohlich_condensation_waterfall.png`, raster plots, and heatmaps are outputs from experiments and can be regenerated by rerunning the corresponding scripts/notebooks.
- System snapshots: rerun `collect_system_info.py` whenever hardware/software changes; keep the newest summary for debugging context.

## Quick Start (handoff checklist)
- Ensure Python env with `pip install pillow numpy` (+ `torch matplotlib streamlit` for spatial brain).
- Run `python3 collect_system_info.py`.
- If working on glyphs/datasets: refresh glyph decode table via `symbol_painter.py`, regenerate datasets with `gen_bit_dataset.py`, validate with `validate_bits_dataset.py`.
- If working on spatial experiments: pick the appropriate runner under `spatial_brain/experiments/`, adjust CLI flags, and stash outputs under `spatial_brain/docs` or `out/` as needed.
- Avoid altering or deleting existing untracked artifacts unless explicitly agreed; repo may be in a dirty state.

## Open Threads / Extension Ideas
- Complete the base-625 ↔ spatial brain integration (pipe real glyph bitmaps into `input_encoder` for 2D WM or nav2d tasks).
- Expand symbol set and ensure decode table + dataset mappings stay consistent when new glyphs are added.
- Improve operator decoding beyond exact boolean match (tolerance for noise) if required by downstream ML tasks.
- Harden nav2d Streamlit app and provide saved demo checkpoints for quick inspection.
