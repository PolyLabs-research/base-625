# Base 625 Toolkit

Toolkit for experimenting with a 25×25 grid numeral system that encodes values from 0–625, renders custom operator glyphs, and produces datasets for downstream modeling. The project includes rendering utilities, interactive editors, dataset generators, and a repeatable system summary utility for tracking the development environment.

## Highlights

- **Grid rendering (`base625_grid.py`)** – Encode numbers as cell-count or base-625 digits, generate symbol glyphs, and render crystal-clear PNG panels.
- **Interactive tooling** – Paint new operator glyphs (`symbol_painter.py`), inspect digits, and run a base-626 expression calculator.
- **Dataset builders** – Export 625-bit vectors and labeled metadata for digits and symbols.
- **Environment snapshots** – Capture hardware/OS/toolchain details with `collect_system_info.py` before each work session.

## Requirements

- Python 3.12+
- `pip install pillow numpy`
- Optional GPU tooling: `nvidia-smi`, `lspci`
- GUI scripts require a windowing environment (Tkinter is bundled with the standard Python build on most platforms).

## Quick Start

```bash
cd "base 625"
python3 collect_system_info.py            # snapshot current system
python3 base625_grid.py -h                # discover rendering CLI options
python3 symbol_painter.py                 # paint or edit operator glyphs
python3 gen_bit_dataset.py --help         # build digit/symbol datasets
python3 base626_calculator.py             # evaluate expressions visually
```

Output artifacts (PNG strips, JSON tables, datasets) land under `out/` or a user-specified directory.

## Regenerating outputs (kept out of version control)

- **Digit strips (0–625):** `python3 export_0_625.py --out out/digits_0_625_row --order row --cell-px 20`
- **Digit/symbol datasets:** `python3 gen_bit_dataset.py --out data/bits --include digits symbols --order row --csv`  
  Verify with `python3 validate_bits_dataset.py --data data/bits --symbols out/symbols/decode_table_symbols.json`.
- **System snapshot:** `python3 collect_system_info.py` writes a fresh `system_summary_*.txt`.
- **Spatial brain plots/checkpoints:** rerun the relevant experiment/train scripts under `spatial_brain/experiments` to regenerate figures and new checkpoints (only the most recent nav2d checkpoint is retained by default).

## Core Modules

- `base625_grid.py` – Foundation for generating 25×25 boolean grids, digit encoders, symbol glyph definitions, dataset helpers, and image rendering utilities.
- `symbol_painter.py` – Tkinter GUI to design glyphs. Saves PNGs plus `decode_table_symbols.json/CSV` in `out/symbols/`.
- `base626_calculator.py` – GUI calculator that parses arithmetic expressions, converts numbers to base-626 panels, and stitches operator glyphs.
- `number_viewer.py` – Lightweight panel browser for exploring number encodings (count and digit modes).
- `gen_bit_dataset.py` – Emits NumPy arrays (`X.npy`, `y.npy`, `dataset.npz`) and optional CSV/JSONL records of digits and symbols.
- `export_0_625.py` – Convenience exporter for digit image strips and metadata.
- `validate_bits_dataset.py` – Sanity-check generated datasets (shape, class mapping, basic statistics).
- `collect_system_info.py` – Writes a timestamped `system_summary_*.txt` capturing OS, hardware, toolchain versions, Python environment, networking, and running services.

## Typical Workflows

### 1. Capture the Environment

Run before changes to produce a fresh system snapshot:

```bash
python3 collect_system_info.py
```

Each execution generates a new `system_summary_YYYYMMDD_HHMMSS.txt` in the project root for later reference.

### 2. Generate Operator Glyphs

```bash
python3 symbol_painter.py
```

- Left-click/drag to draw, right-click to erase.
- Use **Save** to write glyph PNGs and update the decode table in `out/symbols/`.
- Centering and gridline options help maintain a clean template.

### 3. Produce Digit or Symbol Datasets

```bash
python3 gen_bit_dataset.py --out data/bits --include digits symbols --order row --copies 3 --noise-flip 0.01 --csv
```

- Digits 0–625 become 625-bit vectors (row-major).
- Symbols are reconstructed from the PNG decode table.
- Increase `--copies` and `--noise-flip` to augment the dataset.

### 4. Render Expressions

```bash
python3 base626_calculator.py
```

Enter expressions like `12 + 34 =` to see each operand/operator rendered as panels. Switch fill order, adjust panel spacing, or export stitched results.

## Directory Layout (key paths)

- `base625_grid.py` – Grid/digit core.
- `symbol_painter.py` – Glyph painter GUI.
- `base626_calculator.py` – Expression viewer.
- `out/digits_*/` – Generated digit strips and metadata.
- `out/symbols/` – Saved glyph PNGs and decode tables.
- `data/` – Suggested home for dataset outputs (`gen_bit_dataset.py`).
- `system_summary_*.txt` – Timestamped environment snapshots (created on demand).

## Contributing Tips

- Regenerate the decode table after adding or editing glyphs so downstream tools see the new assets.
- Keep the newest `system_summary_*.txt` handy when diagnosing environment-specific issues.
- Verify dataset outputs with `validate_bits_dataset.py` before training or sharing results.
- The repo may contain existing changes; avoid cleaning or resetting without explicit confirmation.

Enjoy experimenting with base-625 panels!
