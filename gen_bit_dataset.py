# gen_bit_dataset.py
#
# Generate a 0/1 "bit" dataset for Base-626 panels + custom symbols.
# - Digits: 0..625 as single-panel "digits" (0..624 via encode_digit; 625 via count-mode 625)
# - Symbols: loaded from out/symbols/decode_table_symbols.json; PNG -> 25x25 bits by sampling cell centers
#
# Outputs in --out folder:
#   X.npy, y.npy, dataset.npz, class_map.json
#   (optional) dataset.csv, dataset.jsonl
#
# Examples:
#   py -3.12 gen_bit_dataset.py --out data\\bits --include digits symbols --order row --copies 1 --noise-flip 0.0 --csv
#   py -3.12 gen_bit_dataset.py --out data\\bits_aug --include digits --order random42 --copies 10 --noise-flip 0.01
#
# Requirements: numpy, pillow
# Uses your base625_grid.py and symbols JSON from symbol_painter.py

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

# from your project
from base625_grid import (
    GRID_N,
    GRID_CELLS,
    encode_digit,
    encode_number_as_count,
)

DEFAULT_SYMBOLS_JSON = Path("out") / "symbols" / "decode_table_symbols.json"


def grid_bool_to_row(bits25x25: np.ndarray) -> np.ndarray:
    """(25,25) bool -> (625,) uint8 in row-major order."""
    return bits25x25.astype(np.uint8).reshape(-1)


def make_digit_grid(d: int, order: str) -> np.ndarray:
    """Return 25x25 bool grid for base-626 digit d."""
    if 0 <= d <= 624:
        g = encode_digit(d, order=order)
    elif d == 625:
        g = encode_number_as_count(625, order=order)
    else:
        raise ValueError("digit out of range [0..625]")
    return g  # bool np.ndarray (25,25)


def sample_png_to_grid25(png_path: Path) -> np.ndarray:
    """
    Reconstruct a 25x25 boolean grid from a PNG created by grid_to_image().
    We infer cell_px from width and sample cell centers (robust to gridlines).
    """
    im = Image.open(png_path).convert("RGB")
    w, h = im.width, im.height
    # width = GRID_N * cell_px + 1 (same for height); infer cell_px
    cell_px = round((w - 1) / GRID_N)
    cell_px = max(1, cell_px)
    grid = np.zeros((GRID_N, GRID_N), dtype=bool)
    for r in range(GRID_N):
        for c in range(GRID_N):
            x = min(w - 1, c * cell_px + cell_px // 2)
            y = min(h - 1, r * cell_px + cell_px // 2)
            R, G, B = im.getpixel((x, y))
            # black cell if dark (threshold halfway)
            grid[r, c] = (R + G + B) < (255 * 3 / 2)
    return grid


def apply_bit_noise(vec625: np.ndarray, flip_prob: float, rng: random.Random) -> np.ndarray:
    """Flip each bit with probability p (independent Bernoulli). vec625 is uint8 {0,1}."""
    if flip_prob <= 0:
        return vec625
    mask = (np.array([rng.random() < flip_prob for _ in range(vec625.size)], dtype=bool))
    out = vec625.copy()
    out[mask] ^= 1  # flip bits
    return out


def build_class_map(symbols_items: List[Dict]) -> Tuple[Dict[int, Dict], Dict[str, int]]:
    """
    Assign class IDs:
      0..625 -> digits
      626..  -> symbols in a stable order (sort by 'symbol' then 'label' as fallback)
    Returns (id_to_meta, symbol_to_id_for_ops)
    """
    id_to_meta: Dict[int, Dict] = {}
    # digits
    for d in range(0, GRID_CELLS + 1):
        id_to_meta[d] = {"type": "DIGIT", "digit": d}

    # symbols
    start = GRID_CELLS + 1  # 626
    # sort for stability
    norm = lambda s: (s or "").strip()
    sym_sorted = sorted(symbols_items, key=lambda it: (norm(it.get("symbol")), norm(it.get("label"))))

    sym_to_id: Dict[str, int] = {}
    cid = start
    for it in sym_sorted:
        sym = norm(it.get("symbol"))
        lab = norm(it.get("label"))
        if not sym:
            continue
        if sym in sym_to_id:
            # skip duplicates of same printed symbol
            continue
        id_to_meta[cid] = {"type": "SYMBOL", "label": lab or sym, "symbol": sym}
        sym_to_id[sym] = cid
        cid += 1

    return id_to_meta, sym_to_id


def main():
    ap = argparse.ArgumentParser(description="Generate 625-bit dataset for Base-626 digits and custom symbols.")
    ap.add_argument("--out", required=True, help="Output folder (created if missing).")
    ap.add_argument("--include", nargs="+", choices=["digits", "symbols"], default=["digits", "symbols"],
                    help="What to include in the dataset.")
    ap.add_argument("--order", choices=["row", "snake", "random42"], default="row",
                    help="Digit fill order (only affects digits 0..624; 625 is full grid).")
    ap.add_argument("--copies", type=int, default=1,
                    help="How many copies per class (digits and symbols). Use >1 with --noise-flip to augment.")
    ap.add_argument("--noise-flip", type=float, default=0.0,
                    help="Bit flip probability per cell (e.g., 0.01 for 1%%). Applied independently to each copy.")
    ap.add_argument("--symbols-json", type=str, default=str(DEFAULT_SYMBOLS_JSON),
                    help="Path to decode_table_symbols.json generated by symbol_painter.py.")
    ap.add_argument("--csv", action="store_true", help="Also export dataset.csv (large).")
    ap.add_argument("--jsonl", action="store_true", help="Also export dataset.jsonl (large).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for noise.")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Load symbols JSON (may be empty if not including symbols)
    symbols_items: List[Dict] = []
    sym_json_path = Path(args.symbols_json)
    if "symbols" in args.include:
        if not sym_json_path.exists():
            raise FileNotFoundError(f"Symbols JSON not found: {sym_json_path}")
        with open(sym_json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            symbols_items = list(obj.get("items", []))

    # Build class map (digits always in the map; symbols added if present)
    id_to_meta, sym_to_id = build_class_map(symbols_items)

    # Prepare lists
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []

    # --- Digits 0..625 ---
    if "digits" in args.include:
        for d in range(0, GRID_CELLS + 1):
            grid = make_digit_grid(d, order=args.order)              # bool (25,25)
            base_vec = grid_bool_to_row(grid)                        # uint8 (625,)
            for _ in range(args.copies):
                vec = apply_bit_noise(base_vec, args.noise_flip, rng)
                X_rows.append(vec)
                y_rows.append(d)

    # --- Symbols from JSON ---
    if "symbols" in args.include:
        # build list of unique symbols with an existing PNG path
        seen_syms = set()
        for item in symbols_items:
            sym = (item.get("symbol") or "").strip()
            p = Path(item.get("path", ""))
            if not sym or not p.exists():
                continue
            if sym in seen_syms:
                continue
            seen_syms.add(sym)
            class_id = sym_to_id.get(sym)
            if class_id is None:
                # shouldn't happen; skip if not in map
                continue
            # reconstruct 25x25 bool from the PNG
            grid = sample_png_to_grid25(p)
            base_vec = grid_bool_to_row(grid)
            for _ in range(args.copies):
                vec = apply_bit_noise(base_vec, args.noise_flip, rng)
                X_rows.append(vec)
                y_rows.append(class_id)

    if not X_rows:
        print("No samples generated (check --include and symbols JSON).")
        return

    X = np.stack(X_rows, axis=0).astype(np.uint8)  # (N,625)
    y = np.array(y_rows, dtype=np.int64)           # (N,)

    # Save npy/npz
    np.save(outdir / "X.npy", X)
    np.save(outdir / "y.npy", y)
    np.savez_compressed(outdir / "dataset.npz", X=X, y=y)

    # Save class map
    with open(outdir / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(id_to_meta, f, indent=2, ensure_ascii=False)

    # Optional CSV
    if args.csv:
        csv_path = outdir / "dataset.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            # header
            cols = ["y"] + [f"x{i}" for i in range(X.shape[1])]
            f.write(",".join(cols) + "\n")
            for i in range(X.shape[0]):
                row = [str(int(y[i]))] + [str(int(b)) for b in X[i]]
                f.write(",".join(row) + "\n")

    # Optional JSONL
    if args.jsonl:
        jl_path = outdir / "dataset.jsonl"
        with open(jl_path, "w", encoding="utf-8") as f:
            for i in range(X.shape[0]):
                rec = {"y": int(y[i]), "x": X[i].tolist()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {X.shape[0]} samples to {outdir}")
    print(f"Shapes: X={X.shape}, y={y.shape}")
    print(f"Classes in class_map.json: {len(id_to_meta)} (digits 0..625 + symbols present)")

if __name__ == "__main__":
    main()
