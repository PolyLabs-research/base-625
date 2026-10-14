# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
base625_grid.py — Framework for a 25×25 grid numeral/symbol system (0..625 states)

Quick overview
--------------
• Grid is 25×25 Boolean.
• Numbers-as-count ("count mode"): k black cells ⇒ value k, for k in [0, 625].
• Base-625 "digit mode": a digit is an integer d in [0, 624] drawn via a canonical fill order.
• Symbols: pre-baked 25×25 glyphs (+, -, ×, ÷, =, (, ), ^, ·). Exact-match decoding provided.
• Rendering: crisp PNGs with optional gridlines.
• Datasets: generate token sequences (numbers converted to base-625 digits + operators), saved as an image strip and JSON.
• CLI included. Run `python base625_grid.py -h` for commands.

Design notes
------------
• Canonical fill order defaults to row-major (top-left → right, then down).
  You can also choose a fixed-seed pseudorandom order for better spatial variety.
• Decoding a *number panel* is simply `sum(grid)` (i.e., count of black cells).
• Operator glyphs are designed so they do NOT look like monotonic row-major fills.
• For base-625 digits, the value 625 is not a valid digit (max 624). In count mode,
  625 (full grid) is allowed and may be useful as a reserved token if you like.

Author: ChatGPT (GPT-5 Thinking)
License: MIT
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw

GRID_N = 25  # 25x25
GRID_CELLS = GRID_N * GRID_N  # 625


# -----------------------------
# Utility: Cell orders
# -----------------------------

def row_major_order(n: int = GRID_N) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(n) for c in range(n)]


def snake_order(n: int = GRID_N) -> List[Tuple[int, int]]:
    coords = []
    for r in range(n):
        cols = range(n) if r % 2 == 0 else range(n - 1, -1, -1)
        for c in cols:
            coords.append((r, c))
    return coords


def seeded_random_order(seed: int = 42, n: int = GRID_N) -> List[Tuple[int, int]]:
    coords = row_major_order(n)
    rng = random.Random(seed)
    rng.shuffle(coords)
    return coords


ORDER_FACTORIES = {
    "row": row_major_order,
    "snake": snake_order,
    # For reproducibility; change seed if you need a different static order
    "random42": lambda n=GRID_N: seeded_random_order(42, n),
}


# -----------------------------
# Grid helpers
# -----------------------------

def empty_grid(n: int = GRID_N) -> np.ndarray:
    return np.zeros((n, n), dtype=bool)


def fill_k_cells(k: int, order: str = "row", n: int = GRID_N) -> np.ndarray:
    """
    Returns a 25x25 boolean grid with exactly k cells filled (True),
    according to the chosen cell order.
    """
    if not (0 <= k <= n * n):
        raise ValueError(f"k must be in [0, {n*n}], got {k}")
    coords = ORDER_FACTORIES[order](n)
    grid = empty_grid(n)
    for (r, c) in coords[:k]:
        grid[r, c] = True
    return grid


def count_filled(grid: np.ndarray) -> int:
    return int(grid.sum())


# -----------------------------
# Symbol glyphs (25x25 boolean)
# -----------------------------

def _draw_rect(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    grid[r0:r1, c0:c1] = True


def _draw_h_line(grid: np.ndarray, r: int, c0: int, c1: int, thickness: int = 2) -> None:
    _draw_rect(grid, max(0, r - thickness // 2), min(GRID_N, r + (thickness + 1) // 2), c0, c1)


def _draw_v_line(grid: np.ndarray, c: int, r0: int, r1: int, thickness: int = 2) -> None:
    _draw_rect(grid, r0, r1, max(0, c - thickness // 2), min(GRID_N, c + (thickness + 1) // 2))


def glyph_plus(thickness: int = 3) -> np.ndarray:
    g = empty_grid()
    mid = GRID_N // 2
    pad = GRID_N // 6
    _draw_h_line(g, mid, pad, GRID_N - pad, thickness)
    _draw_v_line(g, mid, pad, GRID_N - pad, thickness)
    return g


def glyph_minus(thickness: int = 3) -> np.ndarray:
    g = empty_grid()
    mid = GRID_N // 2
    pad = GRID_N // 6
    _draw_h_line(g, mid, pad, GRID_N - pad, thickness)
    return g


def glyph_equals(thickness: int = 3, gap: int = 3) -> np.ndarray:
    g = empty_grid()
    mid = GRID_N // 2
    pad = GRID_N // 6
    _draw_h_line(g, mid - gap, pad, GRID_N - pad, thickness)
    _draw_h_line(g, mid + gap, pad, GRID_N - pad, thickness)
    return g


def glyph_times(thickness: int = 2) -> np.ndarray:
    g = empty_grid()
    # Diagonals for an 'x'
    for i in range(GRID_N):
        _draw_v_line(g, i, max(0, i - thickness), min(GRID_N, i + thickness), 1)
        j = GRID_N - 1 - i
        _draw_v_line(g, i, max(0, j - thickness), min(GRID_N, j + thickness), 1)
    return g


def glyph_divide(thickness: int = 2) -> np.ndarray:
    g = empty_grid()
    mid = GRID_N // 2
    pad = GRID_N // 6
    _draw_h_line(g, mid, pad, GRID_N - pad, thickness)
    # Dots
    dot_r = GRID_N // 6
    _draw_rect(g, pad - dot_r // 2, pad + dot_r // 2, mid - dot_r // 2, mid + dot_r // 2)
    _draw_rect(g, GRID_N - pad - dot_r // 2, GRID_N - pad + dot_r // 2, mid - dot_r // 2, mid + dot_r // 2)
    return g


def glyph_dot(size: int = 3) -> np.ndarray:
    g = empty_grid()
    mid = GRID_N // 2
    _draw_rect(g, mid - size, mid + size + 1, mid - size, mid + size + 1)
    return g


def glyph_pow(thickness: int = 2) -> np.ndarray:
    # Caret "^" like shape
    g = empty_grid()
    # Draw a shallow '^' by two lines
    for i in range(GRID_N // 2):
        _draw_rect(g, GRID_N // 3 - i, GRID_N // 3 - i + thickness, GRID_N // 4 + i, GRID_N // 4 + i + thickness)
        _draw_rect(g, GRID_N // 3 - i, GRID_N // 3 - i + thickness, GRID_N - GRID_N // 4 - i - thickness, GRID_N - GRID_N // 4 - i)
    return g


def glyph_lparen(thickness: int = 2) -> np.ndarray:
    g = empty_grid()
    # Draw a "(" using a vertical stroke with hooks
    pad = GRID_N // 6
    _draw_v_line(g, pad, pad, GRID_N - pad, thickness)
    # Hooks top/bottom
    _draw_h_line(g, pad, pad, pad + GRID_N // 6, thickness)
    _draw_h_line(g, GRID_N - pad, pad, pad + GRID_N // 6, thickness)
    return g


def glyph_rparen(thickness: int = 2) -> np.ndarray:
    g = empty_grid()
    pad = GRID_N - GRID_N // 6
    _draw_v_line(g, pad, GRID_N // 6, GRID_N - GRID_N // 6, thickness)
    _draw_h_line(g, GRID_N // 6, GRID_N - GRID_N // 6 - GRID_N // 6, pad, thickness)
    _draw_h_line(g, GRID_N - GRID_N // 6, GRID_N - GRID_N // 6 - GRID_N // 6, pad, thickness)
    return g


SYMBOLS: Dict[str, np.ndarray] = {
    "PLUS": glyph_plus(),
    "MINUS": glyph_minus(),
    "TIMES": glyph_times(),
    "DIV": glyph_divide(),
    "EQUALS": glyph_equals(),
    "DOT": glyph_dot(),
    "POW": glyph_pow(),
    "LPAREN": glyph_lparen(),
    "RPAREN": glyph_rparen(),
}


def resolve_asset_path(item: Dict, table_path: Optional[Path] = None, max_parents: int = 3) -> Optional[Path]:
    """Locate the PNG referenced by a decode-table item.

    Decode tables written on Windows carry backslash paths relative to the repo
    root, which do not resolve on Linux/macOS. Candidates are tried in order:
    the item's ``filename`` next to the table file, the normalised ``path``
    relative to the current directory, then relative to the table's parents.
    Returns None when nothing exists.
    """
    candidates: List[Path] = []
    filename = (item.get("filename") or "").strip()
    raw_path = (item.get("path") or "").replace("\\", "/").strip()
    if table_path is not None and filename:
        candidates.append(Path(table_path).parent / filename)
    if raw_path:
        candidates.append(Path(raw_path))
        if table_path is not None:
            for parent in list(Path(table_path).resolve().parents)[:max_parents]:
                candidates.append(parent / raw_path)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def decode_symbol(grid: np.ndarray) -> Optional[str]:
    """Exact boolean match decoder. Returns symbol name or None."""
    for name, g in SYMBOLS.items():
        if grid.shape == g.shape and np.array_equal(grid, g):
            return name
    return None


# -----------------------------
# Base conversion helpers
# -----------------------------

def int_to_base625(n: int) -> List[int]:
    """Convert non-negative integer to base-625 digits (MSD → LSD). 0 => [0]."""
    if n < 0:
        raise ValueError("Only non-negative integers supported")
    if n == 0:
        return [0]
    digits = []
    base = 625
    while n > 0:
        n, r = divmod(n, base)
        digits.append(r)
    return list(reversed(digits))


def base625_to_int(digits: List[int]) -> int:
    base = 625
    val = 0
    for d in digits:
        if not (0 <= d <= 624):
            raise ValueError("Base-625 digits must be in [0, 624]")
        val = val * base + d
    return val


# -----------------------------
# Rendering
# -----------------------------

def grid_to_image(grid: np.ndarray, cell_px: int = 20, draw_grid: bool = True) -> Image.Image:
    n = grid.shape[0]
    W = n * cell_px
    H = n * cell_px
    img = Image.new("RGB", (W + 1, H + 1), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Paint cells
    for r in range(n):
        for c in range(n):
            x0 = c * cell_px
            y0 = r * cell_px
            x1 = x0 + cell_px
            y1 = y0 + cell_px
            fill = (0, 0, 0) if grid[r, c] else (255, 255, 255)
            draw.rectangle([x0, y0, x1, y1], fill=fill)

    # Gridlines
    if draw_grid:
        for c in range(n + 1):
            x = c * cell_px
            draw.line([(x, 0), (x, H)], fill=(0, 0, 0), width=1)
        for r in range(n + 1):
            y = r * cell_px
            draw.line([(0, y), (W, y)], fill=(0, 0, 0), width=1)

    return img


def strip_to_image(panels: List[np.ndarray], gap_px: int = 8, cell_px: int = 20, draw_grid: bool = True) -> Image.Image:
    """Concatenate panels horizontally with a small gap."""
    if not panels:
        raise ValueError("No panels")
    rendered = [grid_to_image(p, cell_px=cell_px, draw_grid=draw_grid) for p in panels]
    W = sum(im.width for im in rendered) + gap_px * (len(rendered) - 1)
    H = max(im.height for im in rendered)
    out = Image.new("RGB", (W, H), (255, 255, 255))
    x = 0
    for i, im in enumerate(rendered):
        out.paste(im, (x, 0))
        x += im.width + gap_px
    return out


# -----------------------------
# Token types
# -----------------------------

@dataclass(frozen=True)
class Token:
    kind: str  # "DIGIT", "NUMBER", or a symbol name like "PLUS"
    value: Optional[int] = None  # for digits or number-in-count-mode


def encode_number_as_count(k: int, order: str = "row") -> np.ndarray:
    """0..625 inclusive -> grid."""
    if not (0 <= k <= GRID_CELLS):
        raise ValueError(f"Number must be in [0, {GRID_CELLS}]")
    return fill_k_cells(k, order=order, n=GRID_N)


def encode_digit(d: int, order: str = "row") -> np.ndarray:
    """Render a single base-625 digit (0..624) as a panel.

    The full grid (625 cells) is also accepted so that callers using the
    626-state "base-626" convention (calculator, number viewer) can render
    their top digit; it is NOT a valid base-625 digit and is not a token in
    the Symbolic-Transformers vocabulary.
    """
    if not (0 <= d <= GRID_CELLS):
        raise ValueError(f"Digit must be in [0, {GRID_CELLS}] (0..624 are base-625 digits; {GRID_CELLS} is the reserved full grid)")
    return fill_k_cells(d, order=order, n=GRID_N)


def decode_number_from_count(grid: np.ndarray) -> int:
    """Inverse of encode_number_as_count: simply count black cells."""
    return count_filled(grid)


def encode_symbol(name: str) -> np.ndarray:
    up = name.upper()
    if up not in SYMBOLS:
        raise KeyError(f"Unknown symbol {name}. Known: {list(SYMBOLS.keys())}")
    return SYMBOLS[up].copy()


# -----------------------------
# Expression utilities
# -----------------------------

def parse_simple_expr(expr: str) -> List[str]:
    """
    Tokenize a very simple expression with integers and +-*/^()=.
    E.g. "12 + 3*4 = 24"
    Returns a list like ["12", "+", "3", "*", "4", "=", "24"]
    """
    out = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            out.append(expr[i:j])
            i = j
            continue
        if ch in "+-*/^=()":
            out.append(ch)
            i += 1
            continue
        raise ValueError(f"Unsupported character: {ch!r}")
    return out


SYMBOL_NAME_BY_CHAR = {
    "+": "PLUS",
    "-": "MINUS",
    "*": "TIMES",
    "/": "DIV",
    "=": "EQUALS",
    "^": "POW",
    "(": "LPAREN",
    ")": "RPAREN",
}


def encode_expr_to_panels(expr: str, order: str = "row", as_digits: bool = True) -> List[np.ndarray]:
    """
    Convert an ASCII expression to a sequence of 25x25 panels.
    Numbers become either:
      • as_digits=True  -> base-625 digits (each 0..624) across multiple panels
      • as_digits=False -> single panel in count-mode (0..625)  [WARNING: may overflow 625]
    Operators become their glyph panels.
    """
    tokens = parse_simple_expr(expr)
    panels: List[np.ndarray] = []
    for t in tokens:
        if t.isdigit():
            val = int(t)
            if as_digits:
                digits = int_to_base625(val)  # list of base-625 digits
                for d in digits:
                    panels.append(encode_digit(d, order=order))
            else:
                if val > GRID_CELLS:
                    raise ValueError(f"Value {val} exceeds 625 in count-mode; use digits mode.")
                panels.append(encode_number_as_count(val, order=order))
        else:
            name = SYMBOL_NAME_BY_CHAR[t]
            panels.append(encode_symbol(name))
    return panels


# -----------------------------
# Dataset writer
# -----------------------------

def write_dataset_strip(expr: str, out_png: str, out_json: Optional[str] = None,
                        order: str = "row", as_digits: bool = True,
                        cell_px: int = 20, draw_grid: bool = True) -> None:
    panels = encode_expr_to_panels(expr, order=order, as_digits=as_digits)
    img = strip_to_image(panels, gap_px=8, cell_px=cell_px, draw_grid=draw_grid)
    img.save(out_png)
    if out_json:
        meta = {
            "expr": expr,
            "order": order,
            "as_digits": as_digits,
            "cell_px": cell_px,
            "panels": [{
                "kind": ("DIGIT" if p.sum() <= 624 else "NUMBER") if decode_symbol(p) is None else decode_symbol(p),
                "count": int(p.sum())
            } for p in panels]
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


# -----------------------------
# CLI
# -----------------------------

def cli():
    p = argparse.ArgumentParser(description="Base-625 grid framework (25×25)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # encode-number (count-mode)
    p_num = sub.add_parser("encode-number", help="Encode a single number (0..625) as one panel (count-mode).")
    p_num.add_argument("value", type=int)
    p_num.add_argument("--order", choices=list(ORDER_FACTORIES.keys()), default="row")
    p_num.add_argument("--cell-px", type=int, default=20)
    p_num.add_argument("--no-grid", action="store_true")
    p_num.add_argument("-o", "--out", required=True)

    # encode-digit (base-625 digit)
    p_dig = sub.add_parser("encode-digit", help="Encode a single base-625 digit (0..624).")
    p_dig.add_argument("value", type=int)
    p_dig.add_argument("--order", choices=list(ORDER_FACTORIES.keys()), default="row")
    p_dig.add_argument("--cell-px", type=int, default=20)
    p_dig.add_argument("--no-grid", action="store_true")
    p_dig.add_argument("-o", "--out", required=True)

    # encode-expr
    p_expr = sub.add_parser("encode-expr", help="Encode an ASCII expression into a horizontal strip image.")
    p_expr.add_argument("expr", type=str)
    p_expr.add_argument("--order", choices=list(ORDER_FACTORIES.keys()), default="row")
    p_expr.add_argument("--as-digits", action="store_true", help="Render numbers as base-625 digit panels.")
    p_expr.add_argument("--count-mode", dest="as_digits", action="store_false", help="Render each number in a single count-mode panel (0..625).")
    p_expr.add_argument("--cell-px", type=int, default=20)
    p_expr.add_argument("--no-grid", action="store_true")
    p_expr.add_argument("-o", "--out", required=True)
    p_expr.add_argument("--meta", help="Optional JSON metadata path.")

    # decode-number
    p_dec = sub.add_parser("decode-number", help="Decode a single panel by counting black cells.")
    p_dec.add_argument("png", type=str)

    # symbol-sheet
    p_sy = sub.add_parser("symbol-sheet", help="Export a sheet with all operator glyphs.")
    p_sy.add_argument("-o", "--out", required=True)
    p_sy.add_argument("--cell-px", type=int, default=20)
    p_sy.add_argument("--no-grid", action="store_true")

    args = p.parse_args()

    if args.cmd == "encode-number":
        grid = encode_number_as_count(args.value, order=args.order)
        img = grid_to_image(grid, cell_px=args.cell_px, draw_grid=(not args.no_grid))
        img.save(args.out)
        print(f"Saved {args.out}")

    elif args.cmd == "encode-digit":
        grid = encode_digit(args.value, order=args.order)
        img = grid_to_image(grid, cell_px=args.cell_px, draw_grid=(not args.no_grid))
        img.save(args.out)
        print(f"Saved {args.out}")

    elif args.cmd == "encode-expr":
        panels = encode_expr_to_panels(args.expr, order=args.order, as_digits=args.as_digits)
        img = strip_to_image(panels, cell_px=args.cell_px, draw_grid=(not args.no_grid))
        img.save(args.out)
        if args.meta:
            meta = {
                "expr": args.expr,
                "order": args.order,
                "as_digits": bool(args.as_digits),
                "panels": [{"is_symbol": decode_symbol(p) is not None,
                            "symbol": decode_symbol(p),
                            "black_cells": int(p.sum())}
                           for p in panels]
            }
            with open(args.meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        print(f"Saved {args.out}")

    elif args.cmd == "decode-number":
        # Simple loader + count
        im = Image.open(args.png).convert("RGB")
        # Infer cell size & grid by splitting consistently — assumes the image was generated by this script.
        # Find cell size by dividing width by 25 (approx)
        cell_px = round((im.width - 1) / GRID_N)
        # Rebuild boolean grid by sampling cell center
        g = empty_grid()
        for r in range(GRID_N):
            for c in range(GRID_N):
                x = c * cell_px + cell_px // 2
                y = r * cell_px + cell_px // 2
                if x >= im.width:
                    x = im.width - 1
                if y >= im.height:
                    y = im.height - 1
                pix = im.getpixel((x, y))
                g[r, c] = (sum(pix) < 255 * 3 / 2)  # darker = filled
        print(count_filled(g))

    elif args.cmd == "symbol-sheet":
        panels = [SYMBOLS[name] for name in SYMBOLS.keys()]
        img = strip_to_image(panels, cell_px=args.cell_px, draw_grid=(not args.no_grid))
        img.save(args.out)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    cli()
