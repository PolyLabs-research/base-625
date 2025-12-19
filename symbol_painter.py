# -*- coding: utf-8 -*-
# symbol_painter.py
#
# A 25×25 boolean grid GUI to hand-design math symbols for your Base-625 canvas.
# - Click/drag to toggle cells (Left = draw, Right = erase; or click to toggle).
# - "Label (ASCII)" is a machine-safe label (e.g., PLUS, MINUS, SQRT). Used for filenames.
# - "Symbol (pretty)" is the human symbol (e.g., +, −, √). Stored in decode table.
# - "Center pattern" auto-centers the drawn pixels inside the 25×25 box.
# - "Save" writes PNG to out/symbols and upserts a decode table JSON/CSV.
#
# Requirements: Python 3.12+, pillow, numpy (pip install pillow numpy)
#
# Usage:
#   py -3.12 symbol_painter.py
#
# Files created:
#   out/symbols/<label>.png
#   out/symbols/decode_table_symbols.json   (UTF-8 with real Unicode glyphs)
#   out/symbols/decode_table_symbols.csv

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import ttk, messagebox

GRID_N = 25
DEFAULT_CELL_PX = 20

OUT_DIR = Path("out") / "symbols"
DECODE_JSON = OUT_DIR / "decode_table_symbols.json"
DECODE_CSV = OUT_DIR / "decode_table_symbols.csv"


def sanitize_label(label: str) -> str:
    """Safe filename from label (keep letters, digits, _-)."""
    label = (label or "").strip()
    if not label:
        return ""
    label = label.replace(" ", "_")
    label = re.sub(r"[^A-Za-z0-9_\-]+", "", label)
    return label or ""


def grid_to_image(grid: np.ndarray, cell_px: int = DEFAULT_CELL_PX, draw_grid: bool = True) -> Image.Image:
    n = grid.shape[0]
    W = n * cell_px
    H = n * cell_px
    img = Image.new("RGB", (W + 1, H + 1), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # paint cells
    for r in range(n):
        for c in range(n):
            x0 = c * cell_px
            y0 = r * cell_px
            x1 = x0 + cell_px
            y1 = y0 + cell_px
            fill = (0, 0, 0) if grid[r, c] else (255, 255, 255)
            draw.rectangle([x0, y0, x1, y1], fill=fill)

    # gridlines
    if draw_grid:
        for c in range(n + 1):
            x = c * cell_px
            draw.line([(x, 0), (x, H)], fill=(0, 0, 0), width=1)
        for r in range(n + 1):
            y = r * cell_px
            draw.line([(0, y), (W, y)], fill=(0, 0, 0), width=1)

    return img


def sha256_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def center_pattern(grid: np.ndarray) -> np.ndarray:
    """Return a new 25×25 grid with the drawn pixels centered."""
    n = grid.shape[0]
    rows = np.any(grid, axis=1)
    cols = np.any(grid, axis=0)
    if not rows.any() or not cols.any():
        return grid.copy()  # nothing drawn

    r0, r1 = np.where(rows)[0][0], np.where(rows)[0][-1]
    c0, c1 = np.where(cols)[0][0], np.where(cols)[0][-1]
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    cropped = grid[r0:r1+1, c0:c1+1]
    out = np.zeros((n, n), dtype=bool)

    r_start = (n - h) // 2
    c_start = (n - w) // 2
    out[r_start:r_start+h, c_start:c_start+w] = cropped
    return out


@dataclass
class AppState:
    grid: np.ndarray
    drawing: bool = False
    draw_value: bool = True  # what to paint while dragging (True=black, False=white)


class SymbolPainter(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Symbol Painter — 25×25")
        self.state = AppState(grid=np.zeros((GRID_N, GRID_N), dtype=bool))
        self.cell_px_var = tk.IntVar(value=DEFAULT_CELL_PX)
        self.gridlines_var = tk.BooleanVar(value=True)
        self.autocenter_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._redraw()

    # ---------- UI ----------
    def _build_ui(self):
        # Top controls (labels & actions)
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        # Label (ASCII)
        ttk.Label(top, text="Label (ASCII):").pack(side=tk.LEFT)
        self.label_entry = ttk.Entry(top, width=16)
        self.label_entry.pack(side=tk.LEFT, padx=(4, 12))
        self.label_entry.insert(0, "")  # e.g., PLUS, MINUS, SQRT

        # Symbol (pretty)
        ttk.Label(top, text="Symbol (pretty):").pack(side=tk.LEFT)
        self.symbol_entry = ttk.Entry(top, width=10)
        self.symbol_entry.pack(side=tk.LEFT, padx=(4, 12))
        self.symbol_entry.insert(0, "")  # e.g., +, −, √

        ttk.Button(top, text="Save", command=self.on_save).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Clear", command=self.on_clear).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Invert", command=self.on_invert).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Center pattern", command=self.on_center_once).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Cell px").pack(side=tk.LEFT, padx=(12, 2))
        self.cell_spin = ttk.Spinbox(top, from_=8, to=64, textvariable=self.cell_px_var, width=5, command=self._redraw)
        self.cell_spin.pack(side=tk.LEFT)

        ttk.Checkbutton(top, text="Gridlines", variable=self.gridlines_var, command=self._redraw).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(top, text="Auto-center on save", variable=self.autocenter_var).pack(side=tk.LEFT, padx=8)

        # Canvas
        self.canvas = tk.Canvas(self, bg="#ffffff")
        self.canvas.pack(side=tk.TOP, padx=8, pady=8)

        self.canvas.bind("<Button-1>", self.on_click_left)
        self.canvas.bind("<B1-Motion>", self.on_drag_left)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_click_right)
        self.canvas.bind("<B3-Motion>", self.on_drag_right)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        # Status
        self.status = tk.StringVar(value="Left-drag = draw • Right-drag = erase • Click toggles")
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor="w", padding=6).pack(side=tk.BOTTOM, fill=tk.X)

    # ---------- Grid helpers ----------
    def _grid_dims(self) -> Tuple[int, int]:
        n = GRID_N
        px = self.cell_px_var.get()
        width = n * px + 1
        height = n * px + 1
        return width, height

    def _cell_at(self, x: int, y: int) -> Tuple[int, int] | None:
        px = self.cell_px_var.get()
        r = y // px
        c = x // px
        if 0 <= r < GRID_N and 0 <= c < GRID_N:
            return r, c
        return None

    def _redraw(self):
        W, H = self._grid_dims()
        self.canvas.config(width=W, height=H)
        self.canvas.delete("all")

        px = self.cell_px_var.get()
        g = self.state.grid

        # Draw cells
        for r in range(GRID_N):
            for c in range(GRID_N):
                x0 = c * px
                y0 = r * px
                x1 = x0 + px
                y1 = y0 + px
                fill = "#000000" if g[r, c] else "#ffffff"
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="", fill=fill)

        # Gridlines
        if self.gridlines_var.get():
            for c in range(GRID_N + 1):
                x = c * px
                self.canvas.create_line(x, 0, x, H, fill="#000000")
            for r in range(GRID_N + 1):
                y = r * px
                self.canvas.create_line(0, y, W, y, fill="#000000")

        blacks = int(self.state.grid.sum())
        self.status.set(f"Cells filled: {blacks} • Left-drag=draw • Right-drag=erase • Click toggles")

    # ---------- Events ----------
    def on_click_left(self, event):
        cell = self._cell_at(event.x, event.y)
        if not cell:
            return
        r, c = cell
        # Toggle on click, but set drag mode to draw=True
        self.state.grid[r, c] = not self.state.grid[r, c]
        self.state.drawing = True
        self.state.draw_value = True
        self._redraw()

    def on_drag_left(self, event):
        if not self.state.drawing:
            return
        cell = self._cell_at(event.x, event.y)
        if not cell:
            return
        r, c = cell
        self.state.grid[r, c] = True
        self._redraw()

    def on_click_right(self, event):
        cell = self._cell_at(event.x, event.y)
        if not cell:
            return
        r, c = cell
        self.state.grid[r, c] = False
        self.state.drawing = True
        self.state.draw_value = False
        self._redraw()

    def on_drag_right(self, event):
        if not self.state.drawing:
            return
        cell = self._cell_at(event.x, event.y)
        if not cell:
            return
        r, c = cell
        self.state.grid[r, c] = False
        self._redraw()

    def on_release(self, _event):
        self.state.drawing = False

    # ---------- Buttons ----------
    def on_clear(self):
        self.state.grid[:, :] = False
        self._redraw()

    def on_invert(self):
        self.state.grid[:, :] = ~self.state.grid
        self._redraw()

    def on_center_once(self):
        self.state.grid = center_pattern(self.state.grid)
        self._redraw()

    def on_save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        # Grab fields
        label_raw = (self.label_entry.get() or "").strip()
        symbol_raw = (self.symbol_entry.get() or "").strip()

        # Fallbacks
        # - If label missing but symbol present, derive a label from symbol name/codepoint
        if not label_raw and symbol_raw:
            # Try to make a reasonable label from symbol (e.g., √ -> SQRT)
            fallback = {
                "+": "PLUS", "-": "MINUS", "×": "TIMES", "*": "TIMES", "·": "DOT",
                "/": "DIV", "÷": "DIV", "=": "EQUALS", "^": "POW",
                "(": "LPAREN", ")": "RPAREN", ".": "DOT", "√": "SQRT", "!": "FACTORIAL"
            }.get(symbol_raw, f"SYM_{ord(symbol_raw):04X}")
            label_raw = fallback

        # If symbol is missing, default it to label_raw (so CSV/JSON always has something)
        if not symbol_raw:
            symbol_raw = label_raw

        # Filename-safe label
        label = sanitize_label(label_raw)
        if not label:
            ts = time.strftime("%Y%m%d_%H%M%S")
            label = f"sym_{ts}"

        grid = self.state.grid
        if self.autocenter_var.get():
            grid = center_pattern(grid)

        img = grid_to_image(grid, cell_px=self.cell_px_var.get(), draw_grid=self.gridlines_var.get())
        sha = sha256_png(img)

        fpath = OUT_DIR / f"{label}.png"
        # If file exists, confirm overwrite
        if fpath.exists():
            ans = messagebox.askyesno("Overwrite?", f"'{fpath.name}' exists. Overwrite and update decode table?")
            if not ans:
                return
        img.save(fpath)

        # Save / update decode tables
        record = {
            "label": label_raw,               # machine-ish name as entered
            "symbol": symbol_raw,             # pretty glyph
            "filename": fpath.name,
            "path": str(fpath),
            "sha256": sha,
            "black_cells": int(grid.sum()),
            "cell_px": int(self.cell_px_var.get()),
            "gridlines": bool(self.gridlines_var.get()),
            "autocentered": bool(self.autocenter_var.get()),
        }
        self._upsert_decode_tables(record)
        self.status.set(f"Saved {fpath.name} • cells={record['black_cells']} • sha256={sha[:12]}…")

    # ---------- Decode table I/O ----------
    def _upsert_decode_tables(self, record: dict):
        # JSON
        items = []
        if DECODE_JSON.exists():
            try:
                with open(DECODE_JSON, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    items = obj.get("items", [])
            except Exception:
                items = []

        # replace by filename OR label (case-insensitive)
        def keyer(x):
            return (x.get("filename", "").lower(), (x.get("label") or "").strip().lower())

        new_key = keyer(record)
        filtered = [r for r in items if keyer(r) != new_key]
        filtered.append(record)
        with open(DECODE_JSON, "w", encoding="utf-8") as f:
            # ensure_ascii=False so √, ≤, ≥, ∑ etc. are written as real characters
            json.dump({"items": filtered}, f, indent=2, ensure_ascii=False)

        # CSV
        fieldnames = ["label", "symbol", "filename", "path", "sha256", "black_cells", "cell_px", "gridlines", "autocentered"]
        rows = filtered
        with open(DECODE_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    # Guarantee output dir exists for first run
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = SymbolPainter()
    app.mainloop()


if __name__ == "__main__":
    main()
