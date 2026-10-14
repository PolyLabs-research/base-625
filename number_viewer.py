# number_viewer.py
#
# Simple GUI: enter a number, render its Base-626 digit panels as 25×25 grids.
#
# Requirements: pillow, numpy (already installed for your project).
#
# Usage:
#   py -3.12 number_viewer.py
#
# In the GUI:
#   • Enter a number (0, 1, 42, 123456…)
#   • Click "Render"
#   • The Base-626 digit panels appear side by side.

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from PIL import Image, ImageTk

import numpy as np
from base625_grid import encode_digit, encode_number_as_count, grid_to_image, GRID_CELLS

CELL_PX = 20   # pixel size for each cell
ORDER = "row"  # fill order: row | snake | random42

def digit_to_grid(d: int, order: str = ORDER) -> np.ndarray:
    """Grid for one base-626 digit: 0..624 via digit mode, 625 via full-grid count mode."""
    if 0 <= d <= 624:
        return encode_digit(d, order=order)
    if d == GRID_CELLS:
        return encode_number_as_count(GRID_CELLS, order=order)
    raise ValueError(f"digit out of range for base-626: {d}")


def int_to_base626(n: int) -> list[int]:
    """Convert int -> list of base-626 digits (0..625), most significant first."""
    if n == 0:
        return [0]
    digits = []
    base = GRID_CELLS + 1  # 626
    while n > 0:
        n, r = divmod(n, base)
        digits.append(r)
    return list(reversed(digits))


class NumberViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Base-626 Number Viewer")
        self.geometry("800x400")

        # Top entry + button
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Enter number:").pack(side=tk.LEFT)
        self.entry = ttk.Entry(top, width=20)
        self.entry.pack(side=tk.LEFT, padx=(4, 12))
        self.entry.insert(0, "42")

        ttk.Button(top, text="Render", command=self.on_render).pack(side=tk.LEFT, padx=4)

        # Canvas for image strip
        self.canvas = tk.Canvas(self, bg="#ddd")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.tk_img = None  # keep reference

    def on_render(self):
        text = self.entry.get().strip()
        if not text.isdigit():
            messagebox.showerror("Invalid input", "Please enter a non-negative integer.")
            return

        val = int(text)
        if val < 0:
            messagebox.showerror("Invalid input", "Only non-negative integers supported.")
            return

        # Convert to base-626 digits
        digits = int_to_base626(val)

        # Render panels (digit 625 is the full grid, rendered in count mode)
        panels = [digit_to_grid(d, order=ORDER) for d in digits]
        rendered = [grid_to_image(p, cell_px=CELL_PX, draw_grid=True) for p in panels]

        # Stitch horizontally
        total_w = sum(im.width for im in rendered) + 8 * (len(rendered) - 1)
        total_h = max(im.height for im in rendered)
        out = Image.new("RGB", (total_w, total_h), (240, 240, 240))
        x = 0
        for im in rendered:
            out.paste(im, (x, 0))
            x += im.width + 8

        # Show in canvas
        self.tk_img = ImageTk.PhotoImage(out)
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, out.width, out.height))
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)


def main():
    app = NumberViewer()
    app.mainloop()

if __name__ == "__main__":
    main()
