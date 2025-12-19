# base626_calculator.py
#
# GUI calculator that:
#  • accepts a base-10 expression with normal symbols (+ - * / ^ ( ) [optional "="])
#  • evaluates it safely (supports + - * / ^ and parentheses; integers only)
#  • displays the expression as panels (digits in base-626, operators via your symbols JSON)
#  • displays "= <answer>" as panels, and the plain-text answer
#
# Requirements: pillow, numpy, your base625_grid.py, your symbols JSON generated earlier.
#
# Run:
#   py -3.12 base626_calculator.py
#
# Notes:
#  • Uses row-major fill by default; you can switch order in the dropdown.
#  • For base-626 digits: 0..625 is one digit; we render d=0..624 via encode_digit,
#    and d=625 via count-mode panel (625 filled cells).

from __future__ import annotations

import json
import re
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk, messagebox

from PIL import Image, ImageTk

# Import your renderer
from base625_grid import (
    GRID_CELLS,
    GRID_N,
    encode_digit,
    encode_number_as_count,
    grid_to_image,
)

# ---------- Config ----------
SYMBOLS_JSON = Path("out") / "symbols" / "decode_table_symbols.json"
DEFAULT_ORDER = "row"         # row | snake | random42
DEFAULT_CELL_PX = 20
DEFAULT_GAP_PX = 8            # default gap between panels in a strip

# ---------- Utility: safe eval ----------
# We support integers, +, -, *, /, ^, parentheses.
TOKEN_RE = re.compile(r"\s*([0-9]+|[\+\-\*\/\^\(\)=])\s*")

def tokenize(expr: str):
    pos = 0
    out = []
    while pos < len(expr):
        m = TOKEN_RE.match(expr, pos)
        if not m:
            raise ValueError(f"Invalid character near: {expr[pos:pos+12]!r}")
        tok = m.group(1)
        out.append(tok)
        pos = m.end()
    return out

def eval_safe(expr: str) -> int | float:
    """Evaluate a simple arithmetic expression safely."""
    # If the user typed '=', evaluate only the left side for the result display.
    if "=" in expr:
        expr = expr.split("=", 1)[0]

    # Replace caret with Python power
    expr = expr.replace("^", "**")

    # Only allow digits/operators/parentheses
    if not re.fullmatch(r"[0-9\+\-\*\/\^\(\)\s\=]*", expr.replace("**", "^")):
        raise ValueError("Only digits and + - * / ^ ( ) and = are supported.")

    # Evaluate in a restricted namespace
    try:
        # Use eval with empty globals and no builtins
        val = eval(expr, {"__builtins__": {}}, {})
    except Exception as e:
        raise ValueError(f"Could not evaluate: {e}") from e

    return val

# ---------- Load symbol images ----------
@dataclass
class SymbolGlyph:
    symbol: str     # pretty glyph, e.g., '+'
    path: Path      # PNG path
    image: Image.Image

def load_symbols(json_path: Path) -> dict[str, SymbolGlyph]:
    if not json_path.exists():
        raise FileNotFoundError(f"Missing symbols JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    table: dict[str, SymbolGlyph] = {}
    for item in obj.get("items", []):
        sym = (item.get("symbol") or "").strip()
        p = Path(item.get("path", ""))
        if sym and p.exists():
            try:
                img = Image.open(p).convert("RGB")
                table[sym] = SymbolGlyph(symbol=sym, path=p, image=img)
            except Exception:
                pass
    return table

# We’ll map user-typed operators to the glyph symbol to use from the JSON.
def map_op_to_glyph(op: str, table: dict[str, SymbolGlyph]) -> str | None:
    """Return the 'symbol' key in the table for a given operator token from input."""
    if op in table:
        return op
    # Prefer nice alternates if direct match not found
    if op == "*":
        for alt in ("×", "x", "·"):
            if alt in table:
                return alt
    return None

# ---------- Base-626 helpers ----------
def int_to_base626(n: int) -> list[int]:
    if n == 0:
        return [0]
    base = GRID_CELLS + 1  # 626
    digs = []
    while n > 0:
        n, r = divmod(n, base)
        digs.append(r)
    return list(reversed(digs))

def digit_to_panel(d: int, order: str, cell_px: int, draw_grid: bool = True) -> Image.Image:
    """Render one base-626 digit to a 25×25 panel image at the requested cell size."""
    if 0 <= d <= 624:
        return grid_to_image(encode_digit(d, order=order), cell_px=cell_px, draw_grid=draw_grid)
    elif d == 625:
        # Use count-mode to render 625 (full grid)
        return grid_to_image(encode_number_as_count(625, order=order), cell_px=cell_px, draw_grid=draw_grid)
    else:
        raise ValueError("Digit out of range for base-626")

def scale_panel_to_cell_px(img: Image.Image, cell_px: int) -> Image.Image:
    """Scale a 25×25 panel image to match the given cell size, keeping crisp edges."""
    target_w = GRID_N * cell_px + 1
    target_h = GRID_N * cell_px + 1
    if img.width == target_w and img.height == target_h:
        return img
    return img.resize((target_w, target_h), resample=Image.NEAREST)

# ---------- Stitching ----------
def stitch_h(images: list[Image.Image], gap_px: int, bg=(240, 240, 240)) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), bg)
    w = sum(im.width for im in images) + gap_px * (len(images) - 1)
    h = max(im.height for im in images)
    out = Image.new("RGB", (w, h), bg)
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width + gap_px
    return out

# ---------- GUI ----------
class Calculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Base-626 Calculator")
        self.geometry("1100x720")

        # state
        self.symbols = load_symbols(SYMBOLS_JSON)
        self.tk_expr = None
        self.tk_ans = None

        # top controls
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Expression:").pack(side=tk.LEFT)
        self.entry = ttk.Entry(top, width=46)
        self.entry.pack(side=tk.LEFT, padx=(4, 12))
        self.entry.insert(0, "12 + 3*(4^2)")

        ttk.Button(top, text="Render", command=self.on_render).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Order:").pack(side=tk.LEFT, padx=(12, 4))
        self.order_var = tk.StringVar(value=DEFAULT_ORDER)
        ttk.Combobox(top, textvariable=self.order_var, values=["row", "snake", "random42"], width=10, state="readonly").pack(side=tk.LEFT)

        ttk.Label(top, text="Cell px:").pack(side=tk.LEFT, padx=(12, 4))
        self.cell_var = tk.IntVar(value=DEFAULT_CELL_PX)
        ttk.Spinbox(top, from_=6, to=48, textvariable=self.cell_var, width=6, command=self._refresh_if_rendered).pack(side=tk.LEFT)

        ttk.Label(top, text="Gap px:").pack(side=tk.LEFT, padx=(12, 4))
        self.gap_var = tk.IntVar(value=DEFAULT_GAP_PX)
        ttk.Spinbox(top, from_=0, to=24, textvariable=self.gap_var, width=6, command=self._refresh_if_rendered).pack(side=tk.LEFT)

        # result text
        self.result_var = tk.StringVar(value="Result: ")
        ttk.Label(self, textvariable=self.result_var, padding=(8, 2)).pack(side=tk.TOP, anchor="w")

        # canvases
        ttk.Label(self, text="Expression panels:", padding=(8, 2)).pack(side=tk.TOP, anchor="w")
        self.canvas_expr = tk.Canvas(self, bg="#dddddd", height=240)
        self.canvas_expr.pack(fill=tk.X, padx=8, pady=(0, 8))

        ttk.Label(self, text="= Answer panels:", padding=(8, 2)).pack(side=tk.TOP, anchor="w")
        self.canvas_ans = tk.Canvas(self, bg="#dddddd", height=240)
        self.canvas_ans.pack(fill=tk.X, padx=8, pady=(0, 8))

        # internal cache of last render (to rescale on control changes)
        self._last_tokens: list[str] | None = None
        self._last_value: int | float | None = None

    # When cell/gap changes, re-render current expression if any
    def _refresh_if_rendered(self):
        if self._last_tokens is not None:
            self._render_from_cache()

    def _tokens_to_panel_strip(self, tokens: list[str], order: str, cell_px: int, gap_px: int) -> Image.Image:
        """Render the whole token stream to a single horizontal image strip."""
        imgs: list[Image.Image] = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t.isdigit():  # number
                n = int(t)
                digits = int_to_base626(n)
                for d in digits:
                    imgs.append(digit_to_panel(d, order, cell_px=cell_px, draw_grid=True))
                i += 1
                continue

            # operators / parens / equals
            op = t
            if op == "*" and op not in self.symbols:
                mapped = map_op_to_glyph(op, self.symbols)
                if mapped:
                    op = mapped

            glyph = self.symbols.get(op)
            if not glyph:
                raise ValueError(f"No glyph found for operator '{t}'. Add it to symbols JSON.")
            # scale operator glyph to current Cell px
            imgs.append(scale_panel_to_cell_px(glyph.image, cell_px=cell_px))
            i += 1

        return stitch_h(imgs, gap_px=gap_px)

    def on_render(self):
        expr_in = self.entry.get().strip()
        if not expr_in:
            messagebox.showerror("Input needed", "Please enter an expression.")
            return

        try:
            tokens = tokenize(expr_in)
        except ValueError as e:
            messagebox.showerror("Parse error", str(e))
            return

        # Evaluate
        try:
            value = eval_safe(expr_in)
        except ValueError as e:
            messagebox.showerror("Evaluation error", str(e))
            return

        # For panel rendering of the answer we prefer a non-negative integer.
        if isinstance(value, float):
            if not value.is_integer():
                self.result_var.set(f"Result: {value} (non-integer; digits not rendered)")
            else:
                value = int(value)

        if isinstance(value, int) and value < 0:
            self.result_var.set(f"Result: {value} (negative; add MINUS glyph handling if desired)")
        else:
            self.result_var.set(f"Result: {value}")

        # cache state
        self._last_tokens = tokens
        self._last_value = value

        self._render_from_cache()

    def _render_from_cache(self):
        assert self._last_tokens is not None
        tokens = self._last_tokens
        value = self._last_value

        order = self.order_var.get()
        cell_px = int(self.cell_var.get())
        gap_px = int(self.gap_var.get())

        # Render expression panels
        try:
            expr_img = self._tokens_to_panel_strip(tokens, order=order, cell_px=cell_px, gap_px=gap_px)
        except Exception as e:
            messagebox.showerror("Render error (expression)", str(e))
            return

        # Render "= answer" panels
        ans_imgs: list[Image.Image] = []
        # equals glyph
        eq_glyph = self.symbols.get("=")
        if not eq_glyph:
            messagebox.showerror("Missing glyph", "No '=' glyph in symbols JSON.")
            return
        ans_imgs.append(scale_panel_to_cell_px(eq_glyph.image, cell_px=cell_px))

        if isinstance(value, int) and value >= 0:
            for d in int_to_base626(value):
                ans_imgs.append(digit_to_panel(d, order, cell_px=cell_px, draw_grid=True))
        # else: leave just '=' if non-integer or negative

        ans_img = stitch_h(ans_imgs, gap_px=gap_px)

        # Push to canvases
        self.tk_expr = ImageTk.PhotoImage(expr_img)
        self.canvas_expr.delete("all")
        self.canvas_expr.config(scrollregion=(0, 0, expr_img.width, expr_img.height), height=expr_img.height)
        self.canvas_expr.create_image(0, 0, anchor="nw", image=self.tk_expr)

        self.tk_ans = ImageTk.PhotoImage(ans_img)
        self.canvas_ans.delete("all")
        self.canvas_ans.config(scrollregion=(0, 0, ans_img.width, ans_img.height), height=ans_img.height)
        self.canvas_ans.create_image(0, 0, anchor="nw", image=self.tk_ans)


def main():
    app = Calculator()
    app.mainloop()


if __name__ == "__main__":
    main()
