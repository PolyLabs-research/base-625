"""Placeholders for integrating base-625 glyphs into the spatial brain model.

The goal is to eventually map 25x25 boolean glyph grids into spatial cortical
representations. That likely implies moving from the current 1D cortical line
to a 2D cortical sheet and aligning glyph structure with spatial control fields.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional


def find_base625_root(start: Path | None = None) -> Optional[Path]:
    """Search for base625_grid.py in current or parent directories."""
    if start is None:
        start = Path.cwd()
    for path in [start, *start.parents]:
        candidate = path / "base625_grid.py"
        if candidate.exists():
            return path
    return None


def try_import_base625() -> bool:
    """Attempt a lightweight import to confirm base-625 availability."""
    try:
        __import__("base625_grid")
        return True
    except Exception:
        return False


def suggest_sheet_shape() -> tuple[int, int]:
    """Return a plausible cortical sheet size for 25x25 glyphs."""
    return 25, 25


def locate_glyph_assets(start: Path | None = None) -> Optional[Path]:
    """Return a path pointing to a data directory near base625 if present."""
    root = find_base625_root(start=start)
    if root is None:
        return None
    data_dir = root / "data"
    return data_dir if data_dir.exists() else root


def maybe_import_base625():
    """Return the base625_grid module if available, otherwise None."""
    try:
        import base625_grid  # type: ignore

        return base625_grid
    except Exception:
        return None


def _plus_shape(height: int, width: int) -> np.ndarray:
    glyph = np.zeros((height, width), dtype=np.float32)
    thickness = max(1, min(height, width) // 8)
    cy, cx = height // 2, width // 2
    glyph[max(cy - thickness, 0) : min(cy + thickness + 1, height), :] = 1.0
    glyph[:, max(cx - thickness, 0) : min(cx + thickness + 1, width)] = 1.0
    return glyph


def _horizontal_bar(height: int, width: int) -> np.ndarray:
    glyph = np.zeros((height, width), dtype=np.float32)
    band = max(1, height // 7)
    cy = height // 2
    glyph[max(cy - band, 0) : min(cy + band + 1, height), :] = 1.0
    return glyph


def _x_shape(height: int, width: int) -> np.ndarray:
    glyph = np.zeros((height, width), dtype=np.float32)
    for i in range(min(height, width)):
        glyph[i, i] = 1.0
        glyph[i, width - i - 1] = 1.0
    return glyph


def _vertical_bar(height: int, width: int) -> np.ndarray:
    glyph = np.zeros((height, width), dtype=np.float32)
    band = max(1, width // 7)
    cx = width // 2
    glyph[:, max(cx - band, 0) : min(cx + band + 1, width)] = 1.0
    return glyph


def _box_shape(height: int, width: int) -> np.ndarray:
    glyph = np.zeros((height, width), dtype=np.float32)
    thickness = max(1, min(height, width) // 10)
    glyph[:thickness, :] = 1.0
    glyph[-thickness:, :] = 1.0
    glyph[:, :thickness] = 1.0
    glyph[:, -thickness:] = 1.0
    return glyph


def _diag_bar(height: int, width: int, reverse: bool = False) -> np.ndarray:
    glyph = np.zeros((height, width), dtype=np.float32)
    thickness = max(1, min(height, width) // 12)
    for offset in range(-thickness, thickness + 1):
        for y in range(height):
            x = (width - 1 - y + offset) if reverse else (y + offset)
            if 0 <= x < width:
                glyph[y, x] = 1.0
    return glyph


def _random_blob(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.random((height + 2, width + 2))
    smooth = (
        noise[:-2, :-2]
        + noise[1:-1, :-2]
        + noise[2:, :-2]
        + noise[:-2, 1:-1]
        + noise[1:-1, 1:-1]
        + noise[2:, 1:-1]
        + noise[:-2, 2:]
        + noise[1:-1, 2:]
        + noise[2:, 2:]
    ) / 9.0
    threshold = 0.5 + 0.1 * rng.random()
    return (smooth > threshold).astype(np.float32)


def _maybe_collect_base625_glyphs(num_items: int, height: int, width: int) -> list[np.ndarray]:
    glyphs: list[np.ndarray] = []
    base_mod = maybe_import_base625()
    if base_mod is None:
        return glyphs

    try:
        symbol_dict = getattr(base_mod, "SYMBOLS", {})
        for glyph in symbol_dict.values():
            arr = np.asarray(glyph)
            if arr.shape != (height, width):
                continue
            glyphs.append(arr.astype(np.float32))
            if len(glyphs) >= num_items:
                break
    except Exception:
        glyphs = []
    return glyphs


def build_glyph_bank(
    num_items: int,
    height: int,
    width: int,
    difficulty: str = "hard",
) -> np.ndarray:
    """Construct a glyph bank with simple synthetic shapes.

    difficulty:
        easy: first 3 structured glyphs (plus, horizontal bar, X); num_items must be <= 3.
        hard: uses the same first three glyphs and fills additional slots with simple shapes
              or smooth random blobs until num_items is reached.
    """
    if difficulty not in {"easy", "hard"}:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    if difficulty == "easy" and num_items > 3:
        raise ValueError("Easy difficulty supports at most 3 glyphs.")

    rng = np.random.default_rng(0)
    glyph_funcs = [_plus_shape, _horizontal_bar, _x_shape, _vertical_bar, _box_shape, _diag_bar]

    glyphs: list[np.ndarray] = []
    source_glyphs = _maybe_collect_base625_glyphs(num_items, height, width)
    glyphs.extend(source_glyphs[:num_items])

    for func in glyph_funcs:
        if len(glyphs) >= num_items:
            break
        glyphs.append(func(height, width))

    if difficulty == "hard":
        alt_diag = _diag_bar(height, width, reverse=True)
        if len(glyphs) < num_items:
            glyphs.append(alt_diag)
        while len(glyphs) < num_items:
            glyphs.append(_random_blob(height, width, rng))

    # Trim in case base-625 import filled more than requested.
    glyphs = glyphs[:num_items]
    return np.stack(glyphs, axis=0).astype(np.float32)
