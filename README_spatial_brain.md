# Spatial Brain Playground

This is a small "Spatial Computing" synthetic brain playground that lives alongside the existing base-625 project. It focuses on a 1D cortical line with content vs control separation:

- Content lives in a set of cortical columns, each holding a continuous vector.
- Beta fields implement top-down gating (high beta closes a region); gamma fields provide content expression (anti-correlated with beta).
- The first task is a 3-slot ordered working-memory "combination lock".

Run the first experiment:

```
python -m spatial_brain.experiments.wm_3slot_1d
```

Or try the 2D cortical sheet variant:

```
python -m spatial_brain.experiments.wm_3slot_2d
```

Planned extensions include integrating base-625 glyphs and extending from a 1D line to a 2D cortical sheet. Existing base-625 code is left untouched.

## 2D sheet presets and glyph-mode experiment

Spatial brain now ships with standard 2D cortical sheet presets:

- `small`: 8×8 sheet with 32-dim content vectors.
- `medium`: 16×16 sheet with 64-dim content vectors.
- `glyph`: 25×25 sheet tuned for glyph experiments (64-dim content).

You can select these via `--sheet-preset` on the 2D experiments. A new glyph-mode runner stores synthetic or base-625 glyphs in the three WM slots and visualises the resulting beta maps:

```
python -m spatial_brain.experiments.glyph_slots_25x25 --num-epochs 200 --beta-smooth-lambda 0.02
```
