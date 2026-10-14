# save as: export_0_625.py
# Usage examples (PowerShell):
#   py -3.12 export_0_625.py --out "out\\digits_0_625_row" --order row --cell-px 20
#   py -3.12 export_0_625.py --out "out\\digits_0_625_rand" --order random42 --no-grid

from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path

from base625_grid import (
    GRID_CELLS,
    encode_number_as_count,
    encode_digit,
    grid_to_image,
)

def sha256_of_pil(img) -> str:
    # Save to bytes and hash for integrity / quick matching
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()

def main():
    ap = argparse.ArgumentParser("Export 25×25 symbols for values 0..625 with a decode table.")
    ap.add_argument("--out", required=True, help="Output folder for images + decode table files.")
    ap.add_argument("--order", choices=["row", "snake", "random42"], default="row",
                    help="Cell fill order used for rendering.")
    ap.add_argument("--cell-px", type=int, default=20, help="Cell size in pixels (default 20).")
    ap.add_argument("--no-grid", action="store_true", help="Hide gridlines in output images.")
    ap.add_argument("--mode", choices=["count", "digit"], default="count",
                    help="count = k black cells; digit = base-625 digit panel. (Default: count)")
    ap.add_argument("--prefix", default="sym_", help="Filename prefix (default: sym_)")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for k in range(0, GRID_CELLS + 1):  # 0..625 inclusive
        if args.mode == "count":
            grid = encode_number_as_count(k, order=args.order)
        else:
            # NOTE: base625_grid.encode_digit supports 0..624.
            # If you want digit=625, first extend encode_digit() as discussed.
            if k > 624:
                # still export last one using count so you always get 626 files
                grid = encode_number_as_count(k, order=args.order)
            else:
                grid = encode_digit(k, order=args.order)

        img = grid_to_image(grid, cell_px=args.cell_px, draw_grid=not args.no_grid)
        fname = f"{args.prefix}{k:03d}.png"
        fpath = outdir / fname
        img.save(fpath)

        rows.append({
            "number": k,
            "filename": fname,
            "path": fpath.as_posix(),  # forward slashes so tables work on every OS
            "mode": args.mode if k <= 624 else "count",
            "order": args.order,
            "cell_px": args.cell_px,
            "gridlines": not args.no_grid,
            "black_cells": int(grid.sum()),  # equals k in count-mode
            "sha256": sha256_of_pil(img),
        })

    # Write JSON
    with open(outdir / "decode_table.json", "w", encoding="utf-8") as f:
        json.dump({"items": rows}, f, indent=2)

    # Write CSV (handy in Excel/Power BI)
    with open(outdir / "decode_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Done. Wrote {len(rows)} images + decode_table.json/csv to {outdir}")

if __name__ == "__main__":
    main()
