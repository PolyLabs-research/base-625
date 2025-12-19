#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from datetime import datetime
from collections import Counter, defaultdict

# Optional deps
try:
    from PIL import Image  # type: ignore
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import openpyxl  # type: ignore
    OPENPYXL_AVAILABLE = True
except Exception:
    OPENPYXL_AVAILABLE = False

# --------------------------- Defaults ----------------------------------------
DEFAULT_INCLUDE_EXTS = {
    ".py", ".ipynb",
    ".csv", ".tsv", ".xlsx", ".json", ".jsonl",
    ".npz", ".npy",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"
}

DEFAULT_PRUNE_DIRS = {
    ".venv", "venv", "env", ".env",
    "__pycache__", ".ipynb_checkpoints", ".pytest_cache",
    ".git", ".hg", ".svn",
    "node_modules", "dist", "build",
    "site-packages", "lib", "include", "scripts",
    "egg-info", "eggs",
}
DEFAULT_PRUNE_SUBSTR = {"-info", "dist-info", ".egg-info"}

MAX_SCRIPT_BYTES_DEFAULT = 2 * 1024 * 1024  # 2 MB
MAX_JSON_BYTES = 1 * 1024 * 1024

CSV_FIELDS = [
    "item_type", "classification", "relative_path", "name", "ext",
    "size_bytes", "modified_utc", "sha256", "details", "content"
]

SCRIPT_EXTS = {
    ".py", ".ipynb", ".r", ".sql", ".ps1", ".psm1",
    ".sh", ".bash", ".bat", ".cmd", ".js", ".ts", ".jsx", ".tsx",
    ".html", ".htm", ".css", ".cpp", ".cc", ".c", ".h", ".hpp",
    ".cs", ".java", ".go", ".rb", ".php", ".swift", ".kt", ".scala", ".toml"
}
CONFIG_EXTS = {".yml", ".yaml", ".ini", ".cfg", ".conf", ".env", ".toml"}
DOC_EXTS    = {".md", ".rst", ".txt", ".adoc"}
TABLE_EXTS  = {".csv", ".tsv"}
EXCEL_EXTS  = {".xlsx"}
JSON_EXTS   = {".json", ".jsonl"}
IMAGE_EXTS  = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".ico", ".svg"}

# --------------------------- Helpers -----------------------------------------
def iso_utc(ts: float) -> str:
    return datetime.utcfromtimestamp(ts).isoformat(timespec="seconds") + "Z"

def file_sha256(path: str, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if limit_bytes is None:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            remaining = limit_bytes
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
    return h.hexdigest()

def classify_file(ext: str) -> str:
    e = ext.lower()
    if e in SCRIPT_EXTS: return "script"
    if e in TABLE_EXTS or e in EXCEL_EXTS: return "table"
    if e in JSON_EXTS: return "json"
    if e in IMAGE_EXTS: return "image"
    if e in CONFIG_EXTS: return "config"
    if e in DOC_EXTS: return "doc"
    return "other"

def summarize_csv_headers(path: str) -> str:
    import csv as _csv
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(4096); f.seek(0)
        try:
            dialect = _csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        except Exception:
            dialect = _csv.excel
        reader = _csv.reader(f, dialect)
        try:
            header = next(reader, [])
        except Exception:
            header = []
    return "Headers: " + ", ".join([str(h) for h in header]) if header else "No header row detected."

def summarize_xlsx_headers(path: str) -> str:
    if not OPENPYXL_AVAILABLE:
        return "openpyxl not installed; cannot summarize .xlsx."
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as e:
        return f"Failed to open xlsx: {e!r}"
    parts = []
    try:
        for ws in wb.worksheets:
            header = []
            for row in ws.iter_rows(min_row=1, max_row=5, values_only=True):
                if row and any(cell is not None for cell in row):
                    header = ["" if c is None else str(c) for c in row]
                    if any(h.strip() for h in header):
                        break
            parts.append(f"Sheet '{ws.title}': " + (", ".join(header) if header else "(no non-empty first rows to infer headers)"))
    finally:
        wb.close()
    return " | ".join(parts) if parts else "No sheets found."

def summarize_json(path: str) -> str:
    try:
        raw = open(path, "rb").read(MAX_JSON_BYTES)
        text = raw.decode("utf-8", errors="replace")
        data = json.loads(text)
    except Exception as e:
        return f"JSON parse failed or too large to parse safely: {e!r}"

    def t(x):
        if x is None: return "null"
        if isinstance(x, bool): return "bool"
        if isinstance(x, (int, float)): return "number"
        if isinstance(x, str): return "string"
        if isinstance(x, list): return f"array(len={len(x)})"
        if isinstance(x, dict): return f"object(keys={len(x)})"
        return type(x).__name__

    if isinstance(data, dict):
        keys = list(data.keys())
        annotated = [f"{k}:{t(data[k])}" for k in keys[:100]]
        extra = "" if len(keys) <= 100 else f" (+{len(keys)-100} more keys)"
        return "Top-level keys: " + ", ".join(annotated) + extra
    if isinstance(data, list):
        desc = f"Top-level is array(len={len(data)})"
        if data and isinstance(data[0], dict):
            first_keys = list(data[0].keys())
            desc += "; first item keys: " + ", ".join(first_keys[:50])
            if len(first_keys) > 50:
                desc += f" (+{len(first_keys)-50} more)"
        return desc
    return f"Top-level is {t(data)}"

def summarize_image(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".svg":
        return "Image (SVG vector)."
    if not PIL_AVAILABLE:
        return "Image (Pillow not installed; dimensions unavailable)."
    try:
        with Image.open(path) as im:
            return f"Image: format={im.format}, size={im.size[0]}x{im.size[1]}"
    except Exception as e:
        return f"Image metadata unavailable: {e!r}"

def read_script_content(path: str, max_bytes: int) -> tuple[str, bool]:
    raw = open(path, "rb").read(max_bytes + 1)
    truncated = len(raw) > max_bytes
    if truncated:
        raw = raw[:max_bytes]
    text = raw.decode("utf-8", errors="replace")
    return text, truncated

# --------------------------- Pruning helpers ---------------------------------
def normalize_set(iterable):
    out = set()
    for s in iterable:
        if not s: continue
        s = s.strip()
        if not s: continue
        out.add(s.lower())
    return out

def should_prune_dir(dirname: str, prune_names: set[str], prune_substr: set[str]) -> bool:
    d = dirname.lower()
    if d in prune_names:
        return True
    return any(sub in d for sub in prune_substr)

# ---------------------- Image collection collapse ----------------------------
def aggregate_image_folder(dirpath: str, filenames: list[str], include_exts: set[str], sample_n: int):
    """
    Build an aggregate summary for image-heavy folders.
    Returns (details_str, total_bytes, formats_counter, dims_stats_dict, samples)
    """
    image_files = [f for f in filenames if os.path.splitext(f)[1].lower() in IMAGE_EXTS and
                   (not include_exts or os.path.splitext(f)[1].lower() in include_exts)]
    sizes = []
    total_bytes = 0
    formats = Counter()
    samples = sorted(image_files)[:sample_n]

    for fname in image_files:
        fpath = os.path.join(dirpath, fname)
        try:
            total_bytes += os.path.getsize(fpath)
        except Exception:
            pass
        ext = os.path.splitext(fname)[1].lower()
        if ext == ".svg":
            formats["SVG"] += 1
            # no raster size
        elif PIL_AVAILABLE:
            try:
                with Image.open(fpath) as im:
                    formats[im.format or ext.upper()] += 1
                    sizes.append((im.size[0], im.size[1]))
            except Exception:
                formats[ext.upper().lstrip(".")] += 1
        else:
            formats[ext.upper().lstrip(".")] += 1

    dims_info = {}
    if sizes:
        widths = [w for w, h in sizes]
        heights = [h for w, h in sizes]
        dims_info = {
            "min": f"{min(widths)}x{min(heights)}",
            "median": f"{int(statistics.median(widths))}x{int(statistics.median(heights))}",
            "max": f"{max(widths)}x{max(heights)}",
        }

    fmt_str = ", ".join(f"{k}:{v}" for k, v in sorted(formats.items()))
    parts = [
        f"Image collection: count={len(image_files):,}",
        f"total_bytes={total_bytes:,}",
    ]
    if fmt_str:
        parts.append(f"formats={{ {fmt_str} }}")
    if dims_info:
        parts.append(f"size(min/med/max)={dims_info['min']}/{dims_info['median']}/{dims_info['max']}")
    if samples:
        parts.append(f"samples=[{', '.join(samples)}]")
    details = " | ".join(parts)
    return details, total_bytes, samples

# --------------------------- Main walk ---------------------------------------
def process_folder(root: str, out_csv: str, max_script_bytes: int,
                   include_exts: set[str], prune_names: set[str], prune_substr: set[str],
                   collapse_images: bool, collapse_threshold: int, collapse_force: set[str],
                   sample_images: int):
    root = os.path.abspath(root)
    rows = []

    for dirpath, dirnames, filenames in os.walk(root):
        # prune dirs in-place so walk doesn't descend
        dirnames[:] = [d for d in dirnames if not should_prune_dir(d, prune_names, prune_substr)]

        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".": rel_dir = ""
        rel_dir_unix = rel_dir.replace("\\", "/")
        try:
            mtime = os.stat(dirpath).st_mtime
        except Exception:
            mtime = 0.0

        # Folder row
        rows.append({
            "item_type": "folder",
            "classification": "folder",
            "relative_path": rel_dir_unix,
            "name": os.path.basename(dirpath) if rel_dir else os.path.basename(root),
            "ext": "",
            "size_bytes": 0,
            "modified_utc": iso_utc(mtime) if mtime else "",
            "sha256": "",
            "details": f"Contains {len(dirnames)} subfolder(s), {len(filenames)} file(s).",
            "content": ""
        })

        # Collapse decision for image-heavy folders
        image_files_here = [f for f in filenames if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        force_collapse = rel_dir_unix.lower() in collapse_force
        do_collapse = collapse_images and (force_collapse or len(image_files_here) >= collapse_threshold)

        if do_collapse and image_files_here:
            details, total_bytes, samples = aggregate_image_folder(dirpath, filenames, include_exts, sample_images)
            rows.append({
                "item_type": "file",                     # keep schema stable
                "classification": "image_collection",    # <- key new value
                "relative_path": rel_dir_unix + "/*",
                "name": os.path.basename(dirpath) + " (images)",
                "ext": "",
                "size_bytes": total_bytes,
                "modified_utc": iso_utc(mtime) if mtime else "",
                "sha256": "",
                "details": details,
                "content": ""
            })
            # For collapsed folders: skip adding individual image rows
            # Still include non-image files normally (e.g., decode_table.csv/json)
            non_image_files = [f for f in filenames if os.path.splitext(f)[1].lower() not in IMAGE_EXTS]
            filenames = non_image_files  # continue below with non-images only

        # Normal per-file processing (non-images here if collapsed)
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if include_exts and ext not in include_exts:
                continue

            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, root).replace("\\", "/")
            classification = classify_file(ext)
            try:
                st = os.stat(fpath); size = st.st_size; mtime = st.st_mtime
            except Exception:
                size, mtime = 0, 0.0

            sha = ""
            details = ""
            content = ""

            if os.path.islink(fpath):
                details = "Symlink (target not inlined)."

            if classification == "script":
                try: sha = file_sha256(fpath)
                except Exception: sha = ""
                try:
                    content, truncated = read_script_content(fpath, max_script_bytes)
                    if truncated:
                        details = f"Script content truncated to {max_script_bytes} bytes."
                except Exception as e:
                    details = f"Failed to read script: {e!r}"

            elif classification == "table":
                try: sha = file_sha256(fpath)
                except Exception: sha = ""
                if ext in TABLE_EXTS:
                    try: details = summarize_csv_headers(fpath)
                    except Exception as e: details = f"Failed to summarize CSV/TSV: {e!r}"
                elif ext in EXCEL_EXTS:
                    details = summarize_xlsx_headers(fpath)

            elif classification == "json":
                try: sha = file_sha256(fpath)
                except Exception: sha = ""
                details = summarize_json(fpath)

            elif classification == "image":
                try: sha = file_sha256(fpath)
                except Exception: sha = ""
                details = summarize_image(fpath)

            else:
                if size <= 5 * 1024 * 1024:
                    try: sha = file_sha256(fpath)
                    except Exception: sha = ""
                if classification == "config":
                    details = "Configuration file."
                elif classification == "doc":
                    details = "Documentation/notes."

            rows.append({
                "item_type": "file",
                "classification": classification,
                "relative_path": rel,
                "name": fname,
                "ext": ext,
                "size_bytes": size,
                "modified_utc": iso_utc(mtime) if mtime else "",
                "sha256": sha,
                "details": details,
                "content": content
            })

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

def main():
    p = argparse.ArgumentParser(description="Produce an application summary CSV for an LLM (project files only).")
    p.add_argument("--root", required=True, help="Root folder to scan.")
    p.add_argument("--out", default="application_summary.csv", help="Output CSV path.")
    p.add_argument("--max-script-bytes", type=int, default=MAX_SCRIPT_BYTES_DEFAULT,
                   help=f"Max bytes of script content per file (default {MAX_SCRIPT_BYTES_DEFAULT}).")
    p.add_argument("--include-ext", default=",".join(sorted(DEFAULT_INCLUDE_EXTS)),
                   help="Comma-separated list of file extensions to include (e.g. .py,.csv,.json).")
    p.add_argument("--prune", default=";".join(sorted(DEFAULT_PRUNE_DIRS)),
                   help="Semicolon-separated list of directory names to prune.")
    p.add_argument("--prune-substr", default=";".join(sorted(DEFAULT_PRUNE_SUBSTR)),
                   help="Semicolon-separated list of substrings; directories containing any will be pruned.")

    # NEW: image-collapse controls
    p.add_argument("--collapse-images", action="store_true", default=True,
                   help="Collapse image-heavy folders into a single summary row (default: on).")
    p.add_argument("--collapse-threshold", type=int, default=50,
                   help="If a folder has >= this many images, collapse it (default 50).")
    p.add_argument("--collapse-force", default="out/symbols;out/digits_0_625_row",
                   help="Semicolon-separated relative paths to ALWAYS collapse (case-insensitive).")
    p.add_argument("--sample-images", type=int, default=5,
                   help="Number of sample filenames to list for collapsed folders (default 5).")

    args = p.parse_args()

    include_exts = normalize_set(ext.strip() if ext.startswith(".") else f".{ext.strip()}"
                                 for ext in args.include_ext.split(","))
    prune_names = normalize_set(args.prune.split(";"))
    prune_substr = normalize_set(args.prune_substr.split(";"))
    collapse_force = normalize_set(args.collapse_force.split(";"))

    effective_max_script_bytes = max(1, int(args.max_script_bytes))
    if not os.path.isdir(args.root):
        print(f"ERROR: Root not found or not a directory: {args.root}", file=sys.stderr)
        sys.exit(1)

    process_folder(
        root=args.root,
        out_csv=args.out,
        max_script_bytes=effective_max_script_bytes,
        include_exts=include_exts,
        prune_names=prune_names,
        prune_substr=prune_substr,
        collapse_images=bool(args.collapse_images),
        collapse_threshold=max(1, int(args.collapse_threshold)),
        collapse_force=collapse_force,
        sample_images=max(0, int(args.sample_images)),
    )
    print(f"✅ Application summary written to: {args.out}")

if __name__ == "__main__":
    main()
