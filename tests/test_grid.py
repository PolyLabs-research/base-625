"""Fast regression tests for the base-625 toolkit.

Run from the repo root:  python -m pytest tests
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import base625_grid as b  # noqa: E402
from gen_bit_dataset import sample_png_to_grid25  # noqa: E402

TABLE = ROOT / "out" / "symbols" / "decode_table_symbols.json"


def test_base625_round_trip():
    for n in (0, 1, 624, 625, 626, 390_624, 390_625, 10**9):
        digits = b.int_to_base625(n)
        assert all(0 <= d <= 624 for d in digits)
        assert b.base625_to_int(digits) == n
    with pytest.raises(ValueError):
        b.base625_to_int([625])


@pytest.mark.parametrize("order", ["row", "snake", "random42"])
def test_fill_counts_and_nesting(order):
    prev = b.fill_k_cells(0, order)
    for k in range(0, 626, 25):
        g = b.fill_k_cells(k, order)
        assert g.sum() == k
        assert np.all(prev <= g), "count encoding must be nested (thermometer code)"
        prev = g
    assert b.decode_number_from_count(b.encode_number_as_count(625, order)) == 625


def test_encode_digit_range():
    assert b.encode_digit(624).sum() == 624
    assert b.encode_digit(625).sum() == 625  # reserved full grid, tolerated
    with pytest.raises(ValueError):
        b.encode_digit(626)
    with pytest.raises(ValueError):
        b.encode_digit(-1)


def test_programmatic_glyphs_are_distinct_and_decodable():
    names = list(b.SYMBOLS)
    for i, a in enumerate(names):
        assert b.decode_symbol(b.SYMBOLS[a]) == a
        for c in names[i + 1:]:
            assert not np.array_equal(b.SYMBOLS[a], b.SYMBOLS[c])
        k = int(b.SYMBOLS[a].sum())
        for order in ("row", "snake", "random42"):
            assert not np.array_equal(b.SYMBOLS[a], b.fill_k_cells(k, order)), (a, order)


def test_expression_panels():
    panels = b.encode_expr_to_panels("12 + 3*4 = 24", as_digits=True)
    kinds = [b.decode_symbol(p) or int(p.sum()) for p in panels]
    assert kinds == [12, "PLUS", 3, "TIMES", 4, "EQUALS", 24]
    big = b.encode_expr_to_panels("700", as_digits=True)
    assert [int(p.sum()) for p in big] == [1, 75]


def test_resolve_asset_path_handles_windows_paths(tmp_path):
    item = {"filename": "PLUS.png", "path": "out\\symbols\\PLUS.png"}
    assert b.resolve_asset_path(item, TABLE) == TABLE.parent / "PLUS.png"
    assert b.resolve_asset_path({"filename": "NOPE.png", "path": "out\\symbols\\NOPE.png"}, TABLE) is None
    # Table living somewhere else still resolves via filename next to it.
    other = tmp_path / "t.json"
    other.write_text("{}")
    (tmp_path / "X.png").write_bytes((TABLE.parent / "X.png").read_bytes())
    assert b.resolve_asset_path({"filename": "X.png", "path": "C:\\old\\X.png"}, other) == tmp_path / "X.png"


def test_decode_table_matches_pngs():
    obj = json.load(open(TABLE, encoding="utf-8"))
    assert len(obj["items"]) == 19
    for it in obj["items"]:
        assert "\\" not in it["path"]
        png = b.resolve_asset_path(it, TABLE)
        assert png is not None, it["label"]
        assert int(sample_png_to_grid25(png).sum()) == it["black_cells"], it["label"]


def test_gen_and_validate_bits_dataset(tmp_path):
    out = tmp_path / "bits"
    cmd = [sys.executable, str(ROOT / "gen_bit_dataset.py"), "--out", str(out),
           "--include", "digits", "symbols", "--symbols-json", str(TABLE), "--copies", "2", "--noise-flip", "0.01"]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
    y = np.load(out / "y.npy")
    assert len(set(y.tolist())) == 626 + 19, "all digits and all 19 painted symbols must load"
    assert json.load(open(out / "meta.json"))["noise_flip"] == 0.01
    res = subprocess.run([sys.executable, str(ROOT / "validate_bits_dataset.py"), "--data", str(out),
                          "--symbols", str(TABLE)], check=True, cwd=ROOT, capture_output=True, text=True)
    assert "Digit self-consistency issues: 0" in res.stdout
    assert "Symbol grid parity mismatches: 0" in res.stdout
    assert "Bit collisions (same bits, different labels): 0" in res.stdout
