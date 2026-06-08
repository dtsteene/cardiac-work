"""Unit + equivalence tests for analysis_core — pure NumPy, login-node safe.

    python3 tests/test_analysis_core.py

Two layers:
  1. Hand-computed expected values pin the math.
  2. Equivalence checks compare against the canonical thesis harness
     ``sweep_analysis.py`` so the core is provably identical to the code
     that produced the published numbers.
"""

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import analysis_core as ac
import sweep_analysis as ah5  # canonical reference (constants only at import)


# ── helpers ──────────────────────────────────────────────────────────────────

def _npz(**arrays):
    """Write arrays to a temp .npz and load it back as an NpzFile."""
    path = Path(tempfile.mkdtemp()) / "fixture.npz"
    np.savez(path, **arrays)
    return np.load(path, allow_pickle=True)


# ── hand-computed pins ───────────────────────────────────────────────────────

def test_region_density_hand():
    values = np.array([-1.0, -2.0, -3.0, -4.0])      # stored negative
    cell_volumes = np.array([1.0, 1.0, 1.0, 1.0])
    mask = np.array([True, True, False, False])
    # -(-1 + -2)/2 * 1e-3 = 1.5e-3
    assert math.isclose(ac.region_density(values, mask, cell_volumes), 1.5e-3)


def test_pressure_candidates_hand():
    plv = np.array([10.0, 10.0])
    prv = np.array([4.0, 6.0])
    tau = np.array([0.2, 0.8])
    out = ac.pressure_candidates(plv, prv, tau)
    assert np.allclose(out["Trans"], [6.0, 4.0])
    assert np.allclose(out["Mean"], [7.0, 8.0])
    assert np.allclose(out["NearestSide"], [10.0, 6.0])           # tau<0.5 → plv
    assert np.allclose(out["TauWeighted"], [0.8 * 10 + 0.2 * 4, 0.2 * 10 + 0.8 * 6])


def test_pearson_guards():
    assert math.isnan(ac.pearson_r([1.0, 2.0], [1.0, 2.0]))        # n < 3
    assert math.isnan(ac.pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))  # constant x
    assert math.isclose(ac.pearson_r([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)


def test_correlation_stats_perfect_line():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [3.0, 5.0, 7.0, 9.0]                                       # y = 2x + 1
    s = ac.correlation_stats(x, y)
    assert math.isclose(s["r"], 1.0)
    assert math.isclose(s["r2"], 1.0)
    assert math.isclose(s["slope"], 2.0, abs_tol=1e-9)
    assert math.isclose(s["intercept"], 1.0, abs_tol=1e-9)


def test_ratio_preservation_hand():
    vals = np.array([2.0, 0.5])
    ref = np.array([1.0, 1.0])
    out = ac.ratio_preservation(vals, ref)
    # |log 2|, |log 0.5| are equal → mean = log 2
    assert math.isclose(out["mean_abs_log_error"], math.log(2.0))
    assert math.isclose(out["max_abs_log_error"], math.log(2.0))
    assert math.isclose(out["mean_abs_raw_error"], 0.75)          # (1.0 + 0.5)/2
    assert math.isclose(out["max_abs_raw_error"], 1.0)


# ── equivalence vs canonical sweep_analysis ──────────────────────────────────

def test_equiv_pearson_vs_canonical():
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.normal(size=12)
        y = 0.7 * x + rng.normal(size=12) * 0.3
        a = ac.pearson_r(x, y)
        b = ah5.corr(list(x), list(y))
        assert math.isclose(a, b, rel_tol=1e-12), (a, b)


def test_equiv_density_vs_canonical():
    pc = _npz(
        cell_volumes=np.array([1.0, 2.0, 3.0, 4.0]),
        region_tags=np.array([1, 1, 2, 3]),
        w_total=np.array([-5.0, -6.0, -7.0, -8.0]),
    )
    mask = pc["region_tags"] == 1
    assert math.isclose(
        ac.region_density(pc["w_total"], mask, pc["cell_volumes"]),
        ah5.density(pc, mask, "w_total"),
        rel_tol=1e-12,
    )


def test_equiv_pressure_candidates_vs_canonical():
    pc = _npz(
        proxy_PLV_ll=np.array([10.0, 12.0, 9.0]),
        proxy_PRV_ll=np.array([4.0, 6.0, 8.0]),
        lv_rv_scalar=np.array([0.9, 0.5, 0.1]),     # tau = 1 - this
    )
    canon = ah5.candidate_arrays(pc, "ll")
    mine = ac.pressure_candidates(
        pc["proxy_PLV_ll"], pc["proxy_PRV_ll"], ac.tau_from_per_cell(pc)
    )
    for key in ac.PRESSURE_CHOICES:
        assert np.allclose(mine[key], canon[key]), key


def test_equiv_region_masks_vs_canonical():
    pc = _npz(
        region_tags=np.array([1, 2, 3, 1, 2]),
        is_geometric_septum=np.array([0, 0, 1, 0, 1]),
    )
    mine = ac.region_masks(pc["region_tags"], pc["is_geometric_septum"])
    canon = ah5.masks(pc)
    for key in ("LV", "RV", "Septum"):
        assert np.array_equal(mine[key], canon[key]), key


def test_equiv_log_error_vs_canonical_formula():
    # Mirrors the inline computation in ah5.ratio_preservation_rows.
    vals = np.array([1.2, 0.8, 1.5, 0.6])
    tensor = np.array([1.0, 1.0, 1.3, 0.7])
    canon = float(np.mean(np.abs(np.log(np.abs(vals / tensor)))))
    assert math.isclose(ac.log_mae(vals, tensor), canon, rel_tol=1e-12)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
