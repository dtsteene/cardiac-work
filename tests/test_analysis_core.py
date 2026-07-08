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


# ── agreement metrics (single-global-k calibration) ──────────────────────────

def test_concordance_ccc_identity():
    x = [1.0, 2.0, 3.0, 4.0]
    assert math.isclose(ac.concordance_ccc(x, x), 1.0, abs_tol=1e-12)


def test_concordance_ccc_scale_shift():
    # Pearson r = 1 but a 2x scale shift → CCC penalizes it.
    # x=[1,2,3], y=[2,4,6]: cov=4/3, var_x=2/3, var_y=8/3, (mx-my)^2=4
    # CCC = 2*(4/3)/(2/3 + 8/3 + 4) = (8/3)/(22/3) = 4/11
    assert math.isclose(ac.pearson_r([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)
    assert math.isclose(ac.concordance_ccc([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]),
                        4.0 / 11.0, rel_tol=1e-12)


def test_concordance_ccc_guards():
    assert math.isnan(ac.concordance_ccc([1.0], [1.0]))              # n < 2
    assert math.isnan(ac.concordance_ccc([1.0, 1.0], [1.0, 2.0]))    # constant x


def test_proportional_fit_exact():
    # truth = 2 * proxy exactly → k=2, zero residual
    out = ac.proportional_fit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert math.isclose(out["k"], 2.0, rel_tol=1e-12)
    assert math.isclose(out["resid_rmse"], 0.0, abs_tol=1e-12)
    assert math.isclose(out["rel_rmse"], 0.0, abs_tol=1e-12)


def test_proportional_fit_with_residual():
    # proxy=[1,1], truth=[1,3]: k=(1+3)/(1+1)=2; resid=[-1,1]; rmse=1;
    # rel_rmse = 1 / mean(|truth|)=1/2
    out = ac.proportional_fit([1.0, 1.0], [1.0, 3.0])
    assert math.isclose(out["k"], 2.0, rel_tol=1e-12)
    assert math.isclose(out["resid_rmse"], 1.0, rel_tol=1e-12)
    assert math.isclose(out["rel_rmse"], 0.5, rel_tol=1e-12)


def test_agreement_stats_affine():
    proxy = [1.0, 2.0, 3.0, 4.0]
    truth = [3.0, 5.0, 7.0, 9.0]                                     # truth = 2*proxy + 1
    s = ac.agreement_stats(proxy, truth)
    assert math.isclose(s["slope"], 2.0, abs_tol=1e-9)
    assert math.isclose(s["intercept"], 1.0, abs_tol=1e-9)
    assert math.isclose(s["rel_rmse_affine"], 0.0, abs_tol=1e-9)     # perfect affine fit
    assert s["n"] == 4


def test_pooled_proportional_hand():
    # region A: k_A=2 exactly; region B: k_B=1 exactly.
    proxy = {"A": np.array([1.0, 2.0]), "B": np.array([1.0, 2.0])}
    truth = {"A": np.array([2.0, 4.0]), "B": np.array([1.0, 2.0])}
    out = ac.pooled_proportional(proxy, truth)
    assert math.isclose(out["k_global"], 1.5, rel_tol=1e-12)         # 15/10
    assert math.isclose(out["k_by_region"]["A"], 2.0, rel_tol=1e-12)
    assert math.isclose(out["k_by_region"]["B"], 1.0, rel_tol=1e-12)
    assert math.isclose(out["k_spread"], 2.0, rel_tol=1e-12)         # 2/1
    assert math.isclose(out["rel_rmse"], math.sqrt(0.625) / 2.25, rel_tol=1e-12)
    assert math.isclose(out["ccc_pooled"], 0.6428571428571429, rel_tol=1e-9)


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
