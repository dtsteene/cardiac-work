"""Systematic analysis core — the one home for the per-region statistics.

Pure NumPy/SciPy, **no FEniCSx**: this module reads the precomputed arrays that
``compute_per_cell.py`` / ``postprocess_metrics.py`` write (``per_cell_data.npz``,
``metrics_downsample_*.npy``) and turns them into the quantities every analysis
script needs:

    * region work density (ground-truth S:dE), per LV / RV / Septum
    * pressure-strain proxy work with a **swappable pressure** (P_LV, P_RV,
      transmural P_LV-P_RV, mean, nearest-side, tau-weighted)
    * ratios between regions and proxy-vs-truth ratio preservation (log-MAE)
    * correlations (Pearson r, R^2, slope) between proxy and truth

These primitives were previously copy-pasted across ~6 one-off scripts. The
canonical definitions here are lifted verbatim from the thesis sweep harness
``sweep_analysis.py`` so results are numerically identical.

Because there is no FEniCSx dependency, this module imports instantly and is
fully unit-testable on the login node (see ``tests/test_analysis_core.py``).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import pearsonr

# Region tag ids used by the LDRB tagging algorithm.
REGIONS = {"LV": 1, "RV": 2, "Septum": 3}

# Work-decomposition directions and the two strains that drive the proxies.
# ``ll`` (longitudinal / GLS) is the primary clinical strain; ``ff`` (fibre) is
# secondary.
COMPONENTS = ("ff", "ss", "nn", "cross")
STRAINS = ("ll", "ff")

# The septum is loaded from both sides, so its proxy pressure is ambiguous.
# These are the candidate choices we evaluate; ``Trans`` (transmural) is the
# thesis's preferred septal proxy in PAH.
PRESSURE_CHOICES = ("PLV", "PRV", "Trans", "Mean", "NearestSide", "TauWeighted")

# Per-cell work densities are stored in Pa·(unit strain); ×1e-3 → kPa.
KPA = 1e-3


def region_masks(region_tags, is_geometric_septum) -> dict[str, np.ndarray]:
    """Boolean masks for LV free wall, RV free wall, and the geometric septum.

    LV/RV come from the LDRB tag ids (1/2); the septum uses the geometric
    classification (``is_geometric_septum``) rather than tag 3, matching the
    thesis convention.
    """
    tags = np.asarray(region_tags)
    return {
        "LV": tags == REGIONS["LV"],
        "RV": tags == REGIONS["RV"],
        "Septum": np.asarray(is_geometric_septum).astype(bool),
    }


def region_density(values, mask, cell_volumes, kpa: float = KPA) -> float:
    """Volume-averaged work density over a region, in kPa.

    ``-Σ values[mask] / Σ cell_volumes[mask] × kpa``. The sign flip follows the
    saved convention (work done *on* the cavity is stored negative).
    """
    values = np.asarray(values)
    cell_volumes = np.asarray(cell_volumes)
    volume = float(cell_volumes[mask].sum())
    return float(-values[mask].sum() / volume * kpa)


def pressure_candidates(plv, prv, tau) -> dict[str, np.ndarray]:
    """The six candidate septal pressure-strain integrands, given the PLV and
    PRV proxy arrays and the transmural coordinate ``tau`` (0 on the LV side,
    1 on the RV side).
    """
    plv = np.asarray(plv, dtype=float)
    prv = np.asarray(prv, dtype=float)
    tau = np.asarray(tau, dtype=float)
    return {
        "PLV": plv,
        "PRV": prv,
        "Trans": plv - prv,
        "Mean": 0.5 * (plv + prv),
        "NearestSide": np.where(tau < 0.5, plv, prv),
        "TauWeighted": (1.0 - tau) * plv + tau * prv,
    }


def tau_from_per_cell(pc) -> np.ndarray:
    """Transmural coordinate (tau=0 LV side, tau=1 RV side) from a per-cell npz.

    The saved Laplace scalar ``lv_rv_scalar`` has the opposite orientation, so
    we flip it; falls back to a stored ``tau`` array if present.
    """
    files = getattr(pc, "files", pc)
    if "lv_rv_scalar" in files:
        return 1.0 - np.asarray(pc["lv_rv_scalar"], dtype=float)
    return np.asarray(pc["tau"], dtype=float)


def pearson_r(x, y) -> float:
    """Pearson correlation, NaN-guarded for degenerate inputs.

    Returns NaN when there are fewer than 3 samples or either series is
    constant — matching the thesis harness's ``corr``.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) < 3 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(pearsonr(x_arr, y_arr)[0])


def correlation_stats(x, y) -> dict[str, float]:
    """Correlation summary between a proxy series ``x`` and truth series ``y``.

    Returns ``r``, ``r2``, and the least-squares ``slope``/``intercept`` of
    ``y`` on ``x``. ``r``/``r2`` match the thesis harness exactly.
    """
    r = pearson_r(x, y)
    r2 = r * r if not math.isnan(r) else float("nan")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) < 2 or np.std(x_arr) == 0:
        slope = float("nan")
        intercept = float("nan")
    else:
        slope, intercept = (float(v) for v in np.polyfit(x_arr, y_arr, 1))
    return {"n": len(x_arr), "r": r, "r2": r2, "slope": slope, "intercept": intercept}


def ratio_preservation(values, reference) -> dict[str, float]:
    """How well a proxy ratio series tracks the truth ratio series.

    Reports both raw absolute error ``|v - ref|`` and the scale-free log error
    ``|log|v/ref||`` (the thesis's primary ratio-preservation metric), each as
    mean and max over the cases.
    """
    vals = np.asarray(values, dtype=float)
    ref = np.asarray(reference, dtype=float)
    log_err = np.abs(np.log(np.abs(vals / ref)))
    raw_err = np.abs(vals - ref)
    return {
        "n": len(vals),
        "mean_abs_raw_error": float(np.mean(raw_err)),
        "max_abs_raw_error": float(np.max(raw_err)),
        "mean_abs_log_error": float(np.mean(log_err)),
        "max_abs_log_error": float(np.max(log_err)),
    }


def log_mae(values, reference) -> float:
    """Mean absolute log ratio error — the headline ratio-preservation number."""
    return ratio_preservation(values, reference)["mean_abs_log_error"]


__all__ = [
    "REGIONS",
    "COMPONENTS",
    "STRAINS",
    "PRESSURE_CHOICES",
    "KPA",
    "region_masks",
    "region_density",
    "pressure_candidates",
    "tau_from_per_cell",
    "pearson_r",
    "correlation_stats",
    "ratio_preservation",
    "log_mae",
]
