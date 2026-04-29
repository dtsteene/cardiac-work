#!/usr/bin/env python3
"""Compare old and corrected spectra to explain the septal proxy flip.

The question is why the old handover data made the transmural septal proxy
look best, while the corrected preserved-SBP spectrum makes P_LV x epsilon_ll
look best for total septal tensor work.

This script keeps the comparison deliberately simple:
  * use the same geometric septum mask in each per_cell_data file;
  * compute intensive work densities from per-cell integrated quantities;
  * compare pressure traces from the handover hemodynamic summaries;
  * report correlations for total and directional septal work.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


ROOT = Path("/home/dtsteene/D1/cardiac-work")
HANDOVER = ROOT / "results" / "handover"
KPA = 1e-3


def r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(pearsonr(x, y)[0])


def read_handover(name: str, label: str) -> list[dict]:
    base = HANDOVER / name
    rows = []
    with (base / "hemodynamic_summary.csv").open(newline="") as f:
        summary = list(csv.DictReader(f))

    for s in summary:
        cid = s["case_id"]
        pc_path = base / "data" / "per_cell_data" / f"{cid}_per_cell_data.npz"
        pc = np.load(pc_path, allow_pickle=True)
        mask = pc["is_geometric_septum"].astype(bool)
        vol = float(pc["cell_volumes"][mask].sum())

        def dens(key: str) -> float:
            return float(-pc[key][mask].sum() / vol * KPA)

        plv = dens("proxy_PLV_ll")
        prv = dens("proxy_PRV_ll")
        rows.append(
            {
                "dataset": label,
                "case": s["archival_key"],
                "RV_ESP": float(s["RV_ESP_mmHg"]),
                "LV_ESP": float(s["LV_ESP_mmHg"]),
                "Trans_ESP": float(s["LV_ESP_mmHg"]) - float(s["RV_ESP_mmHg"]),
                "Mean_ESP": 0.5 * (float(s["LV_ESP_mmHg"]) + float(s["RV_ESP_mmHg"])),
                "CO": float(s["CO_Lmin"]),
                "LV_SV": float(s["LV_SV_mL"]),
                "RV_SV": float(s["RV_SV_mL"]),
                "W_total": dens("w_total"),
                "W_ff": dens("w_ff"),
                "W_ss": dens("w_ss"),
                "W_nn": dens("w_nn"),
                "W_cross": dens("w_cross"),
                "proxy_PLV": plv,
                "proxy_PRV": prv,
                "proxy_Trans": plv - prv,
                "proxy_Mean": 0.5 * (plv + prv),
            }
        )
    return rows


def read_n16() -> list[dict]:
    cases = [
        ("sPAP22", 1047450),
        ("sPAP25", 1048194),
        ("sPAP30", 1047451),
        ("sPAP35", 1048195),
        ("sPAP45", 1047452),
        ("sPAP50", 1048196),
        ("sPAP55", 1047453),
        ("sPAP60", 1048197),
        ("sPAP65", 1047454),
        ("sPAP70", 1048198),
        ("sPAP75", 1047455),
        ("sPAP80", 1048199),
        ("sPAP85", 1047456),
        ("sPAP87", 1048200),
        ("sPAP92", 1048201),
        ("sPAP95", 1047457),
    ]
    roots = [
        ROOT / "results" / "sims" / "2026-04-23",
        ROOT / "results" / "sims" / "2026-04-24",
    ]

    out = []
    for label, jid in cases:
        run = None
        for root in roots:
            p = root / f"UKB_6beats_run_{jid}"
            if p.exists():
                run = p
                break
        if run is None:
            raise FileNotFoundError(jid)

        pc = np.load(run / "per_cell_data.npz", allow_pickle=True)
        sp = np.load(run / "solver" / "solver_cavity_pressure_mmHg.npy")
        beat = sp.shape[0] // 6
        last = sp[5 * beat :]
        mask = pc["is_geometric_septum"].astype(bool)
        vol = float(pc["cell_volumes"][mask].sum())

        def dens(key: str) -> float:
            return float(-pc[key][mask].sum() / vol * KPA)

        plv = dens("proxy_PLV_ll")
        prv = dens("proxy_PRV_ll")
        out.append(
            {
                "dataset": "corrected n=16",
                "case": label,
                "RV_ESP": float(last[:, 1].max()),
                "LV_ESP": float(last[:, 0].max()),
                "Trans_ESP": float(last[:, 0].max() - last[:, 1].max()),
                "Mean_ESP": 0.5 * float(last[:, 0].max() + last[:, 1].max()),
                "CO": np.nan,
                "LV_SV": np.nan,
                "RV_SV": np.nan,
                "W_total": dens("w_total"),
                "W_ff": dens("w_ff"),
                "W_ss": dens("w_ss"),
                "W_nn": dens("w_nn"),
                "W_cross": dens("w_cross"),
                "proxy_PLV": plv,
                "proxy_PRV": prv,
                "proxy_Trans": plv - prv,
                "proxy_Mean": 0.5 * (plv + prv),
            }
        )
    return out


def arr(rows, key):
    return np.array([row[key] for row in rows], dtype=float)


def best_alpha(rows: list[dict], target_key: str) -> tuple[float, float]:
    """Return alpha maximizing corr(P_LV_proxy - alpha P_RV_proxy, target)."""
    plv = arr(rows, "proxy_PLV")
    prv = arr(rows, "proxy_PRV")
    y = arr(rows, target_key)
    alphas = np.linspace(-2.0, 2.0, 1601)
    rs = np.array([r(plv - a * prv, y) for a in alphas])
    i = int(np.nanargmax(rs))
    return float(alphas[i]), float(rs[i])


def describe_dataset(name: str, rows: list[dict]) -> None:
    print("\n" + "=" * 88)
    print(name)
    print("=" * 88)

    lv = arr(rows, "LV_ESP")
    rv = arr(rows, "RV_ESP")
    trans = arr(rows, "Trans_ESP")
    mean = arr(rows, "Mean_ESP")
    w = arr(rows, "W_total")

    print("Pressure ranges, end-systolic (mmHg)")
    print(f"  LV_ESP:    {lv.min():6.1f} -> {lv.max():6.1f}   span {lv.ptp():5.1f}")
    print(f"  RV_ESP:    {rv.min():6.1f} -> {rv.max():6.1f}   span {rv.ptp():5.1f}")
    print(f"  Trans_ESP: {trans.min():6.1f} -> {trans.max():6.1f}   span {trans.ptp():5.1f}")
    print(f"  mean_ESP:  {mean.min():6.1f} -> {mean.max():6.1f}   span {mean.ptp():5.1f}")

    print("\nPressure cross-correlations")
    print(f"  corr(LV_ESP, RV_ESP)       = {r(lv, rv):+6.3f}")
    print(f"  corr(Trans_ESP, RV_ESP)    = {r(trans, rv):+6.3f}")
    print(f"  corr(mean_ESP, RV_ESP)     = {r(mean, rv):+6.3f}")
    print(f"  corr(W_total, LV_ESP)      = {r(w, lv):+6.3f}")
    print(f"  corr(W_total, RV_ESP)      = {r(w, rv):+6.3f}")
    print(f"  corr(W_total, Trans_ESP)   = {r(w, trans):+6.3f}")

    print("\nSeptum proxy correlations vs total tensor work")
    for key, label in [
        ("proxy_PLV", "P_LV x eps_ll"),
        ("proxy_PRV", "P_RV x eps_ll"),
        ("proxy_Trans", "(P_LV-P_RV) x eps_ll"),
        ("proxy_Mean", "mean(P_LV,P_RV) x eps_ll"),
    ]:
        rr = r(arr(rows, key), w)
        print(f"  {label:<28} r={rr:+6.3f}  r2={rr * rr:5.3f}")

    print("\nRV term diagnostic")
    print(f"  corr(P_LV proxy, P_RV proxy) = {r(arr(rows, 'proxy_PLV'), arr(rows, 'proxy_PRV')):+6.3f}")
    print(f"  corr(P_RV proxy, W_total)    = {r(arr(rows, 'proxy_PRV'), w):+6.3f}")
    for target in ["W_total", "W_ff", "W_nn"]:
        alpha, rr = best_alpha(rows, target)
        print(
            f"  best P_LV - alpha*P_RV for {target:<7}: "
            f"alpha={alpha:+5.2f}, r={rr:+6.3f}"
        )

    print("\nDirectional work: mean absolute fraction of |W_total|")
    denom = np.abs(w)
    for key in ["W_ff", "W_ss", "W_nn", "W_cross"]:
        frac = np.mean(np.abs(arr(rows, key)) / denom)
        print(f"  |{key}| / |W_total| = {frac:6.1%}")

    print("\nWhich proxy tracks each directional component?")
    print(f"  {'component':<9} {'P_LV':>7} {'Trans':>7} {'Mean':>7} {'P_RV':>7}")
    for comp in ["W_total", "W_ff", "W_ss", "W_nn", "W_cross"]:
        y = arr(rows, comp)
        print(
            f"  {comp:<9} "
            f"{r(arr(rows, 'proxy_PLV'), y):+7.3f} "
            f"{r(arr(rows, 'proxy_Trans'), y):+7.3f} "
            f"{r(arr(rows, 'proxy_Mean'), y):+7.3f} "
            f"{r(arr(rows, 'proxy_PRV'), y):+7.3f}"
        )

    print("\nPer-case diagnostic")
    print(f"  {'case':<16} {'LV':>6} {'RV':>6} {'Trans':>7} {'W_tot':>9} {'P_LVpx':>9} {'Transpx':>9}")
    for row in rows:
        print(
            f"  {row['case']:<16} {row['LV_ESP']:6.1f} {row['RV_ESP']:6.1f} "
            f"{row['Trans_ESP']:7.1f} {row['W_total']:9.3f} "
            f"{row['proxy_PLV']:9.3f} {row['proxy_Trans']:9.3f}"
        )


def main() -> None:
    datasets = [
        ("old handover n=7 (LV pressure drop issue)", read_handover("handover_old", "old handover")),
        ("corrected EXP handover n=8", read_handover("handover_exp", "corrected n=8")),
        ("corrected EXP + in-between cases n=16", read_n16()),
    ]

    for name, rows in datasets:
        describe_dataset(name, rows)


if __name__ == "__main__":
    main()
