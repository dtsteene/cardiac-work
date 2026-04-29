#!/usr/bin/env python3
"""Diagnose why the septal proxy ranking changed.

Core idea:
    Trans proxy = PLV proxy - PRV proxy.

So transmural can only improve on PLV if the PRV term is a nuisance term for
the target. This script asks whether PRV x eps_ll is an opposite severity
signal in the old spectrum and a weak positive signal in the corrected spectrum.
It also normalizes proxies by peak pressure to isolate the strain-loop part.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


ROOT = Path("/home/dtsteene/D1/cardiac-work")
HANDOVER = ROOT / "results" / "handover"
KPA = 1e-3


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(pearsonr(x, y)[0])


def load_handover(name: str) -> list[dict]:
    base = HANDOVER / name
    with (base / "hemodynamic_summary.csv").open(newline="") as f:
        summary = list(csv.DictReader(f))

    rows = []
    for s in summary:
        cid = s["case_id"]
        pc = np.load(base / "data" / "per_cell_data" / f"{cid}_per_cell_data.npz", allow_pickle=True)
        mask = pc["is_geometric_septum"].astype(bool)
        vol = pc["cell_volumes"][mask].sum()

        def dens(key):
            return float(-pc[key][mask].sum() / vol * KPA)

        plv = dens("proxy_PLV_ll")
        prv = dens("proxy_PRV_ll")
        lv = float(s["LV_ESP_mmHg"])
        rv = float(s["RV_ESP_mmHg"])
        trans = lv - rv
        rows.append({
            "case": s["archival_key"],
            "LV": lv,
            "RV": rv,
            "Trans": trans,
            "W": dens("w_total"),
            "Wff": dens("w_ff"),
            "Wnn": dens("w_nn"),
            "PLV": plv,
            "PRV": prv,
            "PTrans": plv - prv,
            # Pressure-normalized loop factors. Units are arbitrary here; they
            # are just the same proxy with the pressure amplitude divided out.
            "Lfac": plv / lv if lv else np.nan,
            "Rfac": prv / rv if rv else np.nan,
            "Tfac": (plv - prv) / trans if abs(trans) > 1e-9 else np.nan,
        })
    return rows


def load_n16() -> list[dict]:
    cases = [
        ("sPAP22", 1047450), ("sPAP25", 1048194), ("sPAP30", 1047451), ("sPAP35", 1048195),
        ("sPAP45", 1047452), ("sPAP50", 1048196), ("sPAP55", 1047453), ("sPAP60", 1048197),
        ("sPAP65", 1047454), ("sPAP70", 1048198), ("sPAP75", 1047455), ("sPAP80", 1048199),
        ("sPAP85", 1047456), ("sPAP87", 1048200), ("sPAP92", 1048201), ("sPAP95", 1047457),
    ]
    roots = [ROOT / "results" / "sims" / "2026-04-23", ROOT / "results" / "sims" / "2026-04-24"]
    rows = []
    for label, jid in cases:
        run = next(root / f"UKB_6beats_run_{jid}" for root in roots if (root / f"UKB_6beats_run_{jid}").exists())
        pc = np.load(run / "per_cell_data.npz", allow_pickle=True)
        sp = np.load(run / "solver" / "solver_cavity_pressure_mmHg.npy")
        beat = sp.shape[0] // 6
        last = sp[5 * beat:]
        lv = float(last[:, 0].max())
        rv = float(last[:, 1].max())
        trans = lv - rv
        mask = pc["is_geometric_septum"].astype(bool)
        vol = pc["cell_volumes"][mask].sum()

        def dens(key):
            return float(-pc[key][mask].sum() / vol * KPA)

        plv = dens("proxy_PLV_ll")
        prv = dens("proxy_PRV_ll")
        rows.append({
            "case": label,
            "LV": lv,
            "RV": rv,
            "Trans": trans,
            "W": dens("w_total"),
            "Wff": dens("w_ff"),
            "Wnn": dens("w_nn"),
            "PLV": plv,
            "PRV": prv,
            "PTrans": plv - prv,
            "Lfac": plv / lv if lv else np.nan,
            "Rfac": prv / rv if rv else np.nan,
            "Tfac": (plv - prv) / trans if abs(trans) > 1e-9 else np.nan,
        })
    return rows


def arr(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def summarize(name: str, rows: list[dict]) -> None:
    print("\n" + "=" * 92)
    print(name)
    print("=" * 92)
    print("Correlations with septal tensor work W")
    for key, label in [
        ("LV", "LV peak pressure"),
        ("RV", "RV peak pressure"),
        ("Trans", "LV-RV peak pressure"),
        ("Lfac", "LV-normalized strain-loop factor"),
        ("Rfac", "RV-normalized strain-loop factor"),
        ("PLV", "PLV x eps_ll"),
        ("PRV", "PRV x eps_ll"),
        ("PTrans", "(PLV-PRV) x eps_ll"),
    ]:
        print(f"  {label:<34} r={corr(arr(rows, key), arr(rows, 'W')):+.3f}")

    print("\nCorrelations with RV pressure / disease axis")
    for key, label in [
        ("W", "septal W"),
        ("Lfac", "LV-normalized strain-loop factor"),
        ("Rfac", "RV-normalized strain-loop factor"),
        ("PLV", "PLV x eps_ll"),
        ("PRV", "PRV x eps_ll"),
        ("PTrans", "(PLV-PRV) x eps_ll"),
    ]:
        print(f"  {label:<34} r={corr(arr(rows, key), arr(rows, 'RV')):+.3f}")

    print("\nSignal algebra")
    print(f"  corr(PLV proxy, PRV proxy) = {corr(arr(rows, 'PLV'), arr(rows, 'PRV')):+.3f}")
    print(f"  corr(PRV proxy, W)         = {corr(arr(rows, 'PRV'), arr(rows, 'W')):+.3f}")
    print(f"  corr(PLV proxy, W)         = {corr(arr(rows, 'PLV'), arr(rows, 'W')):+.3f}")
    print(f"  corr(PLV-PRV proxy, W)     = {corr(arr(rows, 'PTrans'), arr(rows, 'W')):+.3f}")

    # Exact product identity: PRV proxy = RV peak pressure * Rfac.
    # In log space, log(PRV) = log(RV) + log(Rfac). The covariance with W
    # therefore separates into a pressure-amplitude part and a strain-loop part.
    log_rv = np.log(arr(rows, "RV"))
    log_rfac = np.log(arr(rows, "Rfac"))
    log_prv = np.log(arr(rows, "PRV"))
    w = arr(rows, "W")

    def cov(x, y):
        return float(np.mean((x - x.mean()) * (y - y.mean())))

    cov_rv = cov(log_rv, w)
    cov_rfac = cov(log_rfac, w)
    cov_prv = cov(log_prv, w)
    denom = abs(cov_rv) + abs(cov_rfac)
    rv_share = cov_rv / denom if denom else np.nan
    rfac_share = cov_rfac / denom if denom else np.nan
    print("\nProduct decomposition of the RV proxy")
    print("  PRV proxy = RV pressure amplitude x RV-normalized loop factor")
    print(f"  cov(log RV pressure, W)       = {cov_rv:+.4f}")
    print(f"  cov(log loop factor, W)       = {cov_rfac:+.4f}")
    print(f"  cov(log PRV proxy, W)         = {cov_prv:+.4f}")
    print(f"  relative signed shares        = RV {rv_share:+.2f}, loop {rfac_share:+.2f}")

    print("\nPer-case: pressure-normalized factor shows the strain-loop part")
    print(f"  {'case':<16} {'RV':>6} {'W':>6} {'PLV':>7} {'PRV':>7} {'Trans':>7} {'Lfac':>8} {'Rfac':>8}")
    for row in rows:
        print(
            f"  {row['case']:<16} {row['RV']:6.1f} {row['W']:6.2f} "
            f"{row['PLV']:7.3f} {row['PRV']:7.3f} {row['PTrans']:7.3f} "
            f"{row['Lfac']:8.5f} {row['Rfac']:8.5f}"
        )


def main():
    summarize("old handover n=7", load_handover("handover_old"))
    summarize("corrected EXP handover n=8", load_handover("handover_exp"))
    summarize("corrected EXP n=16", load_n16())


if __name__ == "__main__":
    main()
