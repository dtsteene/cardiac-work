#!/usr/bin/env python3
"""Test septal pressure assignment by LV/RV side.

This compares the standard longitudinal septal proxies with two simple
spatially varying pressure choices:

  * closest side: cells closer to the LV use P_LV, cells closer to the RV use P_RV
  * distance weighted: pressure is linearly weighted by distance to LV/RV surfaces

Because the same longitudinal strain is used, these side-pressure choices are
weighted mixtures of the P_LV and P_RV longitudinal proxies.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


ROOT = Path("/home/dtsteene/D1/cardiac-work")
KPA = 1e-3


def corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(pearsonr(x, y)[0])


def load_case(label: str, path: Path) -> dict[str, float]:
    pc = np.load(path, allow_pickle=True)
    mask = pc["is_geometric_septum"].astype(bool)
    volume = float(pc["cell_volumes"][mask].sum())

    def density(values: np.ndarray) -> float:
        return float(-values[mask].sum() / volume * KPA)

    plv = pc["proxy_PLV_ll"]
    prv = pc["proxy_PRV_ll"]
    trans = pc["proxy_Trans_ll"]

    closest = np.where(pc["d_lv"] <= pc["d_rv"], plv, prv)
    lv_weight = pc["d_rv"] / (pc["d_lv"] + pc["d_rv"])
    dist_weighted = lv_weight * plv + (1.0 - lv_weight) * prv

    return {
        "case": label,
        "W_total": density(pc["w_total"]),
        "W_ff": density(pc["w_ff"]),
        "PLV": density(plv),
        "PRV": density(prv),
        "Trans": density(trans),
        "Mean": density(0.5 * (plv + prv)),
        "ClosestSide": density(closest),
        "DistanceWeighted": density(dist_weighted),
    }


def read_old_handover() -> list[dict[str, float]]:
    base = ROOT / "results" / "handover" / "handover_old"
    rows = []
    with (base / "hemodynamic_summary.csv").open(newline="") as handle:
        for summary in csv.DictReader(handle):
            rows.append(
                load_case(
                    summary["archival_key"],
                    base / "data" / "per_cell_data" / f"{summary['case_id']}_per_cell_data.npz",
                )
            )
    return rows


def read_corrected_n16() -> list[dict[str, float]]:
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
    rows = []
    for label, job_id in cases:
        run_dir = next(
            (
                root / f"UKB_6beats_run_{job_id}"
                for root in roots
                if (root / f"UKB_6beats_run_{job_id}").exists()
            ),
            None,
        )
        if run_dir is None:
            raise FileNotFoundError(job_id)
        rows.append(load_case(label, run_dir / "per_cell_data.npz"))
    return rows


def print_table(name: str, rows: list[dict[str, float]]) -> None:
    proxies = ["PLV", "PRV", "Trans", "Mean", "ClosestSide", "DistanceWeighted"]
    print("\n" + "=" * 78)
    print(name)
    print("=" * 78)
    print(f"{'proxy':<18} {'r vs W_total':>14} {'r vs W_ff':>12}")
    print("-" * 50)
    for proxy in proxies:
        r_total = corr([row[proxy] for row in rows], [row["W_total"] for row in rows])
        r_ff = corr([row[proxy] for row in rows], [row["W_ff"] for row in rows])
        print(f"{proxy:<18} {r_total:+14.3f} {r_ff:+12.3f}")


def main() -> None:
    print_table("old handover n=7", read_old_handover())
    print_table("corrected n=16", read_corrected_n16())


if __name__ == "__main__":
    main()
