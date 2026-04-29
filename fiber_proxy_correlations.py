#!/usr/bin/env python3
"""Compare pressure-strain proxies against septal fibre-direction work.

The saved per-cell data contains both the model-resolved fibre work (`w_ff`)
and pressure-strain proxy integrals using either longitudinal strain (`*_ll`)
or fibre strain (`*_ff`). This script keeps the comparison simple and uses
the same geometric septum mask as the main proxy plots.
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


def load_per_cell(case: str, path: Path, extra: dict | None = None) -> dict:
    pc = np.load(path, allow_pickle=True)
    mask = pc["is_geometric_septum"].astype(bool)
    volume = float(pc["cell_volumes"][mask].sum())

    def density(key: str) -> float:
        return float(-pc[key][mask].sum() / volume * KPA)

    row = {
        "case": case,
        "W_total": density("w_total"),
        "W_ff": density("w_ff"),
        "W_ss": density("w_ss"),
        "W_nn": density("w_nn"),
        "W_cross": density("w_cross"),
        "ll_PLV": density("proxy_PLV_ll"),
        "ll_PRV": density("proxy_PRV_ll"),
        "ll_Trans": density("proxy_Trans_ll"),
        "ll_Mean": 0.5 * (density("proxy_PLV_ll") + density("proxy_PRV_ll")),
        "ff_PLV": density("proxy_PLV_ff"),
        "ff_PRV": density("proxy_PRV_ff"),
        "ff_Trans": density("proxy_Trans_ff"),
        "ff_Mean": 0.5 * (density("proxy_PLV_ff") + density("proxy_PRV_ff")),
    }
    if extra:
        row.update(extra)
    return row


def read_old_handover() -> list[dict]:
    base = ROOT / "results" / "handover" / "handover_old"
    rows = []
    with (base / "hemodynamic_summary.csv").open(newline="") as handle:
        for summary in csv.DictReader(handle):
            rows.append(
                load_per_cell(
                    summary["archival_key"],
                    base / "data" / "per_cell_data" / f"{summary['case_id']}_per_cell_data.npz",
                    {
                        "LV_ESP": float(summary["LV_ESP_mmHg"]),
                        "RV_ESP": float(summary["RV_ESP_mmHg"]),
                    },
                )
            )
    return rows


def read_sims(cases: list[tuple[str, int]]) -> list[dict]:
    roots = [
        ROOT / "results" / "sims" / "2026-04-23",
        ROOT / "results" / "sims" / "2026-04-24",
    ]
    rows = []
    for label, job_id in cases:
        run_dir = next((root / f"UKB_6beats_run_{job_id}" for root in roots if (root / f"UKB_6beats_run_{job_id}").exists()), None)
        if run_dir is None:
            raise FileNotFoundError(job_id)

        pressure = np.load(run_dir / "solver" / "solver_cavity_pressure_mmHg.npy")
        beat = pressure.shape[0] // 6
        last = pressure[5 * beat :]
        rows.append(
            load_per_cell(
                label,
                run_dir / "per_cell_data.npz",
                {
                    "LV_ESP": float(last[:, 0].max()),
                    "RV_ESP": float(last[:, 1].max()),
                },
            )
        )
    return rows


def values(rows: list[dict], key: str) -> np.ndarray:
    return np.array([row[key] for row in rows], dtype=float)


def print_table(name: str, rows: list[dict]) -> None:
    proxies = [
        ("ll_PLV", "LV pressure x longitudinal strain"),
        ("ll_PRV", "RV pressure x longitudinal strain"),
        ("ll_Trans", "transmural pressure x longitudinal strain"),
        ("ll_Mean", "mean pressure x longitudinal strain"),
        ("ff_PLV", "LV pressure x fibre strain"),
        ("ff_PRV", "RV pressure x fibre strain"),
        ("ff_Trans", "transmural pressure x fibre strain"),
        ("ff_Mean", "mean pressure x fibre strain"),
    ]

    print("\n" + "=" * 92)
    print(name)
    print("=" * 92)
    print(f"{'proxy':<46} {'r vs W_ff':>10} {'R2':>7} {'r vs W_total':>13} {'R2':>7}")
    print("-" * 92)
    for key, label in proxies:
        r_ff = corr(values(rows, key), values(rows, "W_ff"))
        r_total = corr(values(rows, key), values(rows, "W_total"))
        print(f"{label:<46} {r_ff:+10.3f} {r_ff * r_ff:7.3f} {r_total:+13.3f} {r_total * r_total:7.3f}")

    print("\nWork component correlations with total septal work")
    for key in ["W_ff", "W_ss", "W_nn", "W_cross"]:
        r_work = corr(values(rows, key), values(rows, "W_total"))
        print(f"  {key:<8} r={r_work:+.3f}  R2={r_work * r_work:.3f}")


def main() -> None:
    corrected_8 = [
        ("sPAP22", 1047450),
        ("sPAP30", 1047451),
        ("sPAP45", 1047452),
        ("sPAP55", 1047453),
        ("sPAP65", 1047454),
        ("sPAP75", 1047455),
        ("sPAP85", 1047456),
        ("sPAP95", 1047457),
    ]
    corrected_16 = [
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

    datasets = [
        ("old handover n=7", read_old_handover()),
        ("corrected n=8", read_sims(corrected_8)),
        ("corrected n=16", read_sims(corrected_16)),
    ]
    for name, rows in datasets:
        print_table(name, rows)


if __name__ == "__main__":
    main()
