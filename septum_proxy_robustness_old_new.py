#!/usr/bin/env python3
"""Old-vs-corrected robustness check for septal pressure choices.

This intentionally avoids fitting a new septal pressure.  It only tests fixed,
interpretable candidates:

    P_LV
    P_RV
    P_LV - P_RV
    mean(P_LV, P_RV)
    nearest-side pressure through the septum
    tau-weighted pressure through the septum

The point is to separate two questions:

1. Which proxy ranks cases across a pressure sweep?
2. Which proxy preserves septum/free-wall work-density ratios?

If a candidate only works after tuning to one dataset, it is not useful here.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


ROOT = Path("/home/dtsteene/D1/cardiac-work")
OUT = ROOT / "results" / "analysis" / "septum_proxy_robustness"
KPA = 1e-3

NEW_CASES = [
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

PROXIES = ["PLV", "PRV", "Trans", "Mean", "NearestSide", "TauWeighted"]
STRAINS = [("ll", "longitudinal strain"), ("ff", "fibre strain")]


def find_new_run(job_id: int) -> Path:
    for root in [
        ROOT / "results" / "sims" / "2026-04-23",
        ROOT / "results" / "sims" / "2026-04-24",
    ]:
        run = root / f"UKB_6beats_run_{job_id}"
        if run.exists():
            return run
    raise FileNotFoundError(job_id)


def read_old_cases() -> list[tuple[str, np.lib.npyio.NpzFile]]:
    base = ROOT / "results" / "handover" / "handover_old"
    cases = []
    with (base / "hemodynamic_summary.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            pc_path = base / "data" / "per_cell_data" / f"{row['case_id']}_per_cell_data.npz"
            cases.append((row["archival_key"], np.load(pc_path, allow_pickle=True)))
    return cases


def read_new_cases() -> list[tuple[str, np.lib.npyio.NpzFile]]:
    return [
        (label, np.load(find_new_run(job_id) / "per_cell_data.npz", allow_pickle=True))
        for label, job_id in NEW_CASES
    ]


def masks(pc: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    tags = pc["region_tags"]
    return {
        "LV": tags == 1,
        "RV": tags == 2,
        "Septum": pc["is_geometric_septum"].astype(bool),
    }


def density(pc: np.lib.npyio.NpzFile, mask: np.ndarray, values: str | np.ndarray) -> float:
    arr = pc[values] if isinstance(values, str) else values
    volume = float(pc["cell_volumes"][mask].sum())
    return float(-arr[mask].sum() / volume * KPA)


def candidate_arrays(pc: np.lib.npyio.NpzFile, strain_suffix: str) -> dict[str, np.ndarray]:
    # Canonical convention for pressure choices: tau=0 on the LV side and
    # tau=1 on the RV side. The saved Laplace scalar has the opposite orientation.
    tau = 1.0 - pc["lv_rv_scalar"] if "lv_rv_scalar" in pc.files else pc["tau"]
    plv = pc[f"proxy_PLV_{strain_suffix}"]
    prv = pc[f"proxy_PRV_{strain_suffix}"]
    return {
        "PLV": plv,
        "PRV": prv,
        "Trans": plv - prv,
        "Mean": 0.5 * (plv + prv),
        "NearestSide": np.where(tau < 0.5, plv, prv),
        "TauWeighted": (1.0 - tau) * plv + tau * prv,
    }


def case_rows(dataset: str, cases: list[tuple[str, np.lib.npyio.NpzFile]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case, pc in cases:
        region_masks = masks(pc)
        row: dict[str, object] = {"dataset": dataset, "case": case}
        for region, mask in region_masks.items():
            row[f"{region}_W_total"] = density(pc, mask, "w_total")
            row[f"{region}_W_ff"] = density(pc, mask, "w_ff")
            for suffix, _ in STRAINS:
                adjacent_key = (
                    f"proxy_PLV_{suffix}" if region == "LV" else f"proxy_PRV_{suffix}"
                )
                row[f"{region}_adjacent_{suffix}"] = density(pc, mask, adjacent_key)
        septum = region_masks["Septum"]
        for suffix, _ in STRAINS:
            for proxy, values in candidate_arrays(pc, suffix).items():
                row[f"Septum_{proxy}_{suffix}"] = density(pc, septum, values)
        rows.append(row)
    return rows


def corr(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(pearsonr(x_arr, y_arr)[0])


def ratio_errors(rows: list[dict[str, object]], suffix: str, proxy: str) -> tuple[float, float]:
    log_errors = []
    raw_errors = []
    for row in rows:
        septum_proxy = float(row[f"Septum_{proxy}_{suffix}"])
        for region in ("LV", "RV"):
            true_ratio = float(row["Septum_W_total"]) / float(row[f"{region}_W_total"])
            proxy_ratio = septum_proxy / float(row[f"{region}_adjacent_{suffix}"])
            raw_errors.append(abs(proxy_ratio - true_ratio))
            if true_ratio > 0 and proxy_ratio > 0:
                log_errors.append(abs(np.log(proxy_ratio / true_ratio)))
    return float(np.mean(log_errors)), float(np.mean(raw_errors))


def summary_rows(all_case_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    datasets = []
    for row in all_case_rows:
        if row["dataset"] not in datasets:
            datasets.append(str(row["dataset"]))

    for dataset in datasets:
        rows = [r for r in all_case_rows if r["dataset"] == dataset]
        for suffix, strain_label in STRAINS:
            y_total = [float(r["Septum_W_total"]) for r in rows]
            y_fibre = [float(r["Septum_W_ff"]) for r in rows]
            for proxy in PROXIES:
                x = [float(r[f"Septum_{proxy}_{suffix}"]) for r in rows]
                mean_log, mean_raw = ratio_errors(rows, suffix, proxy)
                out.append(
                    {
                        "dataset": dataset,
                        "n": len(rows),
                        "strain": strain_label,
                        "proxy": proxy,
                        "r_vs_total": corr(x, y_total),
                        "r_vs_fibre": corr(x, y_fibre),
                        "mean_abs_log_ratio_error": mean_log,
                        "mean_abs_raw_ratio_error": mean_raw,
                    }
                )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    for dataset in ["old handover n=7", "corrected n=16"]:
        print("=" * 88)
        print(dataset)
        print("=" * 88)
        for strain in ["longitudinal strain", "fibre strain"]:
            print(f"\n{strain}")
            print(f"{'proxy':<14} {'r total':>9} {'r fibre':>9} {'ratio log err':>15}")
            for proxy in PROXIES:
                row = next(
                    r
                    for r in rows
                    if r["dataset"] == dataset
                    and r["strain"] == strain
                    and r["proxy"] == proxy
                )
                print(
                    f"{proxy:<14} {float(row['r_vs_total']):>+9.3f} "
                    f"{float(row['r_vs_fibre']):>+9.3f} "
                    f"{float(row['mean_abs_log_ratio_error']):>15.3f}"
                )
        print()


def make_figure(rows: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True, constrained_layout=True)
    datasets = ["old handover n=7", "corrected n=16"]
    colors = {
        "PLV": "#4C78A8",
        "PRV": "#B279A2",
        "Trans": "#E15759",
        "Mean": "#59A14F",
        "NearestSide": "#F28E2B",
        "TauWeighted": "#76B7B2",
    }
    for ax, dataset in zip(axes, datasets):
        subset = [
            r
            for r in rows
            if r["dataset"] == dataset and r["strain"] == "longitudinal strain"
        ]
        values = [float(next(r for r in subset if r["proxy"] == p)["mean_abs_log_ratio_error"]) for p in PROXIES]
        xs = np.arange(len(PROXIES))
        ax.bar(xs, values, color=[colors[p] for p in PROXIES], width=0.65)
        ax.set_title(dataset)
        ax.set_xticks(xs)
        ax.set_xticklabels(PROXIES, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("septum/free-wall mean abs log ratio error")
    path = OUT / "fig_septum_proxy_old_new_ratio_error.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "fig_septum_proxy_old_new_ratio_error.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_cases = case_rows("old handover n=7", read_old_cases())
    all_cases += case_rows("corrected n=16", read_new_cases())
    summary = summary_rows(all_cases)
    print_summary(summary)
    write_csv(OUT / "septum_proxy_robustness_case_values.csv", all_cases)
    write_csv(OUT / "septum_proxy_robustness_summary.csv", summary)
    fig = make_figure(summary)
    print(f"Saved {OUT / 'septum_proxy_robustness_case_values.csv'}")
    print(f"Saved {OUT / 'septum_proxy_robustness_summary.csv'}")
    print(f"Saved {fig}")


if __name__ == "__main__":
    main()
