#!/usr/bin/env python3
"""Regional correlations for longitudinal pressure-strain proxies.

This makes the "which pressure works best?" question explicit. The strain
direction is held fixed as longitudinal strain (`eps_ll`). Only the pressure
choice changes:

  * P_LV
  * P_RV
  * P_LV - P_RV
  * mean(P_LV, P_RV)

For each dataset and region, the script correlates each proxy with both
model-resolved total tensor work (`w_total`) and fibre-direction work (`w_ff`).
It writes a long CSV and a compact Markdown summary.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


ROOT = Path("/home/dtsteene/D1/cardiac-work")
OUT = ROOT / "results" / "analysis"
KPA = 1e-3


PROXIES = [
    ("PLV", "P_LV x eps_ll"),
    ("PRV", "P_RV x eps_ll"),
    ("Trans", "(P_LV-P_RV) x eps_ll"),
    ("Mean", "mean(P_LV,P_RV) x eps_ll"),
]

TARGETS = [
    ("W_total", "total tensor work"),
    ("W_ff", "fibre work"),
]


def corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(pearsonr(x, y)[0])


def masks(pc: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    lv = pc["region_tags"] == 1
    rv = pc["region_tags"] == 2
    septum = pc["is_geometric_septum"].astype(bool)
    whole = np.ones_like(pc["region_tags"], dtype=bool)
    return {
        "Whole": whole,
        "LV": lv,
        "RV": rv,
        "Septum": septum,
    }


def density(pc: np.lib.npyio.NpzFile, mask: np.ndarray, key: str) -> float:
    volume = float(pc["cell_volumes"][mask].sum())
    return float(-pc[key][mask].sum() / volume * KPA)


def case_values(case: str, pc_path: Path) -> dict[str, dict[str, float]]:
    pc = np.load(pc_path, allow_pickle=True)
    out: dict[str, dict[str, float]] = {}
    for region, mask in masks(pc).items():
        plv = density(pc, mask, "proxy_PLV_ll")
        prv = density(pc, mask, "proxy_PRV_ll")
        trans = density(pc, mask, "proxy_Trans_ll")
        out[region] = {
            "case": case,
            "W_total": density(pc, mask, "w_total"),
            "W_ff": density(pc, mask, "w_ff"),
            "W_ss": density(pc, mask, "w_ss"),
            "W_nn": density(pc, mask, "w_nn"),
            "W_cross": density(pc, mask, "w_cross"),
            "PLV": plv,
            "PRV": prv,
            "Trans": trans,
            "Mean": 0.5 * (plv + prv),
        }
    return out


def read_old_handover() -> list[dict[str, dict[str, float]]]:
    base = ROOT / "results" / "handover" / "handover_old"
    rows = []
    with (base / "hemodynamic_summary.csv").open(newline="") as handle:
        for summary in csv.DictReader(handle):
            rows.append(
                case_values(
                    summary["archival_key"],
                    base / "data" / "per_cell_data" / f"{summary['case_id']}_per_cell_data.npz",
                )
            )
    return rows


def read_sims(cases: list[tuple[str, int]]) -> list[dict[str, dict[str, float]]]:
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
        rows.append(case_values(label, run_dir / "per_cell_data.npz"))
    return rows


def values(rows: list[dict[str, dict[str, float]]], region: str, key: str) -> np.ndarray:
    return np.array([row[region][key] for row in rows], dtype=float)


def make_long_rows(dataset: str, rows: list[dict[str, dict[str, float]]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for region in ["Whole", "LV", "RV", "Septum"]:
        for target_key, target_label in TARGETS:
            y = values(rows, region, target_key)
            for proxy_key, proxy_label in PROXIES:
                r = corr(values(rows, region, proxy_key), y)
                out.append(
                    {
                        "dataset": dataset,
                        "n": len(rows),
                        "region": region,
                        "target": target_label,
                        "target_key": target_key,
                        "proxy": proxy_label,
                        "proxy_key": proxy_key,
                        "r": r,
                        "r2": r * r,
                    }
                )
    return out


def best_proxy_rows(dataset: str, rows: list[dict[str, dict[str, float]]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for region in ["Whole", "LV", "RV", "Septum"]:
        for target_key, target_label in TARGETS:
            y = values(rows, region, target_key)
            rs = [(proxy_label, corr(values(rows, region, proxy_key), y)) for proxy_key, proxy_label in PROXIES]
            best = max(rs, key=lambda item: item[1])
            out.append(
                {
                    "dataset": dataset,
                    "region": region,
                    "target": target_label,
                    "best_proxy": best[0],
                    "best_r": best[1],
                    "best_r2": best[1] * best[1],
                }
            )
    return out


def component_rows(dataset: str, rows: list[dict[str, dict[str, float]]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    components = ["W_ff", "W_ss", "W_nn", "W_cross"]
    for region in ["Whole", "LV", "RV", "Septum"]:
        total = values(rows, region, "W_total")
        for component in components:
            r = corr(values(rows, region, component), total)
            out.append(
                {
                    "dataset": dataset,
                    "n": len(rows),
                    "region": region,
                    "component": component,
                    "r_vs_W_total": r,
                    "r2_vs_W_total": r * r,
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    old_name = "old handover n=7"
    new_name = "corrected n=16"
    out = []
    for region in ["Whole", "LV", "RV", "Septum"]:
        for target_key, target_label in TARGETS:
            for proxy_key, proxy_label in PROXIES:
                old = next(
                    row
                    for row in long_rows
                    if row["dataset"] == old_name
                    and row["region"] == region
                    and row["target_key"] == target_key
                    and row["proxy_key"] == proxy_key
                )
                new = next(
                    row
                    for row in long_rows
                    if row["dataset"] == new_name
                    and row["region"] == region
                    and row["target_key"] == target_key
                    and row["proxy_key"] == proxy_key
                )
                old_r = float(old["r"])
                new_r = float(new["r"])
                out.append(
                    {
                        "region": region,
                        "target": target_label,
                        "target_key": target_key,
                        "proxy": proxy_label,
                        "proxy_key": proxy_key,
                        "old_r": old_r,
                        "corrected_n16_r": new_r,
                        "delta_r": new_r - old_r,
                        "old_r2": old_r * old_r,
                        "corrected_n16_r2": new_r * new_r,
                    }
                )
    return out


def fmt_r(x: float) -> str:
    if np.isnan(x):
        return "nan"
    return f"{x:+.3f}"


def markdown_table(long_rows: list[dict[str, object]], best_rows: list[dict[str, object]], component_rows_: list[dict[str, object]]) -> str:
    lines = [
        "# Regional longitudinal proxy correlations",
        "",
        "Strain direction is fixed to longitudinal strain (`eps_ll`). Only pressure choice changes.",
        "Targets are model-resolved total tensor work (`w_total`) and fibre-direction work (`w_ff`).",
        "",
    ]

    datasets = []
    for row in long_rows:
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])

    for dataset in datasets:
        lines += [f"## {dataset}", ""]
        for region in ["Whole", "LV", "RV", "Septum"]:
            lines += [f"### {region}", ""]
            lines.append("| target | P_LV | P_RV | P_LV-P_RV | mean(P_LV,P_RV) |")
            lines.append("|---|---:|---:|---:|---:|")
            for target_key, target_label in TARGETS:
                row_bits = []
                for proxy_key, _ in PROXIES:
                    match = next(
                        row
                        for row in long_rows
                        if row["dataset"] == dataset
                        and row["region"] == region
                        and row["target_key"] == target_key
                        and row["proxy_key"] == proxy_key
                    )
                    row_bits.append(fmt_r(float(match["r"])))
                lines.append(f"| {target_label} | {' | '.join(row_bits)} |")
            lines.append("")

        lines += ["Best proxy by positive Pearson r:", ""]
        lines.append("| region | target | best proxy | r | R2 |")
        lines.append("|---|---|---|---:|---:|")
        for row in best_rows:
            if row["dataset"] == dataset:
                lines.append(
                    f"| {row['region']} | {row['target']} | {row['best_proxy']} | "
                    f"{fmt_r(float(row['best_r']))} | {float(row['best_r2']):.3f} |"
                )
        lines.append("")

        lines += ["Work-component correlations with total tensor work:", ""]
        lines.append("| region | W_ff | W_ss | W_nn | W_cross |")
        lines.append("|---|---:|---:|---:|---:|")
        for region in ["Whole", "LV", "RV", "Septum"]:
            vals = []
            for component in ["W_ff", "W_ss", "W_nn", "W_cross"]:
                match = next(
                    row
                    for row in component_rows_
                    if row["dataset"] == dataset and row["region"] == region and row["component"] == component
                )
                vals.append(fmt_r(float(match["r_vs_W_total"])))
            lines.append(f"| {region} | {' | '.join(vals)} |")
        lines.append("")
    return "\n".join(lines)


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

    OUT.mkdir(parents=True, exist_ok=True)
    long_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    comp_rows: list[dict[str, object]] = []
    for dataset, rows in datasets:
        long_rows += make_long_rows(dataset, rows)
        best_rows += best_proxy_rows(dataset, rows)
        comp_rows += component_rows(dataset, rows)
    compare_rows = comparison_rows(long_rows)

    write_csv(OUT / "regional_ll_proxy_correlations.csv", long_rows)
    write_csv(OUT / "regional_ll_proxy_best.csv", best_rows)
    write_csv(OUT / "regional_work_component_correlations.csv", comp_rows)
    write_csv(OUT / "regional_ll_proxy_old_vs_corrected_n16.csv", compare_rows)
    (OUT / "regional_ll_proxy_correlations.md").write_text(
        markdown_table(long_rows, best_rows, comp_rows)
    )

    print(f"Wrote {OUT / 'regional_ll_proxy_correlations.csv'}")
    print(f"Wrote {OUT / 'regional_ll_proxy_best.csv'}")
    print(f"Wrote {OUT / 'regional_work_component_correlations.csv'}")
    print(f"Wrote {OUT / 'regional_ll_proxy_old_vs_corrected_n16.csv'}")
    print(f"Wrote {OUT / 'regional_ll_proxy_correlations.md'}")


if __name__ == "__main__":
    main()
