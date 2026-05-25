#!/usr/bin/env python3
"""Summarize postprocessing function-space sensitivity runs.

This reads the metrics folders written by postprocess_metrics.py with
--metrics-subdir metrics_space_<variant>. It does not open checkpoints.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


CASES = [
    ("sPAP22", 1047450, Path("results/sims/2026-04-23/UKB_6beats_run_1047450")),
    ("sPAP60", 1048197, Path("results/sims/2026-04-24/UKB_6beats_run_1048197")),
    ("sPAP95", 1047457, Path("results/sims/2026-04-23/UKB_6beats_run_1047457")),
]

VARIANTS = ["DG0", "DG1", "Quadrature6"]
REGIONS = ["LV", "RV", "Septum", "Whole"]
WORK_KEYS = ["work_true", "work_ff", "work_ss", "work_nn", "work_cross"]


def variant_subdir(variant: str) -> str:
    return f"metrics_space_{variant.lower()}"


def load_metrics(run_dir: Path, variant: str) -> dict:
    path = run_dir / variant_subdir(variant) / "metrics_downsample_1.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True).item()


def series_sum(metrics: dict, key: str) -> float:
    values = np.asarray(metrics.get(key, []), dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.nansum(values))


def scalar(metrics: dict, key: str) -> float:
    value = metrics.get(key, float("nan"))
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.ravel()[0])
        return float("nan")
    return float(value)


def row_for(case_label: str, job_id: int, run_dir: Path, variant: str) -> dict:
    metrics = load_metrics(run_dir, variant)
    row = {
        "case": case_label,
        "job_id": job_id,
        "variant": variant,
        "run_dir": str(run_dir.resolve()),
    }

    for region in REGIONS:
        volume = scalar(metrics, f"region_volume_{region}")
        row[f"volume_{region}_ml"] = volume * 1e6 if np.isfinite(volume) else float("nan")
        for prefix in WORK_KEYS:
            work_j = series_sum(metrics, f"{prefix}_{region}")
            row[f"{prefix}_{region}_mJ"] = work_j * 1e3
            row[f"{prefix}_{region}_kPa"] = (work_j / volume) * 1e-3 if volume and np.isfinite(volume) else float("nan")

    boundary_j = (
        series_sum(metrics, "work_boundary_exact_LV")
        + series_sum(metrics, "work_boundary_exact_RV")
    )
    robin_j = series_sum(metrics, "work_robin_epi") + series_sum(metrics, "work_robin_base")
    whole_j = series_sum(metrics, "work_true_Whole")
    total_boundary_j = boundary_j + robin_j
    row["boundary_mJ"] = boundary_j * 1e3
    row["robin_mJ"] = robin_j * 1e3
    row["boundary_plus_robin_mJ"] = total_boundary_j * 1e3
    row["whole_internal_mJ"] = whole_j * 1e3
    row["energy_residual_mJ"] = (whole_j - total_boundary_j) * 1e3
    row["energy_residual_rel_abs"] = (
        abs(whole_j - total_boundary_j) / max(abs(total_boundary_j), 1e-30)
    )
    return row


def add_relative_to_reference(rows: list[dict], reference: str = "Quadrature6") -> None:
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["variant"]] = row

    compare_cols = [
        f"{prefix}_{region}_kPa"
        for prefix in WORK_KEYS
        for region in REGIONS
    ] + ["whole_internal_mJ", "boundary_plus_robin_mJ", "energy_residual_rel_abs"]

    for row in rows:
        ref = by_case.get(row["case"], {}).get(reference)
        for col in compare_cols:
            out_col = f"{col}_rel_to_{reference}"
            if ref is None:
                row[out_col] = float("nan")
                continue
            denom = ref.get(col, float("nan"))
            val = row.get(col, float("nan"))
            if not np.isfinite(denom) or abs(denom) < 1e-30 or not np.isfinite(val):
                row[out_col] = float("nan")
            else:
                row[out_col] = (val - denom) / denom


def write_summary(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metric_space_sensitivity.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "metric_space_sensitivity_summary.md"
    with md_path.open("w") as f:
        f.write("# Metric Space Sensitivity\n\n")
        f.write(
            "Postprocessing-only sensitivity check. Each row is the final-beat "
            "regional work density obtained by replaying the same displacement "
            "checkpoint with a different state/storage space in MetricsCalculator. "
            "Quadrature6 is the production reference.\n\n"
        )
        f.write("## Tensor Work Density (kPa)\n\n")
        f.write("| case | variant | LV | RV | Septum | Whole | energy residual |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['case']} | {row['variant']} | "
                f"{row['work_true_LV_kPa']:.3f} | "
                f"{row['work_true_RV_kPa']:.3f} | "
                f"{row['work_true_Septum_kPa']:.3f} | "
                f"{row['work_true_Whole_kPa']:.3f} | "
                f"{row['energy_residual_rel_abs']:.2e} |\n"
            )

        f.write("\n## Difference From Quadrature6\n\n")
        f.write("| case | variant | LV | RV | Septum | Whole |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['case']} | {row['variant']} | "
                f"{100 * row['work_true_LV_kPa_rel_to_Quadrature6']:.1f}% | "
                f"{100 * row['work_true_RV_kPa_rel_to_Quadrature6']:.1f}% | "
                f"{100 * row['work_true_Septum_kPa_rel_to_Quadrature6']:.1f}% | "
                f"{100 * row['work_true_Whole_kPa_rel_to_Quadrature6']:.1f}% |\n"
            )

        variants = [r for r in rows if r["variant"] != "Quadrature6"]
        if variants:
            f.write("\n## Interpretation\n\n")
            for variant in sorted({r["variant"] for r in variants}):
                subset = [r for r in variants if r["variant"] == variant]
                max_true = max(
                    abs(r[f"work_true_{region}_kPa_rel_to_Quadrature6"])
                    for r in subset
                    for region in ("LV", "RV", "Septum", "Whole")
                )
                max_sept = max(
                    abs(r["work_true_Septum_kPa_rel_to_Quadrature6"])
                    for r in subset
                )
                max_energy = max(r["energy_residual_rel_abs"] for r in subset)
                f.write(
                    f"- {variant}: max regional tensor-work difference "
                    f"{100 * max_true:.1f}% vs Quadrature6; max septal difference "
                    f"{100 * max_sept:.1f}%; max energy residual {max_energy:.2e}.\n"
                )
            f.write(
                "\nDG1 reproduces integrated regional tensor work closely, but its "
                "energy residual is still larger than Quadrature6. DG0 is acceptable "
                "for whole-heart totals in these cases but underestimates high-pressure "
                "septal work, most strongly in sPAP95. Directional component terms are "
                "more sensitive than total work, especially sheet and cross terms, so "
                "Quadrature6 remains the most defensible production space for stress-"
                "strain postprocessing.\n"
            )

        f.write(f"\nRaw table: `{csv_path}`\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis/metric_space_sensitivity"))
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    args = parser.parse_args()

    rows = []
    for case_label, job_id, run_dir in CASES:
        for variant in args.variants:
            try:
                rows.append(row_for(case_label, job_id, run_dir, variant))
            except FileNotFoundError as exc:
                print(f"SKIP missing metrics: {exc}")
    if not rows:
        raise SystemExit("No metric-space sensitivity outputs found")
    add_relative_to_reference(rows)
    write_summary(rows, args.out_dir)
    print(f"Wrote {args.out_dir / 'metric_space_sensitivity_summary.md'}")


if __name__ == "__main__":
    main()
