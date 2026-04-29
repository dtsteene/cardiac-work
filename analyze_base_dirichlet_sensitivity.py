#!/usr/bin/env python3
"""Compare basal Dirichlet variants against the current x-clamp baseline."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyze_mesh_convergence import ManifestRow, load_qois, write_csv


ROOT = Path("/home/dtsteene/D1/cardiac-work")
OUT = ROOT / "results" / "analysis" / "base_dirichlet_sensitivity"
BASELINE_MANIFEST = (
    ROOT / "results" / "analysis" / "mesh_convergence" / "submissions_20260426_232103.tsv"
)


@dataclass(frozen=True)
class VariantRow:
    base_dirichlet: str
    mesh_mm: float
    case: str
    sim_job: str
    geometry_dir: str
    circ_file: str


def read_variant_manifest(path: Path) -> list[VariantRow]:
    rows: list[VariantRow] = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                VariantRow(
                    base_dirichlet=row["base_dirichlet"],
                    mesh_mm=float(row["mesh_mm"]),
                    case=row["case"],
                    sim_job=row["sim_job"],
                    geometry_dir=row["geometry_dir"],
                    circ_file=row["circ_file"],
                )
            )
    return rows


def read_baseline_rows(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if abs(float(row["mesh_mm"]) - 5.0) > 1e-9:
                continue
            rows.append(
                ManifestRow(
                    mesh_mm=float(row["mesh_mm"]),
                    case=row["case"],
                    sim_job=row["sim_job"],
                    geometry_dir=row["geometry_dir"],
                    circ_file=row["circ_file"],
                )
            )
    return rows


def rel_diff(value: float, reference: float) -> float:
    if abs(reference) <= 1e-12:
        return float("nan")
    return (value - reference) / reference


def failure_summary(result_dir: object) -> tuple[str, str]:
    path = Path(str(result_dir)) if result_dir else None
    if path is None or not path.exists():
        return "", ""

    stage = ""
    stdout_path = path / "simulation_stdout.log"
    if stdout_path.exists():
        markers = [
            ("prestress", "Start Pre-stressing"),
            ("reference_configuration", "Deforming mesh to Reference Configuration"),
            ("fiber_mapping", "Mapping fibers to Reference Configuration"),
            ("ed_inflation", "Inflating to End-Diastolic Target"),
            ("cycle", "Coupling Time"),
        ]
        for line in stdout_path.read_text(errors="ignore").splitlines():
            for label, marker in markers:
                if marker in line:
                    stage = label

    reason = ""
    stderr_path = path / "simulation_stderr.log"
    if stderr_path.exists():
        for line in stderr_path.read_text(errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("RuntimeError:"):
                reason = stripped
                break
            if "Linear solver did not converge" in stripped and not reason:
                reason = stripped
    return stage, reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-manifest", type=Path, default=BASELINE_MANIFEST)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    baseline_rows = read_baseline_rows(args.baseline_manifest)
    baseline_qois = [load_qois(row) for row in baseline_rows]
    baseline_qois = [r for r in baseline_qois if r is not None]

    variants: list[VariantRow] = []
    for manifest in args.variant_manifest:
        variants.extend(read_variant_manifest(manifest))

    variant_qois = []
    for row in variants:
        qoi = load_qois(
            ManifestRow(
                mesh_mm=row.mesh_mm,
                case=row.case,
                sim_job=row.sim_job,
                geometry_dir=row.geometry_dir,
                circ_file=row.circ_file,
            )
        )
        if qoi is not None:
            qoi["base_dirichlet"] = row.base_dirichlet
            if qoi.get("status") != "complete":
                stage, reason = failure_summary(qoi.get("result_dir", ""))
                qoi["failure_stage"] = stage
                qoi["failure_reason"] = reason
            variant_qois.append(qoi)

    write_csv(OUT / "base_dirichlet_variant_qoi.csv", variant_qois)

    qois = [
        "LV_ESP_mmHg",
        "RV_ESP_mmHg",
        "LV_EDV_mL",
        "RV_EDV_mL",
        "LV_freewall_W_tensor_kPa",
        "RV_freewall_W_tensor_kPa",
        "Septum_W_tensor_kPa",
        "Septum_PLV_ll_kPa",
        "Septum_PRV_ll_kPa",
        "Septum_Trans_ll_kPa",
        "Septum_Mean_ll_kPa",
        "FW_tensor_LV_RV_ratio",
        "FW_adjacent_ll_LV_RV_ratio",
        "Septum_to_FWmean_tensor_ratio",
        "Septum_meanP_ll_to_FW_adjacent_ratio",
    ]

    comparisons: list[dict[str, object]] = []
    for var in variant_qois:
        if var.get("status") != "complete":
            failure_stage, failure_reason = failure_summary(var.get("result_dir", ""))
            comparisons.append({
                "case": var.get("case"),
                "base_dirichlet": var.get("base_dirichlet"),
                "sim_job": var.get("sim_job"),
                "status": var.get("status"),
                "failure_stage": failure_stage,
                "failure_reason": failure_reason,
                "variant_result_dir": var.get("result_dir", ""),
            })
            continue
        base = next(
            (
                row for row in baseline_qois
                if row.get("status") == "complete"
                and row["case"] == var["case"]
                and abs(float(row["mesh_mm"]) - float(var["mesh_mm"])) < 1e-9
            ),
            None,
        )
        if base is None:
            comparisons.append({
                "case": var.get("case"),
                "base_dirichlet": var.get("base_dirichlet"),
                "sim_job": var.get("sim_job"),
                "status": "missing_baseline",
            })
            continue
        for qoi in qois:
            value = float(var[qoi])
            reference = float(base[qoi])
            signed = rel_diff(value, reference)
            comparisons.append({
                "case": var["case"],
                "mesh_mm": var["mesh_mm"],
                "base_dirichlet": var["base_dirichlet"],
                "baseline_base_dirichlet": "x",
                "qoi": qoi,
                "value": value,
                "baseline_value": reference,
                "signed_percent_change": 100.0 * signed,
                "abs_percent_change": 100.0 * abs(signed),
                "variant_result_dir": var["result_dir"],
                "baseline_result_dir": base["result_dir"],
                "status": "complete",
            })

    write_csv(OUT / "base_dirichlet_relative_to_x.csv", comparisons)

    key_qois = {
        "RV_freewall_W_tensor_kPa",
        "Septum_W_tensor_kPa",
        "Septum_PLV_ll_kPa",
        "Septum_PRV_ll_kPa",
        "Septum_Trans_ll_kPa",
        "Septum_Mean_ll_kPa",
        "FW_adjacent_ll_LV_RV_ratio",
        "Septum_meanP_ll_to_FW_adjacent_ratio",
    }
    complete = [r for r in variant_qois if r.get("status") == "complete"]
    pending = [r for r in variant_qois if r.get("status") != "complete"]
    lines = [
        "# Base Dirichlet Sensitivity Summary",
        "",
        "Baseline is the completed h=5 mm mesh-convergence run with `base_dirichlet=x`.",
        "",
        f"Variant runs found: {len(variant_qois)}/{len(variants)}",
        f"Complete variant runs: {len(complete)}/{len(variants)}",
        "",
    ]
    if complete:
        lines += [
            "## Variant Runs",
            "",
            "| case | base Dirichlet | cells | LV ESP | RV ESP | result |",
            "|---|---|---:|---:|---:|---|",
        ]
        for row in sorted(complete, key=lambda r: (str(r["case"]), str(r["base_dirichlet"]))):
            lines.append(
                f"| {row['case']} | {row['base_dirichlet']} | {int(row['n_cells'])} | "
                f"{float(row['LV_ESP_mmHg']):.1f} | {float(row['RV_ESP_mmHg']):.1f} | "
                f"`{row['result_dir']}` |"
            )
        lines.append("")
    elif pending:
        lines += [
            "## Interpretation",
            "",
            "All submitted no-Dirichlet variants reached the reference-configuration step but failed during ED inflation. "
            "This suggests the current basal x-Dirichlet condition is acting as a necessary stability/rigid-motion control in the production setup, not merely as a small energetic spring effect.",
            "",
        ]
    key_comp = [r for r in comparisons if r.get("status") == "complete" and r["qoi"] in key_qois]
    if key_comp:
        lines += [
            "## Percent Change Versus X-Clamp Baseline",
            "",
            "| case | base Dirichlet | quantity | signed change (%) | abs change (%) |",
            "|---|---|---|---:|---:|",
        ]
        for row in sorted(key_comp, key=lambda r: (str(r["case"]), str(r["qoi"]))):
            lines.append(
                f"| {row['case']} | {row['base_dirichlet']} | `{row['qoi']}` | "
                f"{float(row['signed_percent_change']):.2f} | "
                f"{float(row['abs_percent_change']):.2f} |"
            )
        lines.append("")
    if pending:
        lines += [
            "## Failed Or Incomplete",
            "",
            "| case | base Dirichlet | job | status | last stage | reason | result |",
            "|---|---|---:|---|---|---|---|",
        ]
        for row in pending:
            lines.append(
                f"| {row.get('case')} | {row.get('base_dirichlet')} | {row.get('sim_job')} | "
                f"{row.get('status')} | {row.get('failure_stage', '')} | "
                f"{row.get('failure_reason', '')} | `{row.get('result_dir', '')}` |"
            )
        lines.append("")
    (OUT / "base_dirichlet_summary.md").write_text("\n".join(lines))

    n_complete = sum(1 for r in variant_qois if r.get("status") == "complete")
    print(f"Wrote {OUT / 'base_dirichlet_variant_qoi.csv'}")
    print(f"Wrote {OUT / 'base_dirichlet_relative_to_x.csv'}")
    print(f"Wrote {OUT / 'base_dirichlet_summary.md'}")
    print(f"Complete variant runs found: {n_complete}/{len(variants)}")


if __name__ == "__main__":
    main()
