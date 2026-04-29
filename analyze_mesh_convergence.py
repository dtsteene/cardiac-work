#!/usr/bin/env python3
"""Summarise the thesis mesh-convergence study.

The study is intentionally quantity-of-interest based: for each mesh size and
loading case, compute the hemodynamic and work-density quantities used in the
thesis, then compare coarser meshes against the finest completed mesh for that
case.

Typical use:

    python analyze_mesh_convergence.py \
        --manifest results/analysis/mesh_convergence/submissions_YYYYMMDD_HHMMSS.tsv

Multiple manifests can be passed to combine a baseline convergence study with
later high-resolution extension runs.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/dtsteene/D1/cardiac-work")
OUT = ROOT / "results" / "analysis" / "mesh_convergence"
KPA = 1e-3


@dataclass(frozen=True)
class ManifestRow:
    mesh_mm: float
    case: str
    sim_job: str
    geometry_dir: str
    circ_file: str


def read_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
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


def find_result_dir(job_id: str) -> Path | None:
    matches = sorted((ROOT / "results" / "sims").glob(f"*/*_run_{job_id}"))
    complete = [
        p for p in matches
        if (p / "per_cell_data.npz").exists()
        and (p / "metrics" / "metrics_downsample_1.npy").exists()
        and (p / "solver" / "solver_cavity_pressure_mmHg.npy").exists()
    ]
    if complete:
        return complete[-1]
    return matches[-1] if matches else None


def last_beat_slice(n: int, rr: float = 0.8, dt: float = 0.001) -> slice:
    steps = int(round(rr / dt))
    if n <= steps:
        return slice(None)
    return slice(n - steps, n)


def density(pc: np.lib.npyio.NpzFile, mask: np.ndarray, key: str) -> float:
    volume = float(pc["cell_volumes"][mask].sum())
    if volume <= 0:
        return float("nan")
    return float(-pc[key][mask].sum() / volume * KPA)


def region_masks(pc: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    tags = pc["region_tags"]
    return {
        "LV_freewall": tags == 1,
        "RV_freewall": tags == 2,
        "Septum": pc["is_geometric_septum"].astype(bool),
    }


def load_qois(row: ManifestRow) -> dict[str, object] | None:
    result_dir = find_result_dir(row.sim_job)
    if result_dir is None:
        return {
            "mesh_mm": row.mesh_mm,
            "case": row.case,
            "sim_job": row.sim_job,
            "result_dir": "",
            "status": "not_found",
        }

    metrics_path = result_dir / "metrics" / "metrics_downsample_1.npy"
    pc_path = result_dir / "per_cell_data.npz"
    pressure_path = result_dir / "solver" / "solver_cavity_pressure_mmHg.npy"
    if not (metrics_path.exists() and pc_path.exists() and pressure_path.exists()):
        return {
            "mesh_mm": row.mesh_mm,
            "case": row.case,
            "sim_job": row.sim_job,
            "result_dir": str(result_dir),
            "status": "incomplete",
        }

    metrics = np.load(metrics_path, allow_pickle=True).item()
    pc = np.load(pc_path, allow_pickle=True)
    pressure = np.load(pressure_path, allow_pickle=True)

    n = min(len(pressure), len(metrics["V_LV_FEM"]), len(metrics["V_RV_FEM"]))
    sl = last_beat_slice(n)
    p_lv = np.asarray(pressure[:n, 0], dtype=float)[sl]
    p_rv = np.asarray(pressure[:n, 1], dtype=float)[sl]
    v_lv = np.asarray(metrics["V_LV_FEM"], dtype=float)[:n][sl]
    v_rv = np.asarray(metrics["V_RV_FEM"], dtype=float)[:n][sl]

    masks = region_masks(pc)
    out: dict[str, object] = {
        "mesh_mm": row.mesh_mm,
        "case": row.case,
        "sim_job": row.sim_job,
        "result_dir": str(result_dir),
        "status": "complete",
        "n_cells": int(len(pc["cell_volumes"])),
        "LV_ESP_mmHg": float(np.max(p_lv)),
        "RV_ESP_mmHg": float(np.max(p_rv)),
        "LV_EDV_mL": float(np.max(v_lv)),
        "RV_EDV_mL": float(np.max(v_rv)),
        "LV_ESV_mL": float(np.min(v_lv)),
        "RV_ESV_mL": float(np.min(v_rv)),
    }

    for region, mask in masks.items():
        out[f"{region}_W_tensor_kPa"] = density(pc, mask, "w_total")
        out[f"{region}_W_ff_kPa"] = density(pc, mask, "w_ff")
        out[f"{region}_PLV_ll_kPa"] = density(pc, mask, "proxy_PLV_ll")
        out[f"{region}_PRV_ll_kPa"] = density(pc, mask, "proxy_PRV_ll")
        out[f"{region}_Trans_ll_kPa"] = density(pc, mask, "proxy_Trans_ll")
        out[f"{region}_Mean_ll_kPa"] = 0.5 * (
            out[f"{region}_PLV_ll_kPa"] + out[f"{region}_PRV_ll_kPa"]
        )

    out["FW_tensor_LV_RV_ratio"] = (
        out["LV_freewall_W_tensor_kPa"] / out["RV_freewall_W_tensor_kPa"]
    )
    out["FW_adjacent_ll_LV_RV_ratio"] = (
        out["LV_freewall_PLV_ll_kPa"] / out["RV_freewall_PRV_ll_kPa"]
    )
    out["Septum_to_FWmean_tensor_ratio"] = (
        out["Septum_W_tensor_kPa"]
        / (0.5 * (out["LV_freewall_W_tensor_kPa"] + out["RV_freewall_W_tensor_kPa"]))
    )
    out["Septum_meanP_ll_to_FW_adjacent_ratio"] = (
        out["Septum_Mean_ll_kPa"]
        / (0.5 * (out["LV_freewall_PLV_ll_kPa"] + out["RV_freewall_PRV_ll_kPa"]))
    )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> bool:
    if not rows:
        if path.exists():
            path.unlink()
        return False
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def relative_rows(rows: list[dict[str, object]], qois: Iterable[str]) -> list[dict[str, object]]:
    complete = [r for r in rows if r.get("status") == "complete"]
    out: list[dict[str, object]] = []
    for case in sorted({str(r["case"]) for r in complete}):
        case_rows = sorted(
            [r for r in complete if r["case"] == case],
            key=lambda r: float(r["mesh_mm"]),
        )
        if not case_rows:
            continue
        ref = case_rows[0]  # smallest h = finest mesh
        for row in case_rows:
            for qoi in qois:
                value = float(row[qoi])
                ref_value = float(ref[qoi])
                rel = abs(value - ref_value) / abs(ref_value) if abs(ref_value) > 1e-12 else np.nan
                out.append({
                    "case": case,
                    "mesh_mm": row["mesh_mm"],
                    "reference_mesh_mm": ref["mesh_mm"],
                    "qoi": qoi,
                    "value": value,
                    "reference_value": ref_value,
                    "relative_error": rel,
                    "percent_error": 100.0 * rel,
                })
    return out


def apparent_order_unequal(
    h1: float,
    h2: float,
    h3: float,
    phi1: float,
    phi2: float,
    phi3: float,
) -> tuple[float | None, str]:
    """Estimate p in phi(h) = phi_exact + C h^p for three unequal mesh sizes.

    h1 is the finest mesh and h3 the coarsest. If the sequence is non-monotone
    or effectively flat, return a status instead of forcing a misleading order.
    """
    d21 = phi2 - phi1
    d32 = phi3 - phi2
    scale = max(abs(phi1), abs(phi2), abs(phi3), 1.0)
    if abs(d21) < 1e-10 * scale and abs(d32) < 1e-10 * scale:
        return None, "flat"
    if d21 * d32 <= 0:
        return None, "nonmonotone"

    target = abs(d32 / d21)

    def f(p: float) -> float:
        return (h3**p - h2**p) / (h2**p - h1**p) - target

    grid = np.linspace(0.05, 10.0, 400)
    values = [f(float(p)) for p in grid]
    bracket: tuple[float, float] | None = None
    for a, b, fa, fb in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if not np.isfinite(fa) or not np.isfinite(fb):
            continue
        if fa == 0:
            return float(a), "ok"
        if fa * fb < 0:
            bracket = (float(a), float(b))
            break
    if bracket is None:
        return None, "no_positive_order"

    lo, hi = bracket
    flo = f(lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if flo * fmid <= 0:
            hi = mid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi), "ok"


def richardson_gci_rows(rows: list[dict[str, object]], qois: Iterable[str]) -> list[dict[str, object]]:
    complete = [r for r in rows if r.get("status") == "complete"]
    out: list[dict[str, object]] = []
    for case in sorted({str(r["case"]) for r in complete}):
        case_rows = sorted(
            [r for r in complete if r["case"] == case],
            key=lambda r: float(r["mesh_mm"]),
        )
        if len(case_rows) < 3:
            continue
        fine, mid, coarse = case_rows[:3]
        h1 = float(fine["mesh_mm"])
        h2 = float(mid["mesh_mm"])
        h3 = float(coarse["mesh_mm"])
        for qoi in qois:
            phi1 = float(fine[qoi])
            phi2 = float(mid[qoi])
            phi3 = float(coarse[qoi])
            p, status = apparent_order_unequal(h1, h2, h3, phi1, phi2, phi3)
            row: dict[str, object] = {
                "case": case,
                "qoi": qoi,
                "fine_mesh_mm": h1,
                "mid_mesh_mm": h2,
                "coarse_mesh_mm": h3,
                "fine_value": phi1,
                "mid_value": phi2,
                "coarse_value": phi3,
                "status": status,
            }
            if p is not None:
                c = (phi2 - phi1) / (h2**p - h1**p)
                phi_ext = phi1 - c * h1**p
                fine_rel_error = (
                    abs((phi_ext - phi1) / phi_ext)
                    if abs(phi_ext) > 1e-12 else np.nan
                )
                row.update({
                    "apparent_order": p,
                    "extrapolated_value": phi_ext,
                    "fine_relative_error": fine_rel_error,
                    "fine_error_percent": 100.0 * fine_rel_error,
                    "fine_gci_percent": 125.0 * fine_rel_error,
                })
            out.append(row)
    return out


def make_figure(rel: list[dict[str, object]], outdir: Path) -> None:
    if not rel:
        return

    qois = [
        "RV_freewall_W_tensor_kPa",
        "Septum_W_tensor_kPa",
        "FW_tensor_LV_RV_ratio",
        "FW_adjacent_ll_LV_RV_ratio",
        "Septum_meanP_ll_to_FW_adjacent_ratio",
    ]
    labels = {
        "RV_freewall_W_tensor_kPa": "RV FW tensor",
        "Septum_W_tensor_kPa": "Septum tensor",
        "FW_tensor_LV_RV_ratio": "FW tensor ratio",
        "FW_adjacent_ll_LV_RV_ratio": "FW proxy ratio",
        "Septum_meanP_ll_to_FW_adjacent_ratio": "Septum ratio",
    }
    cases = sorted({str(r["case"]) for r in rel})
    fig, axes = plt.subplots(1, len(cases), figsize=(3.6 * len(cases), 3.5), sharey=True)
    if len(cases) == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        for qoi in qois:
            pts = sorted(
                [r for r in rel if r["case"] == case and r["qoi"] == qoi],
                key=lambda r: float(r["mesh_mm"]),
            )
            if not pts:
                continue
            ax.plot(
                [float(p["mesh_mm"]) for p in pts],
                [float(p["percent_error"]) for p in pts],
                marker="o",
                label=labels[qoi],
            )
        ax.invert_xaxis()
        ax.set_title(case)
        ax.set_xlabel("Characteristic length h (mm)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Relative error vs finest mesh (%)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.suptitle("Mesh-convergence check for thesis quantities of interest")
    fig.tight_layout()
    fig.savefig(outdir / "fig_mesh_convergence_qoi.png", dpi=200, bbox_inches="tight")
    fig.savefig(outdir / "fig_mesh_convergence_qoi.pdf", bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    rows: list[dict[str, object]],
    rel: list[dict[str, object]],
    gci: list[dict[str, object]],
    outdir: Path,
) -> None:
    complete = [r for r in rows if r.get("status") == "complete"]
    pending = [r for r in rows if r.get("status") != "complete"]
    lines = [
        "# Mesh Convergence Summary",
        "",
        "This file is generated by `analyze_mesh_convergence.py`.",
        "",
        "The reference for each case is the finest completed mesh for that case.",
        "",
        f"Complete runs: {len(complete)}",
        f"Pending/incomplete runs: {len(pending)}",
        "",
    ]
    if complete:
        lines += [
            "## Completed Runs",
            "",
            "| case | h (mm) | cells | LV ESP | RV ESP | result |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for row in sorted(complete, key=lambda r: (str(r["case"]), float(r["mesh_mm"]))):
            lines.append(
                f"| {row['case']} | {float(row['mesh_mm']):g} | {int(row['n_cells'])} | "
                f"{float(row['LV_ESP_mmHg']):.1f} | {float(row['RV_ESP_mmHg']):.1f} | "
                f"`{row['result_dir']}` |"
            )
        lines.append("")
    if rel:
        key_rel = [
            r for r in rel
            if r["qoi"] in {
                "RV_freewall_W_tensor_kPa",
                "Septum_W_tensor_kPa",
                "FW_tensor_LV_RV_ratio",
                "FW_adjacent_ll_LV_RV_ratio",
                "Septum_meanP_ll_to_FW_adjacent_ratio",
            }
        ]
        lines += [
            "## Relative Error Versus Finest Mesh",
            "",
            "| case | h (mm) | quantity | error (%) |",
            "|---|---:|---|---:|",
        ]
        for row in sorted(key_rel, key=lambda r: (str(r["case"]), float(r["mesh_mm"]), str(r["qoi"]))):
            lines.append(
                f"| {row['case']} | {float(row['mesh_mm']):g} | `{row['qoi']}` | "
                f"{float(row['percent_error']):.2f} |"
            )
        lines.append("")
    if gci:
        key_gci = [
            r for r in gci
            if r["qoi"] in {
                "RV_freewall_W_tensor_kPa",
                "Septum_W_tensor_kPa",
                "FW_tensor_LV_RV_ratio",
                "FW_adjacent_ll_LV_RV_ratio",
                "Septum_meanP_ll_to_FW_adjacent_ratio",
            }
        ]
        lines += [
            "## Three-Mesh Richardson/GCI Check",
            "",
            "The GCI column is reported only where the finest, middle, and coarsest values form a usable monotone sequence.",
            "",
            "| case | quantity | status | apparent order | fine-grid GCI (%) |",
            "|---|---|---|---:|---:|",
        ]
        for row in sorted(key_gci, key=lambda r: (str(r["case"]), str(r["qoi"]))):
            order = row.get("apparent_order", "")
            gci_percent = row.get("fine_gci_percent", "")
            if order != "":
                lines.append(
                    f"| {row['case']} | `{row['qoi']}` | {row['status']} | "
                    f"{float(order):.2f} | {float(gci_percent):.2f} |"
                )
            else:
                lines.append(
                    f"| {row['case']} | `{row['qoi']}` | {row['status']} |  |  |"
                )
        lines.append("")
    if pending:
        lines += ["## Pending Or Incomplete", ""]
        for row in pending:
            lines.append(f"- {row.get('case')} h={row.get('mesh_mm')} job={row.get('sim_job')}")
        lines.append("")
    (outdir / "mesh_convergence_summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[ManifestRow] = []
    for manifest in args.manifest:
        manifest_rows.extend(read_manifest(manifest))
    rows = [r for r in (load_qois(row) for row in manifest_rows) if r is not None]
    qoi_path = OUT / "mesh_convergence_qoi.csv"
    rel_path = OUT / "mesh_convergence_relative_to_finest.csv"
    gci_path = OUT / "mesh_convergence_richardson_gci.csv"
    wrote_qoi = write_csv(qoi_path, rows)

    qois = [
        "LV_ESP_mmHg",
        "RV_ESP_mmHg",
        "LV_EDV_mL",
        "RV_EDV_mL",
        "LV_freewall_W_tensor_kPa",
        "RV_freewall_W_tensor_kPa",
        "Septum_W_tensor_kPa",
        "LV_freewall_PLV_ll_kPa",
        "RV_freewall_PRV_ll_kPa",
        "Septum_PLV_ll_kPa",
        "Septum_PRV_ll_kPa",
        "Septum_Trans_ll_kPa",
        "Septum_Mean_ll_kPa",
        "FW_tensor_LV_RV_ratio",
        "FW_adjacent_ll_LV_RV_ratio",
        "Septum_to_FWmean_tensor_ratio",
        "Septum_meanP_ll_to_FW_adjacent_ratio",
    ]
    rel = relative_rows(rows, qois)
    wrote_rel = write_csv(rel_path, rel)
    gci = richardson_gci_rows(rows, qois)
    wrote_gci = write_csv(gci_path, gci)
    make_figure(rel, OUT)
    write_markdown(rows, rel, gci, OUT)

    complete = sum(1 for r in rows if r.get("status") == "complete")
    if wrote_qoi:
        print(f"Wrote {qoi_path}")
    if wrote_rel:
        print(f"Wrote {rel_path}")
    else:
        print("No completed relative-error rows yet")
    if wrote_gci:
        print(f"Wrote {gci_path}")
    else:
        print("No completed three-mesh GCI rows yet")
    print(f"Wrote {OUT / 'mesh_convergence_summary.md'}")
    print(f"Complete runs found: {complete}/{len(manifest_rows)}")


if __name__ == "__main__":
    main()
