#!/usr/bin/env python3
"""Analyze the small patient-mesh x pressure robustness check.

This reads FEM job manifests written by submit_patient_mesh_fem.sbatch and,
for each mesh separately, computes within-mesh correlations across pressure
cases. Do not pool meshes first; pooling mixes geometry and pressure effects.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


ROOT = Path("/home/dtsteene/D1/cardiac-work")
MANIFEST_ROOT = ROOT / "results" / "patient_mesh_sweep"
SIM_ROOT = ROOT / "results" / "sims"
KPA = 1e-3

CASE_ORDER = ["sPAP22", "sPAP30", "sPAP45", "sPAP55", "sPAP65", "sPAP75", "sPAP85", "sPAP95"]
REGIONS = ["Whole", "LV", "RV", "Septum"]
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
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(pearsonr(x, y)[0])


def find_result_dir(job_id: str) -> Path | None:
    matches = sorted(SIM_ROOT.glob(f"*/**/*_run_{job_id}"))
    return matches[-1] if matches else None


def read_manifests() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not MANIFEST_ROOT.exists():
        return rows
    for path in sorted(MANIFEST_ROOT.glob("*_jobs_*.tsv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                row["manifest"] = str(path)
                rows.append(row)
    return rows


def region_masks(pc: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    whole = np.ones_like(pc["region_tags"], dtype=bool)
    return {
        "Whole": whole,
        "LV": pc["region_tags"] == 1,
        "RV": pc["region_tags"] == 2,
        "Septum": pc["is_geometric_septum"].astype(bool),
    }


def density(pc: np.lib.npyio.NpzFile, mask: np.ndarray, key: str) -> float:
    volume = float(pc["cell_volumes"][mask].sum())
    return float(-pc[key][mask].sum() / volume * KPA)


def load_case(row: dict[str, str]) -> dict[str, object] | None:
    result_dir = find_result_dir(row["job_id"])
    if result_dir is None:
        return None
    pc_path = result_dir / "per_cell_data.npz"
    pressure_path = result_dir / "solver" / "solver_cavity_pressure_mmHg.npy"
    if not pc_path.exists() or not pressure_path.exists():
        return None

    pc = np.load(pc_path, allow_pickle=True)
    pressure = np.load(pressure_path)
    beat = pressure.shape[0] // 6
    last = pressure[5 * beat :]

    out: dict[str, object] = {
        "mesh_key": row["mesh_key"],
        "case": row["case"],
        "job_id": row["job_id"],
        "result_dir": str(result_dir),
        "LV_ESP": float(last[:, 0].max()),
        "RV_ESP": float(last[:, 1].max()),
        "Trans_ESP": float(last[:, 0].max() - last[:, 1].max()),
    }

    for region, mask in region_masks(pc).items():
        plv = density(pc, mask, "proxy_PLV_ll")
        prv = density(pc, mask, "proxy_PRV_ll")
        out[region] = {
            "W_total": density(pc, mask, "w_total"),
            "W_ff": density(pc, mask, "w_ff"),
            "W_ss": density(pc, mask, "w_ss"),
            "W_nn": density(pc, mask, "w_nn"),
            "W_cross": density(pc, mask, "w_cross"),
            "PLV": plv,
            "PRV": prv,
            "Trans": density(pc, mask, "proxy_Trans_ll"),
            "Mean": 0.5 * (plv + prv),
        }
    return out


def values(rows: list[dict[str, object]], region: str, key: str) -> np.ndarray:
    return np.array([row[region][key] for row in rows], dtype=float)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    manifest_rows = read_manifests()
    loaded = [case for row in manifest_rows if (case := load_case(row)) is not None]
    if not loaded:
        print(f"No completed patient-mesh FEM cases found from {MANIFEST_ROOT}")
        return

    out_dir = ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    hemo_rows: list[dict[str, object]] = []

    for row in loaded:
        hemo_rows.append(
            {
                "mesh_key": row["mesh_key"],
                "case": row["case"],
                "job_id": row["job_id"],
                "LV_ESP": row["LV_ESP"],
                "RV_ESP": row["RV_ESP"],
                "Trans_ESP": row["Trans_ESP"],
                "result_dir": row["result_dir"],
            }
        )

    for mesh_key in sorted({str(row["mesh_key"]) for row in loaded}):
        mesh_rows = [row for row in loaded if row["mesh_key"] == mesh_key]
        mesh_rows.sort(key=lambda row: CASE_ORDER.index(str(row["case"])) if row["case"] in CASE_ORDER else 999)
        if len(mesh_rows) < 3:
            continue
        for region in REGIONS:
            for target_key, target_label in TARGETS:
                y = values(mesh_rows, region, target_key)
                for proxy_key, proxy_label in PROXIES:
                    r = corr(values(mesh_rows, region, proxy_key), y)
                    corr_rows.append(
                        {
                            "mesh_key": mesh_key,
                            "n": len(mesh_rows),
                            "region": region,
                            "target": target_label,
                            "target_key": target_key,
                            "proxy": proxy_label,
                            "proxy_key": proxy_key,
                            "r": r,
                            "r2": r * r,
                        }
                    )

            total = values(mesh_rows, region, "W_total")
            for component in ["W_ff", "W_ss", "W_nn", "W_cross"]:
                r = corr(values(mesh_rows, region, component), total)
                component_rows.append(
                    {
                        "mesh_key": mesh_key,
                        "n": len(mesh_rows),
                        "region": region,
                        "component": component,
                        "r_vs_W_total": r,
                        "r2_vs_W_total": r * r,
                    }
                )

    write_csv(out_dir / "patient_mesh_pressure_correlations.csv", corr_rows)
    write_csv(out_dir / "patient_mesh_pressure_components.csv", component_rows)
    write_csv(out_dir / "patient_mesh_pressure_hemodynamics.csv", hemo_rows)

    print("Completed cases:")
    for mesh_key in sorted({str(row["mesh_key"]) for row in loaded}):
        rows = [row for row in hemo_rows if row["mesh_key"] == mesh_key]
        print(f"  {mesh_key}: {len(rows)}")

    print("\nSeptum proxy correlations by mesh")
    print(f"{'mesh':<10} {'target':<18} {'P_LV':>7} {'P_RV':>7} {'Trans':>7} {'Mean':>7}")
    for mesh_key in sorted({str(row["mesh_key"]) for row in loaded}):
        for target_key, target_label in TARGETS:
            vals = []
            for proxy_key, _ in PROXIES:
                match = next(
                    (
                        row
                        for row in corr_rows
                        if row["mesh_key"] == mesh_key
                        and row["region"] == "Septum"
                        and row["target_key"] == target_key
                        and row["proxy_key"] == proxy_key
                    ),
                    None,
                )
                vals.append(float(match["r"]) if match is not None else np.nan)
            print(
                f"{mesh_key:<10} {target_label:<18} "
                f"{vals[0]:+7.3f} {vals[1]:+7.3f} {vals[2]:+7.3f} {vals[3]:+7.3f}"
            )

    print(f"\nWrote {out_dir / 'patient_mesh_pressure_correlations.csv'}")
    print(f"Wrote {out_dir / 'patient_mesh_pressure_components.csv'}")
    print(f"Wrote {out_dir / 'patient_mesh_pressure_hemodynamics.csv'}")


if __name__ == "__main__":
    main()
