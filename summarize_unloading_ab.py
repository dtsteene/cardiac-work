#!/usr/bin/env python3
"""Summarize unloading-only prestress diagnostic runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_comment(comment: str) -> tuple[str, str, str] | None:
    for prefix, sweep in (
        ("unloading_ab_", "ab"),
        ("unloading_calib_", "calib"),
    ):
        if comment.startswith(prefix):
            parts = comment.removeprefix(prefix).split("_")
            case = parts[0] if parts else "unknown"
            variant = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
            return sweep, case, variant
    return None


def read_case(result_dir: Path) -> dict[str, object] | None:
    sim_path = result_dir / "simulation_params.json"
    desc_path = result_dir / "run_description.txt"
    if not sim_path.exists() or not desc_path.exists():
        return None

    comment = desc_path.read_text(errors="replace").strip()
    parsed = parse_comment(comment)
    if parsed is None:
        return None
    sweep, case, variant = parsed

    data = json.loads(sim_path.read_text())
    unload = data.get("unloading", {})
    pre = data.get("pre_circ", {})
    material_scales = data.get("material_region_scales", {})
    if not isinstance(material_scales, dict):
        material_scales = {}

    return {
        "result_dir": str(result_dir),
        "comment": comment,
        "sweep": sweep,
        "case": case,
        "variant": variant,
        "stage": data.get("stage"),
        "p_LV_ED_mmhg": unload.get("p_LV_ED_mmhg"),
        "p_RV_ED_mmhg": unload.get("p_RV_ED_mmhg"),
        "p_RV_ED_raw_mmhg": unload.get("p_RV_ED_raw_mmhg"),
        "LV_unloaded_mL": unload.get("lvv_unloaded_mL"),
        "RV_unloaded_mL": unload.get("rvv_unloaded_mL"),
        "LV_unloaded_fraction": unload.get("lv_unloaded_fraction_of_ED"),
        "RV_unloaded_fraction": unload.get("rv_unloaded_fraction_of_ED"),
        "LV_shrink_percent": unload.get("lv_shrink_percent"),
        "RV_shrink_percent": unload.get("rv_shrink_percent"),
        "ratio_LV": unload.get("ratio_LV"),
        "ratio_RV": unload.get("ratio_RV"),
        "pre_circ_converged": pre.get("converged"),
        "pre_circ_beats": pre.get("requested_beats"),
        "pre_circ_max_volume_cycle_rel_change": pre.get("max_volume_cycle_rel_change"),
        "pre_circ_V_LV_final_cycle_delta_mL": pre.get("V_LV", {}).get("final_cycle_delta") if isinstance(pre.get("V_LV"), dict) else None,
        "pre_circ_V_RV_final_cycle_delta_mL": pre.get("V_RV", {}).get("final_cycle_delta") if isinstance(pre.get("V_RV"), dict) else None,
        "lv_material_scale": material_scales.get("LV"),
        "rv_material_scale": material_scales.get("RV"),
        "septum_material_scale": material_scales.get("Septum"),
        "material_scales": json.dumps(material_scales, sort_keys=True),
        "rv_edp_adjustments": json.dumps(data.get("rv_edp_adjustments", {}), sort_keys=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("results/sims")],
        help="Result root(s) to scan; default: results/sims",
    )
    parser.add_argument("--out", type=Path, default=Path("results/unloading_ab_summary.csv"))
    parser.add_argument(
        "--sweep",
        choices=("ab", "calib"),
        help="Only include one diagnostic sweep type.",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for root in args.roots:
        if not root.exists():
            continue
        for result_dir in sorted(root.rglob("*run_*")):
            if not result_dir.is_dir():
                continue
            row = read_case(result_dir)
            if row is not None:
                if args.sweep and row.get("sweep") != args.sweep:
                    continue
                rows.append(row)

    rows.sort(key=lambda row: (str(row["case"]), str(row["variant"]), str(row["result_dir"])))

    if not rows:
        print("No unloading_ab runs found.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out} ({len(rows)} rows)")
    print("case,variant,LV_unloaded_fraction,RV_unloaded_fraction,RV_shrink_percent,pre_circ_beats,pre_circ_drift")
    for row in rows:
        print(
            f"{row['case']},{row['variant']},"
            f"{row['LV_unloaded_fraction']},{row['RV_unloaded_fraction']},"
            f"{row['RV_shrink_percent']},{row['pre_circ_beats']},"
            f"{row['pre_circ_max_volume_cycle_rel_change']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
