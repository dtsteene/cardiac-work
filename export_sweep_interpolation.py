#!/usr/bin/env python3
"""Interpolated severity-sweep animation from the ED-static exports.

Reads the 8 ED-deformed case meshes
(``paraview_exports/pah_pulmonary_ed/<bundle>/ed_meshes/<case>_ed.vtu``), which all share
the same topology (one shared unloaded mesh), and linearly interpolates BOTH the
ED-deformed geometry and the per-cell density fields between consecutive cases to produce
a smooth severity sweep. Animating it shows two things at once:

  1. the septum bulging toward the LV (the D-sign) as RV pressure rises, and
  2. the work-density distribution shifting toward the RV,

so you can watch whether the qualitative behaviour tracks even though only the RV's
*integrated* work has real dynamic range across the sweep.

Two density fields are carried, kept as SEPARATE cell arrays so each can be colour-scaled
independently (pressure-strain is ~10x smaller than stress-strain, so a shared scale would
mute it):
  w_total_density_Pa            true stress-strain work density (S:dE)
  proxy_combined_ll_density_Pa  region-appropriate pressure-strain (RV->P_RV, LV+septum->P_LV)

Output (under ``paraview_exports/pah_pulmonary_sweep_interp/<bundle>/``):
  frame_NNNN.vtu        interpolated deformed mesh, both density fields + region_tag
  sweep_interp.pvd      PVD collection (timestep = interpolated RV systolic mmHg)
  clim.json             global colour ranges, SEPARATELY for ss and ps (full + robust p2-p98)

This is light post-processing (no FEniCSx) — safe to run on the login node.

Run:  python export_sweep_interpolation.py --bundle no_frank_starling [--steps 12]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pyvista as pv

import paths

CASES = ["case0_rv25", "case1_rv35", "case2_rv45", "case3_rv55",
         "case4_rv65", "case5_rv75", "case6_rv85", "case7_rv95"]
SS = "w_total_density_Pa"               # stress-strain work density
PS = "proxy_combined_ll_density_Pa"     # region-appropriate pressure-strain density
ED_ROOT = paths.RESULTS_ROOT / "paraview_exports/pah_pulmonary_ed"
OUT_ROOT = paths.RESULTS_ROOT / "paraview_exports/pah_pulmonary_sweep_interp"


def write_pvd(pvd_path: Path, entries: list[tuple[float, Path]]) -> None:
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
             "  <Collection>"]
    for t, fp in entries:
        rel = fp.relative_to(pvd_path.parent)
        lines.append(f'    <DataSet timestep="{t:.4f}" group="" part="0" '
                     f'file="{escape(str(rel))}"/>')
    lines += ["  </Collection>", "</VTKFile>"]
    pvd_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="no_frank_starling")
    ap.add_argument("--steps", type=int, default=12,
                    help="interpolation frames per case-to-case segment (default 12)")
    args = ap.parse_args()

    ed_dir = ED_ROOT / args.bundle / "ed_meshes"
    grids = [pv.read(str(ed_dir / f"{c}_ed.vtu")) for c in CASES]
    manifest = json.loads((ED_ROOT / args.bundle / "manifest.json").read_text())
    sev = [float(m["peak_p_RV_mmHg"]) for m in manifest]   # manifest order == CASES order

    n0 = grids[0].n_points
    if any(g.n_points != n0 for g in grids):
        raise SystemExit("ED meshes do not share topology — interpolation not valid")

    out = OUT_ROOT / args.bundle
    out.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[float, Path]] = []
    frame = 0
    # walk each segment [i, i+1); the very last case is appended once at the end
    for i in range(len(CASES) - 1):
        A, B = grids[i], grids[i + 1]
        ssA, ssB = np.asarray(A.cell_data[SS]), np.asarray(B.cell_data[SS])
        psA, psB = np.asarray(A.cell_data[PS]), np.asarray(B.cell_data[PS])
        for k in range(args.steps):
            t = k / args.steps
            g = A.copy(deep=True)
            g.points = (1 - t) * A.points + t * B.points
            g.cell_data[SS] = ((1 - t) * ssA + t * ssB).astype(np.float32)
            g.cell_data[PS] = ((1 - t) * psA + t * psB).astype(np.float32)
            g.field_data["rv_systolic_mmHg"] = np.array(
                [(1 - t) * sev[i] + t * sev[i + 1]], dtype=np.float32)
            fp = out / f"frame_{frame:04d}.vtu"
            g.save(str(fp))
            entries.append(((1 - t) * sev[i] + t * sev[i + 1], fp))
            frame += 1
    # final exact frame = most severe case
    gl = grids[-1].copy(deep=True)
    gl.field_data["rv_systolic_mmHg"] = np.array([sev[-1]], dtype=np.float32)
    fp = out / f"frame_{frame:04d}.vtu"
    gl.save(str(fp))
    entries.append((sev[-1], fp))

    write_pvd(out / "sweep_interp.pvd", entries)

    # Separate global colour ranges for ss and ps (full + robust percentiles).
    clim = {}
    for tag, key in (("ss", SS), ("ps", PS)):
        allv = np.concatenate([np.abs(np.asarray(g.cell_data[key])) for g in grids])
        allv = allv[np.isfinite(allv)]
        clim[tag] = {"field": key,
                     "min": float(allv.min()), "max": float(allv.max()),
                     "p2": float(np.percentile(allv, 2)),
                     "p98": float(np.percentile(allv, 98))}
    (out / "clim.json").write_text(json.dumps(clim, indent=2))

    print(f"[{args.bundle}] wrote {len(entries)} frames -> {out/'sweep_interp.pvd'}")
    print(f"  ss range (abs): {clim['ss']['min']:.3g} .. {clim['ss']['max']:.3g} "
          f"(p98 {clim['ss']['p98']:.3g})")
    print(f"  ps range (abs): {clim['ps']['min']:.3g} .. {clim['ps']['max']:.3g} "
          f"(p98 {clim['ps']['p98']:.3g})")


if __name__ == "__main__":
    main()
