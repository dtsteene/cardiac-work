#!/usr/bin/env python3
"""
export_per_cell_xdmf.py — Export per-cell WORK + PROXY fields to ParaView XDMF
on the canonical atlas mesh, for spatial inspection of simulation outputs.

Purely sim-dependent fields (w_total, proxy_PLV/PRV/Trans, proxy/truth ratios).
For pure-geometry fields (distances, Laplace scalars, region masks), open
<geometry_dir>/geometry_fields.xdmf — written by geometry_generator.py.

Uses the ckpt_to_cg_idx saved in per_cell_data.npz (by compute_per_cell.py
running in canonical mode with linear_sum_assignment). No u_pre recomputation
needed — the permutation is already a guaranteed bijection.

Usage:
    python3 export_per_cell_xdmf.py <result_dir1> <result_dir2> ... \\
        [--beat N] [--output-dir results/analysis/per_beat/spatial]

If --beat is omitted, uses per_cell_data.npz (last beat). If given, uses
per_cell_data_beatN.npz.
"""
import argparse
import numpy as np
from pathlib import Path
import dolfinx
import adios4dolfinx
import cardiac_geometries
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument("result_dirs", nargs="+", type=Path)
parser.add_argument("--beat", type=int, default=None,
                    help="Beat index (uses per_cell_data_beatN.npz). If omitted, uses per_cell_data.npz.")
parser.add_argument("--output-dir", type=Path,
                    default=Path("results/analysis/per_beat/spatial"))
parser.add_argument("--geometry-dir", type=Path,
                    default=Path("data/shared_ukb_mesh/ukb/geometry"),
                    help="Path to the canonical shared geometry folder.")
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)

# Load the canonical reference mesh (cg ordering) — same source every script uses
cg_geo = cardiac_geometries.geometry.Geometry.from_folder(MPI.COMM_SELF, args.geometry_dir)
mesh_cg = cg_geo.mesh
n_cg = mesh_cg.topology.index_map(3).size_local
print(f"cg mesh: {n_cg} cells")

V0 = dolfinx.fem.functionspace(mesh_cg, ("DG", 0))

def get_pc_filename(beat):
    return f"per_cell_data_beat{beat}.npz" if beat is not None else "per_cell_data.npz"

def process_case(case_dir, label, beat):
    fname = get_pc_filename(beat)
    pc_path = case_dir / fname
    if not pc_path.exists():
        print(f"  SKIP {label}: no {fname}")
        return

    pc = np.load(pc_path, allow_pickle=True)

    # Require the canonical permutation. The field may be named either
    # "ckpt_to_cg_idx" (current) or "canonical_permutation" (a naming used
    # in one transitional version). Accept either for backwards compatibility.
    if "ckpt_to_cg_idx" in pc.files:
        perm = pc["ckpt_to_cg_idx"]
    elif "canonical_permutation" in pc.files:
        perm = pc["canonical_permutation"]
    else:
        print(f"  SKIP {label}: per_cell_data missing permutation field. "
              f"Re-run compute_per_cell.py with --geometry-fields on this case.")
        return
    if len(perm) == 0:
        print(f"  SKIP {label}: ckpt_to_cg_idx is empty (case used non-canonical tagging).")
        return
    if len(perm) != n_cg:
        print(f"  SKIP {label}: permutation length ({len(perm)}) != n_cg ({n_cg}).")
        return

    # Bijection check
    n_unique = len(np.unique(perm))
    if n_unique != n_cg:
        print(f"  WARN {label}: permutation has {n_unique}/{n_cg} unique targets (not a true bijection)")

    # The permutation maps global_ckpt_idx → cg_idx.
    # The saved per_cell_data arrays are in global_ckpt order (MPI-gathered).
    # To get a cg-ordered field: cg_field[perm[i]] = ckpt_field[i] for each i.
    def mk(name, arr_ckpt):
        f = dolfinx.fem.Function(V0)
        f.name = name
        cg_field = np.zeros(n_cg, dtype=arr_ckpt.dtype if arr_ckpt.dtype != bool else np.float64)
        # fancy indexing: perm is a permutation, so this is a scatter
        cg_field[perm] = arr_ckpt.astype(cg_field.dtype)
        f.x.array[:n_cg] = cg_field
        return f

    def safe_ratio(num, den):
        r = np.divide(num, den, out=np.zeros_like(num, dtype=np.float64),
                      where=np.abs(den) > 1e-18)
        return np.clip(r, -5, 5)

    fields = [
        mk("w_total",     pc["w_total"]),
        mk("proxy_PLV",   pc["proxy_PLV_ll"]),
        mk("proxy_PRV",   pc["proxy_PRV_ll"]),
        mk("proxy_Trans", pc["proxy_Trans_ll"]),
        mk("ratio_PLV",   safe_ratio(pc["proxy_PLV_ll"],   pc["w_total"])),
        mk("ratio_Trans", safe_ratio(pc["proxy_Trans_ll"], pc["w_total"])),
    ]

    beat_label = f"beat{beat}" if beat is not None else "last"
    out_xdmf = args.output_dir / f"case_{label}_{beat_label}.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_SELF, out_xdmf, "w") as xf:
        xf.write_mesh(mesh_cg)
        for f in fields:
            xf.write_function(f, 0.0)

    n_geo_cg = int(pc["is_geometric_septum"].sum())
    n_ldrb_cg = int(pc["is_ldrb_septum"].sum())
    print(f"  {label:<20} → {out_xdmf.name}  (geo={n_geo_cg}, ldrb={n_ldrb_cg}, bijection={n_unique}/{n_cg})")

print(f"\nExporting per-cell fields on canonical mesh (beat={args.beat})")
for d in sorted(args.result_dirs):
    d = d.resolve()
    desc = (d / "run_description.txt").read_text().strip() if (d / "run_description.txt").exists() else d.name
    for pfx in ["Phase1 shared-mesh v2circ ", "Phase1 v2circ ", "v2circ "]:
        desc = desc.replace(pfx, "")
    label = desc.split()[0]
    process_case(d, label, args.beat)

print("\nDone. Open the XDMF files in ParaView.")
