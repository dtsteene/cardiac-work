#!/usr/bin/env python3
"""
viz_region_split.py — Write one XDMF with all region-split fields.

Builds a single XDMF file on the canonical UKB shared mesh where every
region-definition scalar is a ParaView-toggleable field. Lets the reader
see at a glance how the different tau conventions and the LDRB tagging
partition the wall.

Fields written (all per-cell DG0 scalars):

    Region masks (bool, shown as 0/1)
      region_tag             LDRB marker (1=LV free wall, 2=RV FW, 3=Septum)
      is_ldrb_septum         LDRB-scalar septum mask (marker==3)
      is_geometric_septum    max(d_LV,d_RV) < d_epi
      envelope               study envelope
      study_region           (is_ldrb ∪ is_geo) ∩ envelope

    Tau fields (scalar)
      tau_eu                 d_LV / (d_LV + d_RV)  — Euclidean
      tau_lap                lv_rv_scalar          — Laplace
      side_eu                1 if tau_eu < 0.5 else 2  (categorical LV/RV)
      side_lap               1 if tau_lap > 0.5 else 2  (categorical LV/RV)

    Distance fields (m)
      d_lv                   distance to LV endo
      d_rv                   distance to RV endo
      d_epi                  distance to epicardium
      entry_t                max(d_LV,d_RV) - d_epi  (sweep threshold)

Usage: python3 viz_region_split.py
"""
import numpy as np
from pathlib import Path
from mpi4py import MPI
import dolfinx
import cardiac_geometries

OUT = Path("/home/dtsteene/D1/cardiac-work/results/analysis/cascade")
OUT.mkdir(parents=True, exist_ok=True)

SHARED = "/home/dtsteene/D1/cardiac-work/data/shared_ukb_mesh/ukb/geometry"
geo = cardiac_geometries.geometry.Geometry.from_folder(MPI.COMM_SELF, SHARED)
mesh = geo.mesh
n_cg = mesh.topology.index_map(3).size_local
print(f"Canonical shared mesh: {n_cg} cells")

PC_PATH = ("/home/dtsteene/D1/cardiac-work/results/sims/"
           "2026-04-12/UKB_6beats_run_1020849/per_cell_data.npz")
pc = np.load(PC_PATH, allow_pickle=True)
print(f"Loaded per_cell data: {len(pc['cell_volumes'])} cells, "
      f"tagging_mode={pc['tagging_mode']}")

perm = pc["ckpt_to_cg_idx"]
if len(perm) != n_cg:
    raise SystemExit(f"ckpt_to_cg_idx len {len(perm)} != n_cg {n_cg}")

def to_cg(ckpt_array):
    cg = np.zeros(n_cg, dtype=np.float64)
    cg[perm] = np.asarray(ckpt_array, dtype=np.float64)
    return cg

fields_spec = [
    ("region_tag",          to_cg(pc["region_tags"])),
    ("is_ldrb_septum",      to_cg(pc["is_ldrb_septum"])),
    ("is_geometric_septum", to_cg(pc["is_geometric_septum"])),
    ("envelope",            to_cg(pc["envelope"])),
    ("study_region",        to_cg(pc["study_region"])),
    ("tau_eu",              to_cg(pc["tau"])),
    ("tau_lap",             to_cg(pc["lv_rv_scalar"])),
    ("side_eu",             to_cg(np.where(pc["tau"] < 0.5, 1, 2))),
    ("side_lap",            to_cg(np.where(pc["lv_rv_scalar"] > 0.5, 1, 2))),
    ("d_lv",                to_cg(pc["d_lv"])),
    ("d_rv",                to_cg(pc["d_rv"])),
    ("d_epi",               to_cg(pc["d_epi"])),
    ("entry_t",             to_cg(pc["entry_t"])),
]

V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
out_xdmf = OUT / "region_split.xdmf"
with dolfinx.io.XDMFFile(MPI.COMM_SELF, out_xdmf, "w") as xf:
    xf.write_mesh(mesh)
    for name, arr in fields_spec:
        f = dolfinx.fem.Function(V0)
        f.name = name
        f.x.array[:n_cg] = arr.astype(np.float64)
        xf.write_function(f, 0.0)
        print(f"  wrote {name:<22}  min={arr.min():+.4e}  max={arr.max():+.4e}")

print(f"\nSaved {out_xdmf}")

import json
summary = {
    "mesh": SHARED,
    "n_cells": int(n_cg),
    "counts": {
        "LV_free_wall_ldrb":  int((pc["region_tags"] == 1).sum()),
        "RV_free_wall_ldrb":  int((pc["region_tags"] == 2).sum()),
        "septum_ldrb":        int((pc["region_tags"] == 3).sum()),
        "septum_geometric":   int(pc["is_geometric_septum"].sum()),
        "LV_tau_eu":          int((pc["tau"] < 0.5).sum()),
        "RV_tau_eu":          int((pc["tau"] >= 0.5).sum()),
        "LV_tau_lap":         int((pc["lv_rv_scalar"] > 0.5).sum()),
        "RV_tau_lap":         int((pc["lv_rv_scalar"] <= 0.5).sum()),
        "envelope":           int(pc["envelope"].sum()),
    },
    "fields_written": [name for name, _ in fields_spec],
}
with open(OUT / "region_split_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved {OUT / 'region_split_summary.json'}")
print("\nParaView: open region_split.xdmf, use 'Cell Data Array' dropdown "
      "to switch between fields.")
