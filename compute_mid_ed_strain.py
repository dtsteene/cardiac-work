#!/usr/bin/env python3
"""Mid-ring ED fibre strain via correct AHA mask (sidecar), per region.

The metrics_calculator Mid_RV / Mid_Septum strains are unreliable (gernerate_aha_biv
mis-segments on the adios checkpoint mesh → empty RV/septum mid masks → zero). This
recomputes mean ED fibre strain over the FULL region and the AHA MID ring using the
proven aha_tags.npy sidecar, so the band can be checked against preload.

For each case: load displacement at end-diastole (start of last beat), compute per-cell
E_ff (DG0) on the checkpoint mesh, match to the per_cell npz ordering by reference
centroid (same mesh, exact), then mask by region_tags and the sidecar mid ring.

Serial. Run via sbatch.  Usage: python compute_mid_ed_strain.py <case_dir> [...]
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from mpi4py import MPI
import dolfinx, ufl, basix.ufl, adios4dolfinx
import cardiac_geometries.geometry
from scipy.spatial import cKDTree

REG = {"LV": 1, "RV": 2, "Septum": 3}
NB = 6


def ed_strain(case_dir: Path):
    comm = MPI.COMM_SELF
    sp = json.load(open(case_dir / "simulation_params.json"))
    ckpt = case_dir / "solver" / "checkpoint.bp"
    mesh = adios4dolfinx.read_mesh(ckpt, comm)
    ncell = mesh.topology.index_map(3).size_local

    # fibre f0 (quadrature, deg 6 — as written by the sim)
    q = basix.ufl.quadrature_element(mesh.topology.cell_name(), value_shape=(3,), degree=6)
    Qv = dolfinx.fem.functionspace(mesh, q)
    f0 = dolfinx.fem.Function(Qv)
    adios4dolfinx.read_function(ckpt, f0, time=0.0, name="f0")

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (3,)))
    u = dolfinx.fem.Function(V)

    ts = adios4dolfinx.read_timestamps(ckpt, comm, "displacement")
    spb = int(round((1.0 / (sp["BPM"] / 60.0)) / sp["dt"]))
    t_ed = float(ts[max(0, len(ts) - spb)])   # start of last beat ≈ ED
    adios4dolfinx.read_function(ckpt, u, time=t_ed, name="displacement")

    I = ufl.Identity(3)
    F = ufl.grad(u) + I
    E = 0.5 * (F.T * F - I)
    Eff = ufl.inner(ufl.dot(E, f0), f0)
    # f0 is a quadrature element → cannot interpolate the expression into DG0.
    # Use the DG0 test-function trick with a deg-6 quadrature measure: per-cell mean
    # E_ff = (∫_cell E_ff dx) / (∫_cell dx).
    DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    v = ufl.TestFunction(DG0)
    dx_q = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 6})
    num = dolfinx.fem.assemble_vector(dolfinx.fem.form(Eff * v * dx_q)).array[:ncell]
    den = dolfinx.fem.assemble_vector(dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * v * dx_q)).array[:ncell]
    eff_arr = num / den
    centroids = dolfinx.mesh.compute_midpoints(mesh, 3, np.arange(ncell, dtype=np.int32))

    # map onto per_cell npz ordering (both are checkpoint reference centroids → exact)
    z = np.load(case_dir / "per_cell_data.npz", allow_pickle=True)
    npz_c = np.asarray(z["centroids"], float)
    tags = np.asarray(z["region_tags"]).astype(np.int32)
    aha = np.load(case_dir / "aha_tags.npy").astype(np.int32)
    _, idx = cKDTree(centroids).query(npz_c)
    eff_npz = eff_arr[idx]
    mid = np.isin(aha, [4, 5, 6])

    out = {}
    for r, tag in REG.items():
        full_m = tags == tag
        mid_m = full_m & mid
        out[f"{r}_full"] = float(eff_npz[full_m].mean()) * 100
        out[f"{r}_mid"] = float(eff_npz[mid_m].mean()) * 100 if mid_m.any() else float("nan")
        out[f"{r}_mid_n"] = int(mid_m.sum())
    sp_p = np.load(case_dir / "solver/solver_cavity_pressure_mmHg.npy").astype(float)
    out["rvsys"] = float(sp_p[:, 1][len(sp_p) - len(sp_p) // NB:].max())
    return out


def main():
    cases = [Path(p) for p in sys.argv[1:]]
    print("case        RVsys   LV_full LV_mid   RV_full RV_mid   Sep_full Sep_mid   (mid n LV/RV/Sep)")
    for cd in cases:
        o = ed_strain(cd)
        print("%-11s %5.0f   %6.2f %6.2f   %6.2f %6.2f   %6.2f %6.2f   (%d/%d/%d)" % (
            cd.name, o["rvsys"], o["LV_full"], o["LV_mid"], o["RV_full"], o["RV_mid"],
            o["Septum_full"], o["Septum_mid"], o["LV_mid_n"], o["RV_mid_n"], o["Septum_mid_n"]))


if __name__ == "__main__":
    main()
