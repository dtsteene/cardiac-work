#!/usr/bin/env python3
"""
verify_tau_coordinates.py — visual ParaView comparison of Euclidean vs Laplace tau.

REWRITTEN 2026-04-14: the previous version loaded tau / d_lv / d_rv from
`per_cell_data.npz` and tried to paste those arrays onto a serially-loaded
mesh. That is wrong: compute_per_cell.py saves arrays in
parallel-rank-concatenated order (rank0_cells then rank1_cells then ...),
which does NOT match the serial cell ordering from
`adios4dolfinx.read_mesh(..., MPI.COMM_SELF)`. The result was a completely
scrambled XDMF.

This version computes tau_E, tau_L and all the per-cell fields DIRECTLY on
the serially-loaded mesh (the same pattern used by verify_sweep_envelope.py).
It depends on nothing from per_cell_data.npz, and so is immune to the
parallel/serial cell-ordering mismatch.

NOTE on mesh state: this script computes distances on the mesh's **unloaded
reference configuration** (the coordinates stored in solver/checkpoint.bp,
before any solver displacement is applied). That matches how canonical-mode
tagging works, and it is sufficient for the Eucl-vs-Laplace side comparison:
the Laplace bowing / non-local contamination at the septal junctions is
present in both the unloaded and the ED-deformed state.

For each sim dir, writes tau_verification/tau_check.xdmf containing:
  - tau_E:              Euclidean tau = d_LV / (d_LV + d_RV)        (LV=0, RV=1)
  - tau_L:              Laplace  tau  = 1 - u_LVRV                  (LV=0, RV=1)
  - tau_E_septum / tau_L_septum: same fields with a large sentinel outside
                        the geometric septum so you can threshold to see a
                        LV->RV sweep in the septum only
  - tau_diff_E_minus_L: signed disagreement per cell
  - side_agreement:     categorical — which cells Eucl and Lap disagree on
                        re: LV side vs RV side
  - is_geometric_septum
  - d_LV_mm, d_RV_mm, d_sum_mm
  - lv_rv_scalar, epi_scalar (raw Laplace fields, for debugging)

Usage:
    python3 verify_tau_coordinates.py <sim_dir1> [<sim_dir2> ...]

    # Healthy vs severe vs end_stage contrast:
    python3 verify_tau_coordinates.py \\
        results/sims/2026-04-08/UKB_6beats_run_1017516 \\
        results/sims/2026-04-08/UKB_6beats_run_1017521 \\
        results/sims/2026-04-08/UKB_6beats_run_1017523

In ParaView, color by 'side_agreement' with a categorical colormap:
    0 = not in septum      (grey/hidden)
    1 = both LV            (green)  Eucl and Lap agree it's on the LV side
    2 = both RV            (blue)   Eucl and Lap agree it's on the RV side
    3 = Eucl LV, Lap RV    (red)    Laplace sent this cell wrongly to RV side
    4 = Eucl RV, Lap LV    (orange) Laplace sent this cell wrongly to LV side
Red and orange cells are the Laplace mis-assignments.
"""
import argparse
from pathlib import Path

import numpy as np
import dolfinx
import dolfinx.fem.petsc
import ufl
import adios4dolfinx
import pyvista as pv
from mpi4py import MPI
from petsc4py import PETSc


# Facet markers in the cardiac_geometries convention (checked empirically
# against geo.markers in compute_per_cell.py: {'LV':[1,2], 'RV':[2,2],
# 'EPI':[3,2], 'BASE':[4,2]}). First element is the marker ID; same as what
# verify_sweep_envelope.py uses.
LV_MARKER, RV_MARKER, EPI_MARKER = 1, 2, 3


def build_surface_polydata(mesh, ffun, f2v, marker):
    """Extract a pyvista PolyData triangle mesh for the given facet marker."""
    facets = ffun.find(marker)
    tri_verts = []
    for f_idx in facets:
        v = f2v.links(f_idx)
        if len(v) == 3:
            tri_verts.append(v)
    if not tri_verts:
        return None
    tri_verts = np.array(tri_verts)
    # Build per-face explicit vertex array (each facet independent, duplicated vertices)
    pts = mesh.geometry.x[tri_verts.flatten()]
    n_tri = len(tri_verts)
    faces = np.zeros(n_tri * 4, dtype=np.int64)
    for i in range(n_tri):
        faces[i*4]     = 3
        faces[i*4 + 1] = i*3
        faces[i*4 + 2] = i*3 + 1
        faces[i*4 + 3] = i*3 + 2
    return pv.PolyData(pts, faces=faces)


def solve_laplace_dirichlet(mesh, dirichlet_specs):
    """Solve Laplace(u) = 0 with the given (facet_marker, value) Dirichlet BCs.

    Returns a CG1 Function u on `mesh`.
    """
    V = dolfinx.fem.functionspace(mesh, ("CG", 1))
    u_tr = ufl.TrialFunction(V)
    v_t  = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u_tr), ufl.grad(v_t)) * ufl.dx
    L = dolfinx.fem.Constant(mesh, 0.0) * v_t * ufl.dx
    bcs = []
    for ffun, marker, value in dirichlet_specs:
        facets = ffun.find(marker)
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
        bcs.append(dolfinx.fem.dirichletbc(PETSc.ScalarType(value), dofs, V))
    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix=f"tau_laplace_",
    )
    return problem.solve()


def process_case(sim_dir: Path):
    sim_dir = sim_dir.resolve()
    ckpt_path = sim_dir / "solver" / "checkpoint.bp"
    if not ckpt_path.exists():
        print(f"SKIP {sim_dir.name}: no solver/checkpoint.bp")
        return

    print(f"\nProcessing {sim_dir.name}")
    print(f"  Loading mesh (SERIAL) from {ckpt_path}")
    comm = MPI.COMM_SELF

    # Load mesh + facet tags in SERIAL.
    # (The same mesh topology that the solver and compute_per_cell.py see,
    # but in a single-process cell ordering — self-consistent for this script.)
    mesh = adios4dolfinx.read_mesh(ckpt_path, comm)
    ffun = adios4dolfinx.read_meshtags(ckpt_path, mesh, meshtag_name="ffun")

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    mesh.topology.create_connectivity(2, 0)
    mesh.topology.create_connectivity(3, 2)

    n_cells = mesh.topology.index_map(3).size_local
    print(f"  Cells: {n_cells}")
    print(f"  Facet markers found: {sorted(set(int(v) for v in ffun.values))}")

    f2v = mesh.topology.connectivity(2, 0)
    c2f = mesh.topology.connectivity(3, 2)

    # ── Surfaces ──────────────────────────────────────────────────────────
    lv_poly  = build_surface_polydata(mesh, ffun, f2v, LV_MARKER)
    rv_poly  = build_surface_polydata(mesh, ffun, f2v, RV_MARKER)
    epi_poly = build_surface_polydata(mesh, ffun, f2v, EPI_MARKER)
    if None in (lv_poly, rv_poly, epi_poly):
        print(f"  !! Missing one of LV/RV/EPI surfaces — skipping")
        return
    print(f"  Surface triangles: "
          f"LV={lv_poly.n_faces_strict}, "
          f"RV={rv_poly.n_faces_strict}, "
          f"EPI={epi_poly.n_faces_strict}")

    # ── Cell centroids in mesh units (solver checkpoint uses meters) ─────
    cells = np.arange(n_cells, dtype=np.int32)
    centroids = dolfinx.mesh.compute_midpoints(mesh, 3, cells)

    bbox = mesh.geometry.x.max(axis=0) - mesh.geometry.x.min(axis=0)
    mesh_to_mm = 1.0 if bbox.max() > 10 else 1000.0
    print(f"  Mesh bbox: {bbox[0]:.2f} x {bbox[1]:.2f} x {bbox[2]:.2f} "
          f"(unit = {'mm' if mesh_to_mm == 1.0 else 'm'})")

    # ── Distances ────────────────────────────────────────────────────────
    centroids_poly = pv.PolyData(centroids.astype(np.float64))
    d_lv  = np.abs(centroids_poly.compute_implicit_distance(lv_poly )["implicit_distance"]) * mesh_to_mm
    d_rv  = np.abs(centroids_poly.compute_implicit_distance(rv_poly )["implicit_distance"]) * mesh_to_mm
    d_epi = np.abs(centroids_poly.compute_implicit_distance(epi_poly)["implicit_distance"]) * mesh_to_mm
    d_sum = d_lv + d_rv

    # ── Euclidean tau ────────────────────────────────────────────────────
    tau_E = d_lv / np.maximum(d_sum, 1e-12)

    # ── Laplace fields ───────────────────────────────────────────────────
    print("  Solving Laplace equations...")
    u_lvrv_cg = solve_laplace_dirichlet(
        mesh, [(ffun, LV_MARKER, 1.0), (ffun, RV_MARKER, 0.0)]
    )
    u_epi_cg = solve_laplace_dirichlet(
        mesh, [(ffun, EPI_MARKER, 1.0),
               (ffun, LV_MARKER,  0.0),
               (ffun, RV_MARKER,  0.0)]
    )

    # CG1 -> DG0 per cell
    V_DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    lvrv_dg0 = dolfinx.fem.Function(V_DG0)
    lvrv_dg0.interpolate(u_lvrv_cg)
    lvrv_vals = lvrv_dg0.x.array[:n_cells].copy()

    epi_dg0 = dolfinx.fem.Function(V_DG0)
    epi_dg0.interpolate(u_epi_cg)
    epi_vals = epi_dg0.x.array[:n_cells].copy()

    # Laplace tau, flipped so it goes LV->RV in the same direction as tau_E
    tau_L = 1.0 - lvrv_vals

    # ── Septum mask: geometric + topological epi exclusion ──────────────
    # The geometric-septum definition is max(d_LV, d_RV) < d_epi.
    # Exclude cells directly touching the epicardium (topological).
    epi_facets_set = set(ffun.find(EPI_MARKER).tolist())
    touches_epi = np.zeros(n_cells, dtype=bool)
    for ci in range(n_cells):
        for fi in c2f.links(ci):
            if fi in epi_facets_set:
                touches_epi[ci] = True
                break
    is_geometric_septum = (np.maximum(d_lv, d_rv) < d_epi) & ~touches_epi

    # ── Side-agreement categorical field ────────────────────────────────
    # 0 = outside septum
    # 1 = both agree LV (tau_E < 0.5 AND tau_L < 0.5)
    # 2 = both agree RV (tau_E > 0.5 AND tau_L > 0.5)
    # 3 = Eucl LV, Lap RV  (Laplace pushed cell to wrong side: toward RV)
    # 4 = Eucl RV, Lap LV  (Laplace pushed cell to wrong side: toward LV)
    side = np.zeros(n_cells, dtype=np.float64)
    E_lv = tau_E < 0.5
    E_rv = tau_E > 0.5
    L_lv = tau_L < 0.5
    L_rv = tau_L > 0.5
    side[is_geometric_septum & E_lv & L_lv] = 1.0
    side[is_geometric_septum & E_rv & L_rv] = 2.0
    side[is_geometric_septum & E_lv & L_rv] = 3.0
    side[is_geometric_septum & E_rv & L_lv] = 4.0

    n_sept        = int(is_geometric_septum.sum())
    n_both_lv     = int((side == 1).sum())
    n_both_rv     = int((side == 2).sum())
    n_disag_to_rv = int((side == 3).sum())
    n_disag_to_lv = int((side == 4).sum())
    n_disag       = n_disag_to_rv + n_disag_to_lv
    frac_disag    = n_disag / max(1, n_sept)

    print(f"  Septum cells:                    {n_sept}")
    print(f"    both agree LV (green):         {n_both_lv}")
    print(f"    both agree RV (blue):          {n_both_rv}")
    print(f"    Eucl LV, Lap RV (red):         {n_disag_to_rv}")
    print(f"    Eucl RV, Lap LV (orange):      {n_disag_to_lv}")
    print(f"    TOTAL disagreements:           {n_disag}  ({100*frac_disag:.1f}% of septum)")

    # ── Sentineled septum-only versions of tau for ParaView Threshold ───
    SENT = 1e6
    tau_E_septum = np.where(is_geometric_septum, tau_E, SENT)
    tau_L_septum = np.where(is_geometric_septum, tau_L, SENT)

    tau_diff = tau_E - tau_L

    # ── Write XDMF ───────────────────────────────────────────────────────
    def mkf(name, arr):
        f = dolfinx.fem.Function(V_DG0, name=name)
        f.x.array[:n_cells] = arr.astype(np.float64)
        return f

    fields = [
        mkf("tau_E",                tau_E),
        mkf("tau_L",                tau_L),
        mkf("tau_E_septum",         tau_E_septum),
        mkf("tau_L_septum",         tau_L_septum),
        mkf("tau_diff_E_minus_L",   tau_diff),
        mkf("tau_diff_abs",         np.abs(tau_diff)),
        mkf("side_agreement",       side),
        mkf("is_geometric_septum",  is_geometric_septum.astype(np.float64)),
        mkf("d_LV_mm",              d_lv),
        mkf("d_RV_mm",              d_rv),
        mkf("d_sum_mm",             d_sum),
        mkf("d_epi_mm",             d_epi),
        mkf("lv_rv_scalar",         lvrv_vals),
        mkf("epi_scalar",           epi_vals),
    ]

    outdir = sim_dir / "tau_verification"
    outdir.mkdir(exist_ok=True)
    outfile = outdir / "tau_check.xdmf"

    with dolfinx.io.XDMFFile(comm, str(outfile), "w") as xf:
        xf.write_mesh(mesh)
        for f in fields:
            xf.write_function(f)

    print(f"  Wrote {outfile}")

    return {
        "name": sim_dir.name,
        "n_septum": n_sept,
        "n_disagree": n_disag,
        "frac_disagree": frac_disag,
        "both_lv": n_both_lv,
        "both_rv": n_both_rv,
        "disag_to_rv": n_disag_to_rv,
        "disag_to_lv": n_disag_to_lv,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sim_dirs", nargs="+", type=Path,
                        help="Simulation result dirs (each must contain solver/checkpoint.bp)")
    args = parser.parse_args()

    results = []
    for sd in args.sim_dirs:
        r = process_case(sd)
        if r is not None:
            results.append(r)

    # Summary table
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY — fraction of septum cells mis-sided by Laplace vs Euclidean")
        print("=" * 70)
        print(f"{'case':30s}  {'n_sept':>8s}  {'n_disag':>8s}  {'%':>6s}  "
              f"{'E_LV/L_RV':>10s}  {'E_RV/L_LV':>10s}")
        for r in results:
            print(f"{r['name']:30s}  {r['n_septum']:>8d}  {r['n_disagree']:>8d}  "
                  f"{100*r['frac_disagree']:>5.1f}%  "
                  f"{r['disag_to_rv']:>10d}  {r['disag_to_lv']:>10d}")

    print("\n" + "=" * 70)
    print("In ParaView:")
    print("  1. File -> Open -> tau_check.xdmf, Apply")
    print("  2. Optional: Clip to see inside the heart")
    print()
    print("  VIEW A) Side disagreement (THE key view)")
    print("    Color by 'side_agreement' with a categorical / step colormap:")
    print("      0 = not in septum    (grey/hidden)")
    print("      1 = both LV          (green)")
    print("      2 = both RV          (blue)")
    print("      3 = Eucl LV, Lap RV  (red)    <- Laplace mis-sent to RV")
    print("      4 = Eucl RV, Lap LV  (orange) <- Laplace mis-sent to LV")
    print()
    print("  VIEW B) Signed disagreement magnitude")
    print("    Color by 'tau_diff_E_minus_L' with a diverging colormap around 0")
    print()
    print("  VIEW C) Side-by-side LV->RV sweeps")
    print("    Threshold 1: array='tau_E_septum', upper slider 0.0 -> 1.0")
    print("    Threshold 2: array='tau_L_septum', upper slider 0.0 -> 1.0")
    print("    At matched slider values the two should pick the SAME layer.")
    print()
    print("  VIEW D) Physical sanity: color by 'd_LV_mm' to confirm Euclidean")
    print("    tau really does track distance from the LV endo cleanly.")


if __name__ == "__main__":
    main()
