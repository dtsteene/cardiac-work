#!/usr/bin/env python3
"""Export the LDRB AHA biventricular segmentation to a VTU for ParaView inspection.

Uses the same `ldrb.aha.gernerate_aha_biv` call as postprocess_metrics.py, so the
"Mid" ring here is exactly the Mid_LV/Mid_RV/Mid_Septum the metrics already report.
The mid ring is the middle third along the apex-to-base Laplace coordinate — away
from both the base and the apex — split into LV / RV / septum.

Writes <out>/aha_band.vtu with per-cell fields:
  aha_tag     0=Apical, 1=Basal_LV, 2=Basal_RV, 3=Basal_Septum,
                        4=Mid_LV,   5=Mid_RV,   6=Mid_Septum
  region_tag  1=LV, 2=RV, 3=Septum (LDRB markers)
  mid_band    1 if aha_tag in {4,5,6} (the canonical mid third), else 0
  apicobasal  geometric apex->base coordinate 0..1 (for a custom band: threshold this
              in ParaView, e.g. 0.3<apicobasal<0.7, if the AHA mid third is too thin/thick)

Run (serial):  python export_aha_band.py <geometry_dir> --output-dir <out>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI
import dolfinx
import dolfinx.plot
import cardiac_geometries.geometry
import ldrb.aha
from cardiac_geometries.mesh import transform_markers
import pyvista as pv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("geometry_dir", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    comm = MPI.COMM_SELF
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm, args.geometry_dir.resolve())
    mesh, ffun = geo.mesh, geo.ffun

    # AHA biventricular tags (DG0) — identical call to postprocess_metrics.py
    ldrb_markers = transform_markers(geo.markers, clipped=True)
    aha_func = ldrb.aha.gernerate_aha_biv(mesh=mesh, ffun=ffun, markers=ldrb_markers,
                                          function_space="DG_0")
    ncell = mesh.topology.index_map(3).size_local
    aha = aha_func.x.array[:ncell].astype(np.int32)

    # LV/RV/Septum region tags from the LDRB markers meshtag
    reg = np.zeros(ncell, np.int32)
    mmt = geo.additional_data.get("markers_mt")
    if mmt is not None:
        sel = mmt.indices < ncell
        reg[mmt.indices[sel]] = mmt.values[sel].astype(np.int32)

    # Proper apex->base coordinate: Laplace solve (phi=0 at the apex vertex, phi=1 on
    # the base facets). The global PCA long axis is NOT each chamber's apex-base axis on
    # a biventricle, so a geometric projection mis-bands the LV; the diffusion field is
    # balanced across chambers (this is the LDRB-style apicobasal scalar).
    import ufl
    from dolfinx.fem.petsc import LinearProblem
    tdim = mesh.topology.dim
    V1 = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    base_facets = ffun.find(geo.markers["BASE"][0])
    base_dofs = dolfinx.fem.locate_dofs_topological(V1, tdim - 1, base_facets)
    base_mid = dolfinx.mesh.compute_midpoints(mesh, tdim - 1, base_facets).mean(0)
    Xv = mesh.geometry.x
    apex_coord = Xv[np.argmax(np.linalg.norm(Xv - base_mid, axis=1))]
    span = float(np.ptp(Xv, 0).max())
    apex_dofs = dolfinx.fem.locate_dofs_geometrical(
        V1, lambda x: np.linalg.norm(x.T - apex_coord, axis=1) < 1e-3 * span)
    bcs = [dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(1.0), base_dofs, V1),
           dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), apex_dofs, V1)]
    uu, vv = ufl.TrialFunction(V1), ufl.TestFunction(V1)
    a = ufl.inner(ufl.grad(uu), ufl.grad(vv)) * ufl.dx
    Lform = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)) * vv * ufl.dx
    _sol = LinearProblem(a, Lform, bcs=bcs, petsc_options_prefix="ab_",
                         petsc_options={"ksp_type": "preonly", "pc_type": "lu"}).solve()
    phi = _sol[0] if isinstance(_sol, tuple) else _sol   # newer dolfinx may return a tuple
    phi0 = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", 0)))
    phi0.interpolate(phi)
    ab = phi0.x.array[:ncell]
    ab = np.clip((ab - ab.min()) / np.ptp(ab), 0.0, 1.0)

    topo, ctypes, pts = dolfinx.plot.vtk_mesh(mesh, 3)
    ctypes = np.full_like(ctypes, pv.CellType.TETRA, dtype=np.uint8)
    g = pv.UnstructuredGrid(topo, ctypes, pts)
    g.cell_data["aha_tag"] = aha
    g.cell_data["region_tag"] = reg
    g.cell_data["mid_band"] = np.isin(aha, [4, 5, 6]).astype(np.uint8)
    g.cell_data["apicobasal"] = ab.astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "aha_band.vtu"
    g.save(str(out))

    lab = {0: "Apical", 1: "Basal_LV", 2: "Basal_RV", 3: "Basal_Septum",
           4: "Mid_LV", 5: "Mid_RV", 6: "Mid_Septum"}
    counts = {lab[v]: int((aha == v).sum()) for v in range(7)}
    print("AHA cell counts:", counts)
    print(f"mid_band cells: {int(np.isin(aha,[4,5,6]).sum())} / {ncell}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
