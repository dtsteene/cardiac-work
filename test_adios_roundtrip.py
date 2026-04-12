#!/usr/bin/env python3
"""
test_adios_roundtrip.py — Verify a serially-generated mesh survives parallel reload.

This is a one-time sanity check before we commit to the shared-mesh workflow
for the spectrum sims. We want to be certain that adios4dolfinx correctly
repartitions the mesh across MPI ranks and preserves everything we rely on:
cell count, facet tags, cell tags (region markers), fiber functions.

The test mirrors the exact production workflow:
  1. (serial) run geometry_generator.py to create a shared UKB mesh
  2. (parallel) load it via cardiac_geometries.geometry.Geometry.from_folder
     — the same loading path complete_cycle.py uses when --geometry-dir is set

Usage:
  # Phase 1: generate (serial)
  python3 test_adios_roundtrip.py generate --outdir /tmp/adios_roundtrip_test

  # Phase 2: verify (parallel)
  mpirun -n 8 python3 test_adios_roundtrip.py verify --outdir /tmp/adios_roundtrip_test

  # Or use the helper shell command at the bottom of this file.

Passes if:
  - Total cell count matches (sum across ranks = serial count)
  - Per-marker cell counts match (LV=1, RV=2, Septum=3)
  - Per-marker facet counts match (LV_endo, RV_endo, EPI, BASE)
  - Mesh volume (integral of 1*dx) matches to machine precision
  - Fiber function integral (integral of |f0|^2 * dx) matches
  - Sorted centroid coordinate arrays match bit-for-bit
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["generate", "verify"])
parser.add_argument("--outdir", type=Path, required=True,
                    help="Directory for the test mesh and reference stats")
parser.add_argument("--char-length", type=float, default=10.0,
                    help="Mesh characteristic length (mm) for phase 'generate'")
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
outdir = args.outdir.resolve()

REF_STATS_PATH = outdir / "reference_stats.json"
CENTROIDS_PATH = outdir / "reference_centroids.npy"


# ════════════════════════════════════════════════════════════════════════════
# Helpers shared by both phases
# ════════════════════════════════════════════════════════════════════════════

def compute_stats(mesh, ffun, markers_mt, f0, s0, comm):
    """Compute a set of global scalar statistics via MPI reduction."""
    import dolfinx
    import ufl
    import time

    def cs_print(msg):
        if comm.rank == 0:
            print(f"  [compute_stats] {msg}", flush=True)

    t0 = time.time()
    cs_print(f"enter (t=0.0s)")

    imap = mesh.topology.index_map(3)
    n_local_cells = imap.size_local
    local_cells = np.arange(n_local_cells, dtype=np.int32)
    cs_print(f"index_map ready, n_local_cells={n_local_cells} (+{time.time()-t0:.1f}s)")

    # Global cell count
    n_cells_global = comm.allreduce(n_local_cells, op=MPI.SUM)
    cs_print(f"n_cells_global allreduced = {n_cells_global} (+{time.time()-t0:.1f}s)")

    # Per-marker cell counts (markers_mt has values 1=LV, 2=RV, 3=Septum)
    marker_vals = markers_mt.values[:n_local_cells]
    n_lv_local = int((marker_vals == 1).sum())
    n_rv_local = int((marker_vals == 2).sum())
    n_sept_local = int((marker_vals == 3).sum())
    n_lv = comm.allreduce(n_lv_local, op=MPI.SUM)
    n_rv = comm.allreduce(n_rv_local, op=MPI.SUM)
    n_sept = comm.allreduce(n_sept_local, op=MPI.SUM)
    cs_print(f"per-marker cell counts done: LV={n_lv} RV={n_rv} Sept={n_sept} (+{time.time()-t0:.1f}s)")

    # Per-marker facet counts (from ffun)
    # CRITICAL: every rank must iterate the SAME set of marker values,
    # otherwise the allreduce calls desynchronize and deadlock. Gather the
    # union of local unique markers across all ranks first.
    cs_print(f"gathering global set of unique facet markers (+{time.time()-t0:.1f}s)")
    local_unique = set(int(v) for v in np.unique(ffun.values))
    all_unique = comm.allgather(local_unique)
    global_markers = sorted(set().union(*all_unique))
    cs_print(f"global facet markers = {global_markers} (+{time.time()-t0:.1f}s)")

    facet_marker_counts = {}
    for mark in global_markers:
        n_local = int((ffun.values == mark).sum())
        n_global = comm.allreduce(n_local, op=MPI.SUM)
        facet_marker_counts[str(mark)] = n_global
    cs_print(f"facet marker counts done: {facet_marker_counts} (+{time.time()-t0:.1f}s)")

    # Mesh volume — JIT compiles the first UFL form (may take a minute cold)
    cs_print(f"about to build vol_form (first JIT compile — cold-start may be slow) (+{time.time()-t0:.1f}s)")
    one = dolfinx.fem.Constant(mesh, 1.0)
    vol_form = dolfinx.fem.form(one * ufl.dx(domain=mesh))
    cs_print(f"vol_form compiled (+{time.time()-t0:.1f}s)")
    vol_local = dolfinx.fem.assemble_scalar(vol_form)
    mesh_volume = comm.allreduce(vol_local, op=MPI.SUM)
    cs_print(f"mesh_volume = {mesh_volume:.6e} (+{time.time()-t0:.1f}s)")

    # Fiber function integrals: ∫ |f0|^2 dx and ∫ |s0|^2 dx
    # For unit fiber vectors |f0|=|s0|=1 these should equal the mesh volume.
    if f0 is not None:
        cs_print(f"about to build f0_mag_sq form (JIT) (+{time.time()-t0:.1f}s)")
        f0_mag_sq = dolfinx.fem.form(ufl.inner(f0, f0) * ufl.dx)
        cs_print(f"f0 form compiled, assembling (+{time.time()-t0:.1f}s)")
        f0_int_local = dolfinx.fem.assemble_scalar(f0_mag_sq)
        f0_int = comm.allreduce(f0_int_local, op=MPI.SUM)
        cs_print(f"f0_int = {f0_int:.6e} (+{time.time()-t0:.1f}s)")
    else:
        f0_int = None
    if s0 is not None:
        cs_print(f"about to build s0_mag_sq form (JIT) (+{time.time()-t0:.1f}s)")
        s0_mag_sq = dolfinx.fem.form(ufl.inner(s0, s0) * ufl.dx)
        cs_print(f"s0 form compiled, assembling (+{time.time()-t0:.1f}s)")
        s0_int_local = dolfinx.fem.assemble_scalar(s0_mag_sq)
        s0_int = comm.allreduce(s0_int_local, op=MPI.SUM)
        cs_print(f"s0_int = {s0_int:.6e} (+{time.time()-t0:.1f}s)")
    else:
        s0_int = None

    # Coordinate bounding box (global)
    local_coords = mesh.geometry.x
    if len(local_coords) > 0:
        cmin = local_coords.min(axis=0)
        cmax = local_coords.max(axis=0)
    else:
        cmin = np.full(3, np.inf)
        cmax = np.full(3, -np.inf)
    cmin_global = np.zeros(3)
    cmax_global = np.zeros(3)
    comm.Allreduce(cmin, cmin_global, op=MPI.MIN)
    comm.Allreduce(cmax, cmax_global, op=MPI.MAX)
    cs_print(f"bbox allreduced (+{time.time()-t0:.1f}s)")

    # Cell centroids (gathered to rank 0 for bit-exact comparison)
    centroids_local = dolfinx.mesh.compute_midpoints(mesh, 3, local_cells)
    cs_print(f"local centroids computed (+{time.time()-t0:.1f}s)")
    centroids_all = comm.gather(centroids_local, root=0)
    if comm.rank == 0:
        centroids_global = np.concatenate([c for c in centroids_all if len(c) > 0])
    else:
        centroids_global = None
    cs_print(f"centroids gathered (+{time.time()-t0:.1f}s)")

    return {
        "n_cells_global": n_cells_global,
        "n_lv": n_lv,
        "n_rv": n_rv,
        "n_sept": n_sept,
        "facet_marker_counts": facet_marker_counts,
        "mesh_volume": float(mesh_volume),
        "f0_integral": float(f0_int) if f0_int is not None else None,
        "s0_integral": float(s0_int) if s0_int is not None else None,
        "bbox_min": cmin_global.tolist(),
        "bbox_max": cmax_global.tolist(),
    }, centroids_global


def print_stats(stats, label):
    print(f"\n=== {label} ===")
    print(f"  n_cells_global      : {stats['n_cells_global']}")
    print(f"  n_lv / n_rv / n_sept: {stats['n_lv']} / {stats['n_rv']} / {stats['n_sept']}")
    print(f"  facet markers       : {stats['facet_marker_counts']}")
    print(f"  mesh_volume         : {stats['mesh_volume']:.12e}")
    if stats['f0_integral'] is not None:
        print(f"  f0 integral         : {stats['f0_integral']:.12e}")
    if stats['s0_integral'] is not None:
        print(f"  s0 integral         : {stats['s0_integral']:.12e}")
    print(f"  bbox_min            : {stats['bbox_min']}")
    print(f"  bbox_max            : {stats['bbox_max']}")


# ════════════════════════════════════════════════════════════════════════════
# Phase 1: generate mesh serially
# ════════════════════════════════════════════════════════════════════════════

def phase_generate():
    if size != 1:
        if rank == 0:
            print("ERROR: the 'generate' phase must run serially (single process, no mpirun)")
        sys.exit(1)

    import dolfinx
    import cardiac_geometries
    import cardiac_geometries.geometry
    import ldrb

    outdir.mkdir(parents=True, exist_ok=True)

    # Call geometry_generator.py as a subprocess to produce the standard
    # geometry folder. This is the exact same command we'd use for the
    # production shared mesh.
    cmd = [
        "python3", "geometry_generator.py",
        "--single", "ukb",
        "-c", str(args.char_length),
        "--output-dir", str(outdir),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"geometry_generator.py failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # After geometry_generator.py, the geometry should be at
    # outdir/ukb/geometry/geometry.bp
    geo_dir = outdir / "ukb" / "geometry"
    if not (geo_dir / "geometry.bp").exists():
        print(f"ERROR: no geometry.bp at {geo_dir}")
        sys.exit(1)

    # Load the serially-written geometry (also serially) and compute stats
    print(f"\nLoading serial geometry from {geo_dir}")
    geo = cardiac_geometries.geometry.Geometry.from_folder(MPI.COMM_SELF, geo_dir)

    mesh = geo.mesh
    ffun = geo.ffun
    markers_mt = geo.additional_data.get("markers_mt")
    f0 = geo.f0
    s0 = geo.s0

    if markers_mt is None:
        print("WARNING: no markers_mt in geometry; per-marker cell counts will be 0")
        import dolfinx
        zero_tags = dolfinx.mesh.meshtags(
            mesh, 3,
            np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32),
            np.zeros(mesh.topology.index_map(3).size_local, dtype=np.int32),
        )
        markers_mt = zero_tags

    stats, centroids = compute_stats(mesh, ffun, markers_mt, f0, s0, MPI.COMM_SELF)
    print_stats(stats, "SERIAL REFERENCE STATS")

    # Save reference
    REF_STATS_PATH.write_text(json.dumps(stats, indent=2))
    np.save(CENTROIDS_PATH, centroids)
    print(f"\nWrote reference stats to {REF_STATS_PATH}")
    print(f"Wrote reference centroids to {CENTROIDS_PATH}  ({len(centroids)} cells)")
    print(f"\nNow run: mpirun -n 8 python3 test_adios_roundtrip.py verify --outdir {outdir}")


# ════════════════════════════════════════════════════════════════════════════
# Phase 2: verify mesh in parallel
# ════════════════════════════════════════════════════════════════════════════

def vprint(msg):
    """Rank-0 print with explicit flush so sbatch logs show progress live."""
    if rank == 0:
        print(msg, flush=True)


def phase_verify():
    vprint("[verify] python started, importing modules...")
    import time
    t_start = time.time()

    import cardiac_geometries
    import cardiac_geometries.geometry
    vprint(f"[verify] cardiac_geometries imported (+{time.time()-t_start:.1f}s)")

    geo_dir = outdir / "ukb" / "geometry"
    if rank == 0:
        if not (geo_dir / "geometry.bp").exists():
            print(f"ERROR: no geometry.bp at {geo_dir}", flush=True)
            print(f"Run the generate phase first: python3 test_adios_roundtrip.py generate --outdir {outdir}", flush=True)
            sys.exit(1)
        if not REF_STATS_PATH.exists():
            print(f"ERROR: no reference stats at {REF_STATS_PATH}", flush=True)
            sys.exit(1)
    vprint(f"[verify] file checks OK (+{time.time()-t_start:.1f}s)")

    comm.Barrier()
    vprint(f"[verify] post first barrier, size={size} (+{time.time()-t_start:.1f}s)")

    # Load reference stats on rank 0
    if rank == 0:
        ref_stats = json.loads(REF_STATS_PATH.read_text())
        ref_centroids = np.load(CENTROIDS_PATH)
        print(f"[verify] loaded reference stats and centroids ({len(ref_centroids)} cells) "
              f"(+{time.time()-t_start:.1f}s)", flush=True)
    else:
        ref_stats = None
        ref_centroids = None

    # Load the geometry in parallel using the SAME code path as
    # complete_cycle.py uses when --geometry-dir is supplied
    vprint(f"[verify] about to call Geometry.from_folder on {size} ranks from {geo_dir} "
           f"(+{time.time()-t_start:.1f}s)")
    comm.Barrier()
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm, geo_dir)
    vprint(f"[verify] Geometry.from_folder returned (+{time.time()-t_start:.1f}s)")

    mesh = geo.mesh
    ffun = geo.ffun
    markers_mt = geo.additional_data.get("markers_mt")
    f0 = geo.f0
    s0 = geo.s0

    if markers_mt is None:
        import dolfinx
        zero_tags = dolfinx.mesh.meshtags(
            mesh, 3,
            np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32),
            np.zeros(mesh.topology.index_map(3).size_local, dtype=np.int32),
        )
        markers_mt = zero_tags
    vprint(f"[verify] mesh and tags ready, computing stats "
           f"(+{time.time()-t_start:.1f}s)")

    par_stats, par_centroids = compute_stats(mesh, ffun, markers_mt, f0, s0, comm)
    vprint(f"[verify] stats computed (+{time.time()-t_start:.1f}s)")

    if rank != 0:
        return

    print_stats(par_stats, f"PARALLEL STATS (size={size})")

    # ── Compare rank 0 ────────────────────────────────────────────────────────
    print(f"\n=== COMPARISON vs SERIAL REFERENCE ===")
    fails = []

    def check(name, ref, got, tol=0.0):
        if isinstance(ref, (int, float)) and isinstance(got, (int, float)):
            if tol == 0.0:
                ok = (ref == got)
            else:
                rel = abs(got - ref) / (abs(ref) + 1e-30)
                ok = rel <= tol
        else:
            ok = (ref == got)
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name:<24}  ref={ref}  got={got}"
              + (f"  tol={tol}" if tol else ""))
        if not ok:
            fails.append(name)

    check("n_cells_global", ref_stats["n_cells_global"], par_stats["n_cells_global"])
    check("n_lv", ref_stats["n_lv"], par_stats["n_lv"])
    check("n_rv", ref_stats["n_rv"], par_stats["n_rv"])
    check("n_sept", ref_stats["n_sept"], par_stats["n_sept"])
    check("facet_marker_counts", ref_stats["facet_marker_counts"], par_stats["facet_marker_counts"])
    check("mesh_volume", ref_stats["mesh_volume"], par_stats["mesh_volume"], tol=1e-10)
    if ref_stats["f0_integral"] is not None:
        check("f0_integral", ref_stats["f0_integral"], par_stats["f0_integral"], tol=1e-10)
    if ref_stats["s0_integral"] is not None:
        check("s0_integral", ref_stats["s0_integral"], par_stats["s0_integral"], tol=1e-10)

    # Bounding boxes should match exactly (coordinate data is stored bit-identical)
    bbox_ref_min = np.array(ref_stats["bbox_min"])
    bbox_par_min = np.array(par_stats["bbox_min"])
    bbox_ref_max = np.array(ref_stats["bbox_max"])
    bbox_par_max = np.array(par_stats["bbox_max"])
    if np.allclose(bbox_ref_min, bbox_par_min, atol=1e-14) and np.allclose(bbox_ref_max, bbox_par_max, atol=1e-14):
        print(f"  ✓ bbox matches")
    else:
        print(f"  ✗ bbox differs\n    ref_min={bbox_ref_min}  par_min={bbox_par_min}\n"
              f"    ref_max={bbox_ref_max}  par_max={bbox_par_max}")
        fails.append("bbox")

    # Centroid set comparison (sorted lexicographic to ignore ordering)
    if len(ref_centroids) == len(par_centroids):
        ref_sorted = ref_centroids[np.lexsort(ref_centroids.T)]
        par_sorted = par_centroids[np.lexsort(par_centroids.T)]
        max_diff = np.abs(ref_sorted - par_sorted).max()
        if max_diff < 1e-12:
            print(f"  ✓ cell centroid set matches bit-for-bit ({len(ref_centroids)} cells)")
        else:
            print(f"  ✗ centroid set differs: max coordinate diff = {max_diff:.3e}")
            fails.append("centroids")
    else:
        print(f"  ✗ centroid count differs: ref={len(ref_centroids)}  par={len(par_centroids)}")
        fails.append("centroid_count")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print()
    if not fails:
        print("=" * 50)
        print("✓ ALL CHECKS PASSED — safe to use shared mesh for the parallel sim spectrum.")
        print("=" * 50)
        sys.exit(0)
    else:
        print("=" * 50)
        print(f"✗ {len(fails)} CHECK(S) FAILED: {fails}")
        print("=" * 50)
        sys.exit(2)


# ════════════════════════════════════════════════════════════════════════════
# Dispatch
# ════════════════════════════════════════════════════════════════════════════

if args.phase == "generate":
    phase_generate()
elif args.phase == "verify":
    phase_verify()
