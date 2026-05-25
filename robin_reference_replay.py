#!/usr/bin/env python3
"""
robin_reference_replay.py

Replay just the Robin support work using the buggy reference-configuration
formula on the saved displacement checkpoint. The production postprocessor
uses the variationally correct Nanson formulation:

    W_robin = - alpha * (u_avg . NN) * (du . NN) * cofnorm * ds_ref

This script assembles instead the simpler / wrong reference-config formula
that pre-dated the fix:

    W_robin_buggy = - alpha * inner(u_avg, du) * ds_ref

(full vector inner product, no surface mapping). The output is
metrics/robin_reference_work_per_step.npy with shape (n_steps, 2) for
[epi, base] per-step work increments in Joules. The audit script reads
this file to construct the V0-bug-3 variant.

Run with sbatch on slurm (do not run on login node — it does mesh assembly
and ADIOS reads).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import adios4dolfinx
import dolfinx
import numpy as np
import ufl
from mpi4py import MPI

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [rank %(message)s")
logger = logging.getLogger("robin_replay")


def main(case_dir: Path) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    n_ranks = comm.size

    def log(msg: str) -> None:
        if rank == 0:
            print(f"[robin_replay] {msg}", flush=True)

    log(f"Case dir: {case_dir}")
    log(f"MPI ranks: {n_ranks}")

    solver_dir = case_dir / "solver"
    checkpoint_path = solver_dir / "checkpoint.bp"
    if not checkpoint_path.exists():
        raise RuntimeError(f"Missing checkpoint: {checkpoint_path}")

    with open(case_dir / "simulation_params.json") as f:
        sim_params = json.load(f)

    alpha_epi = float(sim_params["alpha_epi"])
    alpha_base = float(sim_params["alpha_base"])
    log(f"alpha_epi = {alpha_epi}, alpha_base = {alpha_base}")

    # Load mesh and facet tags from checkpoint (same DOF ordering as displacement)
    mesh = adios4dolfinx.read_mesh(checkpoint_path, comm)
    ffun = adios4dolfinx.read_meshtags(checkpoint_path, mesh, meshtag_name="ffun")
    log(f"mesh ndofs (rank 0 view): {mesh.geometry.x.shape[0]}")

    # Marker tags for EPI and BASE — read from geometry metadata in simulation_params,
    # or fall back to scanning facet tags
    geometry_dir = sim_params.get("geometry_dir") or (case_dir / "geometry")
    geometry_dir = Path(geometry_dir)
    markers_json = geometry_dir / "markers.json"
    if markers_json.exists():
        with open(markers_json) as f:
            markers = json.load(f)
    else:
        # Last-resort fallback: assume canonical mapping
        markers = {
            "EPI":  [40, 2],
            "BASE": [50, 2],
            "LV":   [30, 2],
            "RV":   [20, 2],
        }
    tag_epi = int(markers["EPI"][0])
    tag_base = int(markers["BASE"][0])
    log(f"tag_epi = {tag_epi}, tag_base = {tag_base}")

    # Set up function space matching what the solver used (pulse uses P2 by default)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (3,)))
    u_cur = dolfinx.fem.Function(V, name="u_cur")
    u_prev = dolfinx.fem.Function(V, name="u_prev")
    log(f"P2 vector space ndofs (local): {V.dofmap.index_map.size_local}")

    # Buggy Robin forms:
    #   - reference-configuration ds (no surface mapping)
    #   - full vector inner product u_avg . du (no normal projection)
    u_avg = 0.5 * (u_cur + u_prev)
    du = u_cur - u_prev
    ds_ref = ufl.Measure("ds", domain=mesh, subdomain_data=ffun)

    form_epi_buggy = dolfinx.fem.form(
        -alpha_epi * ufl.inner(u_avg, du) * ds_ref(tag_epi))
    form_base_buggy = dolfinx.fem.form(
        -alpha_base * ufl.inner(u_avg, du) * ds_ref(tag_base))

    # Read displacement timestamps
    timestamps = adios4dolfinx.read_timestamps(checkpoint_path, comm, "displacement")
    n_steps = len(timestamps)
    log(f"Number of timesteps: {n_steps}")

    # Initialize u_prev with t=0 displacement
    adios4dolfinx.read_function(checkpoint_path, u_prev,
                                time=float(timestamps[0]), name="displacement")
    u_prev.x.scatter_forward()

    # Allocate per-step output: [epi_work, base_work] per step in Joules
    per_step = np.zeros((n_steps, 2), dtype=float)

    for step_idx in range(1, n_steps):
        adios4dolfinx.read_function(checkpoint_path, u_cur,
                                    time=float(timestamps[step_idx]),
                                    name="displacement")
        u_cur.x.scatter_forward()

        w_epi_local = dolfinx.fem.assemble_scalar(form_epi_buggy)
        w_base_local = dolfinx.fem.assemble_scalar(form_base_buggy)
        w_epi = comm.allreduce(w_epi_local, op=MPI.SUM)
        w_base = comm.allreduce(w_base_local, op=MPI.SUM)

        per_step[step_idx, 0] = w_epi
        per_step[step_idx, 1] = w_base

        # Roll: u_prev <- u_cur for next iter
        u_prev.x.array[:] = u_cur.x.array
        u_prev.x.scatter_forward()

        if rank == 0 and (step_idx % 100 == 0 or step_idx == n_steps - 1):
            cum_epi = per_step[: step_idx + 1, 0].sum()
            cum_base = per_step[: step_idx + 1, 1].sum()
            print(f"  step {step_idx:4d}/{n_steps}: cum w_epi={cum_epi:+.4e} J, "
                  f"cum w_base={cum_base:+.4e} J", flush=True)

    if rank == 0:
        out_dir = case_dir / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "robin_reference_work_per_step.npy"
        np.save(out_path, per_step)
        total_epi = per_step[:, 0].sum()
        total_base = per_step[:, 1].sum()
        print(f"\n[robin_replay] Saved {out_path}", flush=True)
        print(f"[robin_replay] Cycle-end buggy Robin work: "
              f"epi = {total_epi:+.4e} J, base = {total_base:+.4e} J, "
              f"total = {total_epi + total_base:+.4e} J", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mpirun -n N python robin_reference_replay.py <case_dir>",
              file=sys.stderr)
        sys.exit(1)
    main(Path(sys.argv[1]).resolve())
