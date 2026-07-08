#!/bin/bash

# Recover failed post-processing from a completed simulation directory.
#
# Intended for runs where the coupled FEM/circulation simulation finished, but
# offline post-processing failed due to transient shared-filesystem I/O errors.
# The script copies only the checkpoint/history inputs needed for post-processing
# to node-local scratch, runs the replay there, and syncs the derived artifacts
# back to the original results directory.

#SBATCH --job-name=mc_repost
#SBATCH --partition=genoaxq
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

SIM_DIR=${SIM_DIR:-"/home/dtsteene/D1/cardiac-work"}
RESULTS_DIR=${RESULTS_DIR:?"Set RESULTS_DIR to the simulation results directory"}
GEOMETRY_FIELDS=${GEOMETRY_FIELDS:-""}
WORK_ROOT=${WORK_ROOT:-"/work/$USER"}

RESULTS_DIR=$(readlink -f "$RESULTS_DIR")
WORK_DIR="${WORK_ROOT}/repost_${SLURM_JOB_ID}"
SCRATCH_RUN="${WORK_DIR}/run"

mkdir -p "$SCRATCH_RUN" "$RESULTS_DIR"

sync_back() {
    set +e
    if [ -d "$SCRATCH_RUN" ]; then
        rsync -a "$SCRATCH_RUN"/compute_per_cell_recovery.* "$RESULTS_DIR"/ 2>/dev/null || true
        rsync -a "$SCRATCH_RUN"/postprocess_metrics_recovery.* "$RESULTS_DIR"/ 2>/dev/null || true
        rsync -a "$SCRATCH_RUN"/run_postprocessing_recovery.* "$RESULTS_DIR"/ 2>/dev/null || true
        rsync -a "$SCRATCH_RUN"/per_cell_data*.npz "$RESULTS_DIR"/ 2>/dev/null || true
        rsync -a "$SCRATCH_RUN"/metrics "$RESULTS_DIR"/ 2>/dev/null || true
        rsync -a "$SCRATCH_RUN"/analysis "$RESULTS_DIR"/ 2>/dev/null || true
    fi
    rm -rf "$WORK_DIR" 2>/dev/null || true
}
trap sync_back EXIT

echo "--- Repostprocess recovery started: ${SLURM_JOB_ID} ---"
date
echo "Results dir: $RESULTS_DIR"
echo "Scratch dir: $SCRATCH_RUN"

module load anaconda3
set +u
eval "$(conda shell.bash hook)"
conda activate "${CARDIAC_CONDA_ENV:-/home/dtsteene/.conda/envs/RV}"
set -u

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork
export XDG_CACHE_HOME="${WORK_DIR}/.cache"

echo "--- Copying replay inputs to scratch ---"
mkdir -p "$SCRATCH_RUN/solver" "$SCRATCH_RUN/geometry"

for name in \
    simulation_params.json parameters.json initial_conditions.json \
    history.npy ode_state_history.npy results_state.txt results_var.txt \
    time.txt state.npy state.txt state_names.txt var_names.txt run_description.txt
do
    if [ -e "${RESULTS_DIR}/${name}" ]; then
        rsync -a "${RESULTS_DIR}/${name}" "$SCRATCH_RUN/"
    fi
done

rsync -a "${RESULTS_DIR}/solver/" "$SCRATCH_RUN/solver/"

if [ -e "${RESULTS_DIR}/geometry" ]; then
    rsync -aL "${RESULTS_DIR}/geometry/" "$SCRATCH_RUN/geometry/"
else
    GEOMETRY_DIR=$("$CONDA_PREFIX/bin/python" - <<'PY' "$RESULTS_DIR"
import json
import sys
from pathlib import Path

params = Path(sys.argv[1]) / "simulation_params.json"
if params.exists():
    print(json.load(open(params)).get("geometry_dir", ""))
PY
)
    if [ -n "$GEOMETRY_DIR" ] && [ -d "$GEOMETRY_DIR" ]; then
        rsync -aL "${GEOMETRY_DIR}/" "$SCRATCH_RUN/geometry/"
    else
        echo "ERROR: no geometry directory available for post-processing" >&2
        exit 2
    fi
fi

cd "$SCRATCH_RUN"

echo "--- Running compute_per_cell.py ---"
PER_CELL_ARGS="."
if [ -n "$GEOMETRY_FIELDS" ] && [ -f "$GEOMETRY_FIELDS" ]; then
    PER_CELL_ARGS="$PER_CELL_ARGS --geometry-fields $GEOMETRY_FIELDS"
fi
"$CONDA_PREFIX/bin/mpirun" -n "${SLURM_NTASKS}" -launcher fork \
    "$CONDA_PREFIX/bin/python" "${SIM_DIR}/compute_per_cell.py" $PER_CELL_ARGS \
    > compute_per_cell_recovery.log 2> compute_per_cell_recovery.err

echo "--- Running postprocess_metrics.py ---"
"$CONDA_PREFIX/bin/mpirun" -n "${SLURM_NTASKS}" -launcher fork \
    "$CONDA_PREFIX/bin/python" "${SIM_DIR}/postprocess_metrics.py" . \
    > postprocess_metrics_recovery.log 2> postprocess_metrics_recovery.err

echo "--- Running run_postprocessing.py ---"
"$CONDA_PREFIX/bin/python" "${SIM_DIR}/run_postprocessing.py" . \
    > run_postprocessing_recovery.log 2> run_postprocessing_recovery.err

echo "--- Repostprocess recovery finished successfully ---"
date
