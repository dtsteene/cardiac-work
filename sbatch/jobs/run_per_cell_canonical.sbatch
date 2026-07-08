#!/bin/bash
# Per-cell work computation in CANONICAL ATLAS MODE.
#
# Runs compute_per_cell.py in canonical/reference-config mode, reading
# geometry_fields.npz from the case's own geometry/ dir (written by
# geometry_generator.py at mesh-generation time — no precompute step needed).
#
# → writes per_cell_data_canonical.npz alongside the existing ED-tagged
#   per_cell_data.npz (which is NOT overwritten)
#
# Companion to run_per_cell.sbatch (which does the default ED-tagging path).
# Submit this to rerun the per-cell computation in canonical/reference-config
# mode so you have BOTH per_cell_data.npz (ED) and per_cell_data_canonical.npz
# (reference) available for cross-comparison analyses.
#
# Required env vars (set via --export):
#   RESULTS_DIR=/path/to/sim/results_dir

#SBATCH --job-name=per_cell_canon
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

if [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: RESULTS_DIR not set"
    exit 1
fi

SIM_DIR="${SIM_DIR:-/home/dtsteene/D1/cardiac-work}"
GEOM_DIR="${RESULTS_DIR}/geometry"
GEOM_FIELDS="${GEOM_DIR}/geometry_fields.npz"

echo "--- Per-Cell Canonical Job Started: $SLURM_JOB_ID ---"
date
echo "  Results dir:       $RESULTS_DIR"
echo "  Geometry dir:      $GEOM_DIR"
echo "  Geometry fields:   $GEOM_FIELDS"
echo "  Output file:       ${RESULTS_DIR}/per_cell_data_canonical.npz"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "${CARDIAC_CONDA_ENV:-/home/dtsteene/.conda/envs/RV}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

# Per-job FFCx cache. NOTE: /tmp on milanq nodes is frequently out of space,
# so put the cache on D1 (which has plenty of space) instead. Per-job unique
# directory still isolates concurrent sbatch jobs from each other.
CACHE_BASE="${CACHE_BASE:-$HOME/tmp/fenics-cache}"
export XDG_CACHE_HOME="${CACHE_BASE}/${SLURM_JOB_ID}"
mkdir -p "${XDG_CACHE_HOME}"
# Also point TMPDIR at the same area so FFCx's linker scratch writes go there
# rather than to the node-local /tmp that may be full.
export TMPDIR="${XDG_CACHE_HOME}/tmp"
mkdir -p "${TMPDIR}"
echo "  FEniCS cache: ${XDG_CACHE_HOME}"
echo "  TMPDIR:       ${TMPDIR}"
trap "rm -rf ${XDG_CACHE_HOME}" EXIT

cd "$SIM_DIR"

if [ ! -f "$GEOM_FIELDS" ]; then
    echo "ERROR: $GEOM_FIELDS not found"
    echo "Geometry fields should be written automatically by geometry_generator.py"
    echo "at mesh generation time. If this geometry was produced before that change,"
    echo "regenerate it with geometry_generator.py to populate geometry_fields.npz."
    exit 2
fi

# ── compute_per_cell.py in canonical mode ─────────────────────────────────
echo
echo "=== compute_per_cell.py --geometry-fields ==="
date
time $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
    $CONDA_PREFIX/bin/python "${SIM_DIR}/compute_per_cell.py" \
        "$RESULTS_DIR" \
        --geometry-fields "$GEOM_FIELDS" \
        --output-tag canonical
EXIT=$?

echo
echo "--- Job Finished (exit=$EXIT) ---"
date

if [ -f "${RESULTS_DIR}/per_cell_data_canonical.npz" ]; then
    echo "OK: ${RESULTS_DIR}/per_cell_data_canonical.npz exists"
else
    echo "MISSING: ${RESULTS_DIR}/per_cell_data_canonical.npz not written"
fi

exit $EXIT

exit $STEP2_EXIT
