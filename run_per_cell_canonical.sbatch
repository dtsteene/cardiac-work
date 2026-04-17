#!/bin/bash
# Per-cell work computation in CANONICAL ATLAS MODE.
#
# Two steps in one job:
#   1. precompute_geometry_fields.py on the case's own geometry/ dir
#      → writes geometry/geometry_fields.npz (zero-strain reference tagging
#        computed on the unloaded mesh of THIS case — consistent cell count
#        with its solver mesh, but tags are material/reference-config based)
#   2. compute_per_cell.py $RESULTS_DIR --geometry-fields ... --output-tag canonical
#      → writes per_cell_data_canonical.npz alongside the existing ED-tagged
#        per_cell_data.npz (which is NOT overwritten)
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

SIM_DIR="/home/dtsteene/D1/cardiac-work"
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
conda activate RV
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

# Per-job FFCx cache. NOTE: /tmp on milanq nodes is frequently out of space,
# so put the cache on D1 (which has plenty of space) instead. Per-job unique
# directory still isolates concurrent sbatch jobs from each other.
CACHE_BASE="/global/D1/homes/dtsteene/tmp/fenics-cache"
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

# ── Step 1: precompute_geometry_fields.py on this case ────────────────────
echo
echo "=== Step 1: precompute_geometry_fields.py ==="
date
$CONDA_PREFIX/bin/python "${SIM_DIR}/precompute_geometry_fields.py" "$GEOM_DIR"
STEP1_EXIT=$?
if [ $STEP1_EXIT -ne 0 ]; then
    echo "Step 1 FAILED with exit $STEP1_EXIT — aborting"
    exit $STEP1_EXIT
fi
if [ ! -f "$GEOM_FIELDS" ]; then
    echo "Step 1 did not produce $GEOM_FIELDS — aborting"
    exit 2
fi

# ── Step 2: compute_per_cell.py in canonical mode ─────────────────────────
echo
echo "=== Step 2: compute_per_cell.py --geometry-fields ==="
date
time $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
    $CONDA_PREFIX/bin/python "${SIM_DIR}/compute_per_cell.py" \
        "$RESULTS_DIR" \
        --geometry-fields "$GEOM_FIELDS" \
        --output-tag canonical
STEP2_EXIT=$?

echo
echo "--- Job Finished (step1=$STEP1_EXIT, step2=$STEP2_EXIT) ---"
date

if [ -f "${RESULTS_DIR}/per_cell_data_canonical.npz" ]; then
    echo "OK: ${RESULTS_DIR}/per_cell_data_canonical.npz exists"
else
    echo "MISSING: ${RESULTS_DIR}/per_cell_data_canonical.npz not written"
fi

exit $STEP2_EXIT
