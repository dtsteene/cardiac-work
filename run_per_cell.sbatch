#!/bin/bash
# Per-cell work computation job
# Replays the last beat from checkpoint and computes per-cell work + tau + study region

#SBATCH --job-name=per_cell
#SBATCH --partition=habanaq
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Required env vars (set via --export):
#   RESULTS_DIR=/path/to/sim/results_dir
#   RETAG_SEPTUM=0 or 1 (default 0)
#   GEOMETRY_FIELDS=/path/to/canonical/geometry_fields.npz (optional)
#     If set, uses canonical atlas tagging via u_pre permutation.
#   TAG_AT_UNLOADED=0 or 1 (default 0)
#     If set to 1, uses legacy per-case unloaded-reference tagging.
#     Otherwise (default), uses per-case tag-at-ED (clinically meaningful).

if [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: RESULTS_DIR not set"
    exit 1
fi

SIM_DIR="/home/dtsteene/D1/cardiac-work"

echo "--- Per-Cell Job Started: $SLURM_JOB_ID ---"
date
echo "  Results dir: $RESULTS_DIR"
echo "  Retag septum: ${RETAG_SEPTUM:-0}"
echo "  Geometry fields: ${GEOMETRY_FIELDS:-<none>}"
echo "  Tag at unloaded: ${TAG_AT_UNLOADED:-0}"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

# Per-job FFCx cache directory to avoid JIT compilation races when many
# per_cell jobs start simultaneously. Without this, parallel jobs collide on
# the shared ~/.cache/fenics directory and some fail with JIT timeouts.
export XDG_CACHE_HOME="/tmp/fenics-cache-${SLURM_JOB_ID}"
mkdir -p "${XDG_CACHE_HOME}"
echo "  FEniCS cache: ${XDG_CACHE_HOME}"
# Clean up on exit so /tmp doesn't fill up
trap "rm -rf ${XDG_CACHE_HOME}" EXIT

EXTRA_ARGS=""
if [ "${RETAG_SEPTUM:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --retag-septum"
fi
if [ -n "${GEOMETRY_FIELDS}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --geometry-fields ${GEOMETRY_FIELDS}"
fi
if [ "${TAG_AT_UNLOADED:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --tag-at-unloaded"
fi
if [ -n "${BEAT}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --beat ${BEAT}"
fi
if [ "${ALL_BEATS:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --all-beats"
fi

cd $SIM_DIR
time $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
    $CONDA_PREFIX/bin/python ${SIM_DIR}/compute_per_cell.py "$RESULTS_DIR" $EXTRA_ARGS

EXIT=$?
echo "--- Job Finished (exit=$EXIT) ---"
date
exit $EXIT
