#!/bin/bash
# Backfill AHA biventricular mid-ring tags onto existing per_cell_data.npz files.
# Geometry-only (no checkpoint replay), serial, cheap. Writes aha_tags.npy sidecars.
#
# Required env (via --export):
#   SWEEP_ROOT=/path/to/sweep   (backfills every <bundle>/<case> beneath it)
# Optional:
#   CASE_DIRS="dir1 dir2 ..."   (explicit case dirs instead of / in addition to SWEEP_ROOT)

#SBATCH --job-name=aha_band
#SBATCH --partition=habanaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

SIM_DIR="${SIM_DIR:-/home/dtsteene/D1/cardiac-work}"

echo "--- AHA band backfill: $SLURM_JOB_ID ---"
date
echo "  SWEEP_ROOT: ${SWEEP_ROOT:-<none>}"
echo "  CASE_DIRS:  ${CASE_DIRS:-<none>}"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "${CARDIAC_CONDA_ENV:-/home/dtsteene/.conda/envs/RV}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

CACHE_BASE="${CACHE_BASE:-$HOME/tmp/fenics-cache}"
export XDG_CACHE_HOME="${CACHE_BASE}/${SLURM_JOB_ID}"
export TMPDIR="${XDG_CACHE_HOME}/tmp"
mkdir -p "${XDG_CACHE_HOME}" "${TMPDIR}"
trap "rm -rf ${XDG_CACHE_HOME}" EXIT

cd "$SIM_DIR"
ARGS=()
if [ -n "${SWEEP_ROOT}" ]; then ARGS+=(--sweep "${SWEEP_ROOT}"); fi
if [ -n "${CASE_DIRS}" ]; then ARGS+=(${CASE_DIRS}); fi

time $CONDA_PREFIX/bin/python ${SIM_DIR}/compute_aha_band.py "${ARGS[@]}"
EXIT=$?
echo "--- Finished (exit=$EXIT) ---"
date
exit $EXIT
