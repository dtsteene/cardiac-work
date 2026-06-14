#!/bin/bash
# Through-beat PVD animation export for baseline + severe PAH cases.
# Replays the last beat from the displacement checkpoint, accumulates
# per-cell cumulative work density and pressure-strain proxy density,
# and writes step_NNN.vtu + beat.pvd for each requested case.
#
# Optional env vars (set via --export or environment):
#   BUNDLE=no_frank_starling | frank_starling_preload | frank_starling_relax
#           (default: no_frank_starling)
#   CASES="case0_rv25 case7_rv95"  (space-separated, default: case0_rv25 case7_rv95)

#SBATCH --job-name=beat_anim
#SBATCH --partition=habanaq
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

SIM_DIR="/home/dtsteene/D1/cardiac-work"

echo "--- Beat Animation Export Started: $SLURM_JOB_ID ---"
date
echo "  Bundle: ${BUNDLE:-no_frank_starling}"
echo "  Cases:  ${CASES:-case0_rv25 case7_rv95}"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

# Per-job FFCx / PyVista cache directory to avoid JIT compilation races.
CACHE_BASE="/global/D1/homes/dtsteene/tmp/fenics-cache"
export XDG_CACHE_HOME="${CACHE_BASE}/exportBeat_${SLURM_JOB_ID}"
export TMPDIR="${XDG_CACHE_HOME}/tmp"
mkdir -p "${XDG_CACHE_HOME}" "${TMPDIR}"
echo "  FEniCS cache: ${XDG_CACHE_HOME}"
echo "  TMPDIR:       ${TMPDIR}"
trap "rm -rf ${XDG_CACHE_HOME}" EXIT

CASES_ARG=""
if [ -n "${CASES}" ]; then
    CASES_ARG="--cases ${CASES}"
fi

cd $SIM_DIR
time $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
    $CONDA_PREFIX/bin/python ${SIM_DIR}/export_beat_animation.py \
    --bundle "${BUNDLE:-no_frank_starling}" ${CASES_ARG}

EXIT=$?
echo "--- Beat Animation Export Finished (exit=$EXIT) ---"
date
exit $EXIT
