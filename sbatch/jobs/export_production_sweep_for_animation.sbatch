#!/bin/bash
# Export ED-deformed meshes for the PAH pulmonary-windkessel sweep,
# ready for local PyVista animation.  Runs serial; very light I/O job.
#
# Required env var (set via --export or environment):
#   BUNDLE=no_frank_starling | frank_starling_preload | frank_starling_relax
#           (default: no_frank_starling)

#SBATCH --job-name=export_sweep_anim
#SBATCH --partition=habanaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

SIM_DIR="${SIM_DIR:-/home/dtsteene/D1/cardiac-work}"

echo "--- Export Sweep Animation Job Started: $SLURM_JOB_ID ---"
date
echo "  Bundle: ${BUNDLE:-no_frank_starling}"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "${CARDIAC_CONDA_ENV:-/home/dtsteene/.conda/envs/RV}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

# Per-job FFCx / PyVista cache directory to avoid races when jobs run in parallel.
CACHE_BASE="${CACHE_BASE:-$HOME/tmp/fenics-cache}"
export XDG_CACHE_HOME="${CACHE_BASE}/exportED_${SLURM_JOB_ID}"
export TMPDIR="${XDG_CACHE_HOME}/tmp"
mkdir -p "${XDG_CACHE_HOME}" "${TMPDIR}"
echo "  FEniCS cache: ${XDG_CACHE_HOME}"
echo "  TMPDIR:       ${TMPDIR}"
trap "rm -rf ${XDG_CACHE_HOME}" EXIT

cd $SIM_DIR
time $CONDA_PREFIX/bin/python ${SIM_DIR}/export_production_sweep_for_animation.py \
    --bundle "${BUNDLE:-no_frank_starling}" "$@"

EXIT=$?
echo "--- Export Sweep Animation Job Finished: exit=$EXIT ---"
date
exit $EXIT
