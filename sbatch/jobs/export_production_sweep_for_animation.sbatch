#!/bin/bash
# Export ED-deformed meshes for the production capped L5 sweep,
# ready for local PyVista animation. Runs serial; very light I/O job.

#SBATCH --job-name=export_sweep_anim
#SBATCH --partition=habanaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

SIM_DIR="/home/dtsteene/D1/cardiac-work"

echo "--- Export Sweep Animation Job Started: $SLURM_JOB_ID ---"
date

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

CACHE_BASE="/global/D1/homes/dtsteene/tmp/fenics-cache"
export XDG_CACHE_HOME="${CACHE_BASE}/${SLURM_JOB_ID}"
export TMPDIR="${XDG_CACHE_HOME}/tmp"
mkdir -p "${XDG_CACHE_HOME}" "${TMPDIR}"
trap "rm -rf ${XDG_CACHE_HOME}" EXIT

cd $SIM_DIR
time $CONDA_PREFIX/bin/python ${SIM_DIR}/export_production_sweep_for_animation.py "$@"

EXIT=$?
echo "--- Export Sweep Animation Job Finished: exit=$EXIT ---"
date
exit $EXIT
