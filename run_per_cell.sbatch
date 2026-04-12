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

if [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: RESULTS_DIR not set"
    exit 1
fi

SIM_DIR="/home/dtsteene/D1/cardiac-work"

echo "--- Per-Cell Job Started: $SLURM_JOB_ID ---"
date
echo "  Results dir: $RESULTS_DIR"
echo "  Retag septum: ${RETAG_SEPTUM:-0}"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

EXTRA_ARGS=""
if [ "${RETAG_SEPTUM:-0}" = "1" ]; then
    EXTRA_ARGS="--retag-septum"
fi

cd $SIM_DIR
time $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
    $CONDA_PREFIX/bin/python ${SIM_DIR}/compute_per_cell.py "$RESULTS_DIR" $EXTRA_ARGS

EXIT=$?
echo "--- Job Finished (exit=$EXIT) ---"
date
exit $EXIT
