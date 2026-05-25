#!/bin/bash

# Replay the buggy reference-configuration Robin work formula on a saved
# displacement checkpoint. Used to populate the three-bugs energy-budget
# audit figure for the thesis (chapter 3).
#
# Usage:
#   sbatch --export=RESULTS_DIR="/path/to/case_dir" run_robin_reference_replay.sbatch

#SBATCH --job-name=robin_replay
#SBATCH --partition=mi50q
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --exclude=n052
#SBATCH --output=robin_replay_%j.out
#SBATCH --error=robin_replay_%j.err

if [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: RESULTS_DIR not set. Use: sbatch --export=RESULTS_DIR=... sbatch/jobs/run_robin_reference_replay.sbatch"
    exit 1
fi

SIM_DIR="/home/dtsteene/D1/cardiac-work"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

echo "=== Robin reference-config replay: ${RESULTS_DIR} ==="
date

if [ ! -d "${RESULTS_DIR}/solver/checkpoint.bp" ]; then
    echo "ERROR: Missing ${RESULTS_DIR}/solver/checkpoint.bp"
    exit 1
fi

$CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
    $CONDA_PREFIX/bin/python ${SIM_DIR}/robin_reference_replay.py "${RESULTS_DIR}"

EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo "ERROR: robin_reference_replay.py failed (exit $EXIT)"
    exit $EXIT
fi

echo "=== Done ==="
date
