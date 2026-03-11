#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=postprocess
#SBATCH --partition=mi50q
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --exclude=n052
#SBATCH --output=postprocess_%j.out
#SBATCH --error=postprocess_%j.err

# Usage:
#   sbatch --export=RESULTS_DIR="/path/to/results_dir" run_postprocess_only.sbatch

if [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: RESULTS_DIR not set. Use: sbatch --export=RESULTS_DIR=... run_postprocess_only.sbatch"
    exit 1
fi

SIM_DIR="/home/dtsteene/D1/cardiac-work"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

cd ${RESULTS_DIR}

echo "=== Postprocessing: ${RESULTS_DIR} ==="
date

# 1. Compute metrics from checkpoint
if [ -f "simulation_params.json" ] && [ -d "solver/checkpoint.bp" ]; then
    echo "Running postprocess_metrics.py..."
    $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork $CONDA_PREFIX/bin/python ${SIM_DIR}/postprocess_metrics.py .
    if [ $? -ne 0 ]; then
        echo "ERROR: postprocess_metrics.py failed"
        exit 1
    fi
    echo "Metrics computed."

    # 2. Run analysis pipeline
    echo "Running run_postprocessing.py..."
    python3 ${SIM_DIR}/run_postprocessing.py .
    echo "Analysis complete."
else
    echo "ERROR: Missing simulation_params.json or solver/checkpoint.bp"
    exit 1
fi

echo "=== Done ==="
date
