#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=postprocess
#SBATCH --partition=mi50q
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --exclude=n052
#SBATCH --output=postprocess_%j.out
#SBATCH --error=postprocess_%j.err

# Usage:
#   sbatch --export=RESULTS_DIR="/path/to/results_dir" run_postprocess_only.sbatch
#   sbatch --export=RESULTS_DIR="/path/to/results_dir",RETAG_SEPTUM=1 run_postprocess_only.sbatch
#   sbatch --export=RESULTS_DIR="/path/to/results_dir",RETAG_SEPTUM_ED=1 run_postprocess_only.sbatch
#   sbatch --export=RESULTS_DIR="/path/to/results_dir",LAST_BEAT=1 run_postprocess_only.sbatch
#   sbatch --export=RESULTS_DIR="/path/to/results_dir",GEOMETRY_DIR="/path/to/src/geometry" run_postprocess_only.sbatch

if [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: RESULTS_DIR not set. Use: sbatch --export=RESULTS_DIR=... run_postprocess_only.sbatch"
    exit 1
fi

RETAG_SEPTUM=${RETAG_SEPTUM:-0}
RETAG_SEPTUM_ED=${RETAG_SEPTUM_ED:-0}
LAST_BEAT=${LAST_BEAT:-0}

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
    EXTRA_ARGS=""
    if [ "$RETAG_SEPTUM" = "1" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --retag-septum"
        echo "  Using geometric septum retagging in REFERENCE frame"
    fi
    if [ "$RETAG_SEPTUM_ED" = "1" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --retag-septum-ed"
        echo "  Using geometric septum retagging in END-DIASTOLIC frame (with preload)"
    fi
    if [ -n "$GEOMETRY_DIR" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --geometry-dir $GEOMETRY_DIR"
        echo "  Using external geometry dir: $GEOMETRY_DIR"
    fi
    if [ "$LAST_BEAT" = "1" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --last-beat"
        echo "  Processing last beat only"
    fi
    if [ "${REGIONAL_INTERNAL:-0}" = "1" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --no-skip-regional-internal"
        echo "  Regional internal work computation ENABLED"
    fi
    if [ "${DECOMP:-0}" = "1" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --no-skip-decomp"
        echo "  Directional decomposition (ss/nn) ENABLED"
    fi
    echo "Running postprocess_metrics.py ${EXTRA_ARGS}..."
    $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork $CONDA_PREFIX/bin/python ${SIM_DIR}/postprocess_metrics.py . ${EXTRA_ARGS}
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
