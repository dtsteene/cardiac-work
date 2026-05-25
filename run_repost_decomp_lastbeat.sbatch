#!/bin/bash
#SBATCH --job-name=repost_decomp
#SBATCH --partition=mi50q
#SBATCH --exclude=n052
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Re-postprocess two existing runs with both --no-skip-regional-internal
# (so work_active/passive/comp arrays are written) and --no-skip-decomp
# (so mean_sigma_ff_* Cauchy stress arrays are written). --last-beat
# restricts the replay to the final 0.8 s cycle for speed on multi-beat
# runs. Required inputs for the deviatoric-stress counterfactual figure.

set -e

SIM_DIR="/home/dtsteene/D1/cardiac-work"
BUGGY_DIR="${BUGGY_DIR:-}"
FIXED_DIR="${FIXED_DIR:-/global/D1/homes/dtsteene/cardiac-work/results/important_sims/mesh_convergence_study/04_sPAP22_h10p00_run_1050294}"

if [ -z "$BUGGY_DIR" ]; then
    echo "FATAL: set BUGGY_DIR to the buggy-active sim results dir."
    exit 1
fi

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

POST_FLAGS="--no-skip-regional-internal --no-skip-decomp --last-beat"

echo "=== Re-postprocess BUGGY (legacy active stress, $POST_FLAGS) ==="
echo "  Dir: $BUGGY_DIR"
cd "$BUGGY_DIR"
$CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
    $CONDA_PREFIX/bin/python ${SIM_DIR}/postprocess_metrics_buggy_active.py . $POST_FLAGS \
    > "${BUGGY_DIR}/postprocess_metrics_full.log" 2> "${BUGGY_DIR}/postprocess_metrics_full.err"
echo "Buggy repost done"

echo "=== Re-postprocess FIXED (production active stress, $POST_FLAGS) ==="
echo "  Dir: $FIXED_DIR"
cd "$FIXED_DIR"
$CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
    $CONDA_PREFIX/bin/python ${SIM_DIR}/postprocess_metrics.py . $POST_FLAGS \
    > "${FIXED_DIR}/postprocess_metrics_full.log" 2> "${FIXED_DIR}/postprocess_metrics_full.err"
echo "Fixed repost done"

echo "--- Job Finished ---"
date
