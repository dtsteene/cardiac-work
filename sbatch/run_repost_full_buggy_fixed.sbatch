#!/bin/bash
#SBATCH --job-name=repost_full_dev
#SBATCH --partition=mi50q
#SBATCH --exclude=n052
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Re-postprocess two existing runs with --no-skip-regional-internal so the
# metrics file gets mean_sigma_ff_LV/RV/Septum and the work_active/passive/comp
# decomposition. Required for the deviatoric-stress counterfactual figure in
# the Validating-the-Mechanical-Reference chapter.

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

echo "=== Re-postprocess BUGGY (with legacy active stress) ==="
echo "  Dir: $BUGGY_DIR"
cd "$BUGGY_DIR"
$CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
    $CONDA_PREFIX/bin/python ${SIM_DIR}/postprocess_metrics_buggy_active.py . --no-skip-regional-internal \
    > "${BUGGY_DIR}/postprocess_metrics_full.log" 2> "${BUGGY_DIR}/postprocess_metrics_full.err"
echo "Buggy re-postprocess done"

echo "=== Re-postprocess FIXED (production active stress) ==="
echo "  Dir: $FIXED_DIR"
cd "$FIXED_DIR"
$CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
    $CONDA_PREFIX/bin/python ${SIM_DIR}/postprocess_metrics.py . --no-skip-regional-internal \
    > "${FIXED_DIR}/postprocess_metrics_full.log" 2> "${FIXED_DIR}/postprocess_metrics_full.err"
echo "Fixed re-postprocess done"

echo "--- Job Finished ---"
date
