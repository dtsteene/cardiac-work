#!/bin/bash
#SBATCH --job-name=repost_fixed_lb
#SBATCH --partition=mi50q
#SBATCH --exclude=n052
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Re-postprocess the 10mm fixed baseline with --no-skip-regional-internal
# and --last-beat. Last-beat restriction avoids replaying all 6 beats when
# the deviatoric counterfactual figure only needs cycle-end peak stress and
# closure on the same single beat the buggy comparator covers.

set -e

SIM_DIR="/home/dtsteene/D1/cardiac-work"
FIXED_DIR="${FIXED_DIR:-/global/D1/homes/dtsteene/cardiac-work/results/important_sims/mesh_convergence_study/04_sPAP22_h10p00_run_1050294}"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

echo "=== Re-postprocess FIXED last beat only ==="
echo "  Dir: $FIXED_DIR"
cd "$FIXED_DIR"
$CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
    $CONDA_PREFIX/bin/python ${SIM_DIR}/postprocess_metrics.py . \
    --no-skip-regional-internal --last-beat \
    > "${FIXED_DIR}/postprocess_metrics_full.log" 2> "${FIXED_DIR}/postprocess_metrics_full.err"
echo "Fixed last-beat repost done"

echo "--- Job Finished ---"
date
