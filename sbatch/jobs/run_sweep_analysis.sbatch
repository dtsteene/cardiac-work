#!/bin/bash
# Run the lightweight cross-simulation sweep analysis for a manifest selected by env vars.
#
# Required/optional env:
#   SWEEP_MANIFEST=/path/to/cases.tsv
#   SWEEP_OUT=/path/to/output_dir

#SBATCH --job-name=sweep_analysis
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j-sweep-analysis.out
#SBATCH --error=slurm-%j-sweep-analysis.err

set -euo pipefail

SIM_DIR="/home/dtsteene/D1/cardiac-work"

set +u
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
set -u

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd "${SIM_DIR}"

echo "=== cross-simulation sweep analysis ==="
echo "job_id=${SLURM_JOB_ID}"
echo "manifest=${SWEEP_MANIFEST:-<default>}"
echo "out=${SWEEP_OUT:-<default>}"
date

python "${SIM_DIR}/sweep_analysis.py"

echo "=== done ==="
date
