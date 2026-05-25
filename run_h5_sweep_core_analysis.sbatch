#!/bin/bash
# Run the lightweight h=5 sweep analysis for a manifest selected by env vars.
#
# Required/optional env:
#   H5_SWEEP_MANIFEST=/path/to/cases.tsv
#   H5_SWEEP_OUT=/path/to/output_dir

#SBATCH --job-name=h5_core_analysis
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j-h5-core-analysis.out
#SBATCH --error=slurm-%j-h5-core-analysis.err

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

echo "=== h=5 sweep core analysis ==="
echo "job_id=${SLURM_JOB_ID}"
echo "manifest=${H5_SWEEP_MANIFEST:-<default>}"
echo "out=${H5_SWEEP_OUT:-<default>}"
date

python "${SIM_DIR}/analyze_h5_sweep_core.py"

echo "=== done ==="
date
