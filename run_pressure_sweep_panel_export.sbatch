#!/bin/bash
# Build a lightweight side-by-side ParaView panel for the capped RV-pressure sweep.
#
# Optional env:
#   OUT_DIR=/path/to/export
#   CASES="sPAP22 sPAP35 sPAP50 sPAP65 sPAP80 sPAP95"

#SBATCH --job-name=rv_panel_export
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-rv-panel-export.out
#SBATCH --error=slurm-%j-rv-panel-export.err

set -euo pipefail

SIM_DIR="/home/dtsteene/D1/cardiac-work"
OUT_DIR="${OUT_DIR:-${SIM_DIR}/paraview_exports/rv_pressure_work_panel}"
CASES="${CASES:-sPAP22 sPAP35 sPAP50 sPAP65 sPAP80 sPAP95}"

set +u
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
set -u

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd "${SIM_DIR}"

echo "--- RV pressure work panel export started: ${SLURM_JOB_ID} ---"
date
echo "OUT_DIR=${OUT_DIR}"
echo "CASES=${CASES}"

python "${SIM_DIR}/export_pressure_sweep_panel.py" \
    --output-dir "${OUT_DIR}" \
    --cases ${CASES} \
    --force

echo "--- RV pressure work panel export finished ---"
date
