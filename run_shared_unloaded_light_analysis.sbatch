#!/bin/bash
# Lightweight analysis for the shared-unloaded L5 sweep.
#
# Reads existing per_cell_data.npz files only. No FEM replay.

#SBATCH --job-name=sharedB_light
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j-sharedB-light.out
#SBATCH --error=slurm-%j-sharedB-light.err

set -euo pipefail

SIM_DIR="/home/dtsteene/D1/cardiac-work"
MANIFEST="${MANIFEST:-${SIM_DIR}/results/analysis/shared_unloaded_l5_sweep_20260512_175915/shared_unloaded_l5_cases.tsv}"
OUT_DIR="${OUT_DIR:-${SIM_DIR}/results/analysis/shared_unloaded_l5_sweep_20260512_175915}"
T_MIN="${T_MIN:--10}"
T_MAX="${T_MAX:-20}"
N_STEPS="${N_STEPS:-61}"

echo "--- Shared-unloaded light analysis started: ${SLURM_JOB_ID} ---"
date
echo "  MANIFEST: ${MANIFEST}"
echo "  OUT_DIR:  ${OUT_DIR}"
echo "  T range:  ${T_MIN} to ${T_MAX} mm (${N_STEPS} steps)"

set +u
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
set -u

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export H5_SWEEP_MANIFEST="${MANIFEST}"
export H5_SWEEP_OUT="${OUT_DIR}"

cd "${SIM_DIR}"

echo
echo "=== Core sweep tables ==="
python "${SIM_DIR}/analyze_h5_sweep_core.py"

mapfile -t RUN_DIRS < <(
    python - <<'PY' "${MANIFEST}"
from pathlib import Path
import csv
import sys

manifest = Path(sys.argv[1])
with manifest.open(newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        run = Path(row["result_dir"])
        if (run / "per_cell_data.npz").exists():
            print(run)
PY
)

if [ "${#RUN_DIRS[@]}" -lt 3 ]; then
    echo "ERROR: need at least 3 completed per-cell cases, found ${#RUN_DIRS[@]}"
    exit 2
fi

echo
echo "=== Septum envelope sweep: epi-excluded ==="
python "${SIM_DIR}/analyze_sweep.py" \
    --output-dir "${OUT_DIR}/septum_sensitivity_epi_excluded" \
    --t-min "${T_MIN}" --t-max "${T_MAX}" --n-steps "${N_STEPS}" \
    "${RUN_DIRS[@]}"

echo
echo "=== Septum envelope sweep: epi-inclusive ==="
python "${SIM_DIR}/analyze_sweep.py" \
    --include-epi \
    --output-dir "${OUT_DIR}/septum_sensitivity_epi_inclusive" \
    --t-min "${T_MIN}" --t-max "${T_MAX}" --n-steps "${N_STEPS}" \
    "${RUN_DIRS[@]}"

echo
echo "--- Shared-unloaded light analysis finished ---"
date
