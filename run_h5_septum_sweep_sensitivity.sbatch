#!/bin/bash
# Run h=5 septum-boundary sensitivity from existing per_cell_data.npz files.
#
# This is analysis-only: it does not replay checkpoints or rerun FEM. It runs
# the historical epi-excluded sweep and the new epi-inclusive variant side by
# side so the effect of allowing epicardial cells into the growing septum is
# directly comparable.
#
# Optional env vars:
#   T_MIN=-10
#   T_MAX=20
#   N_STEPS=61

#SBATCH --job-name=h5_sept_sweep
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-h5-sept-sweep.out
#SBATCH --error=slurm-%j-h5-sept-sweep.err

set -euo pipefail

SIM_DIR="/home/dtsteene/D1/cardiac-work"
MANIFEST="${SIM_DIR}/results/analysis/h5_sweep_submission/h5_corrected_sweep_cases.tsv"
OUT_ROOT="${SIM_DIR}/results/analysis"
T_MIN="${T_MIN:--10}"
T_MAX="${T_MAX:-20}"
N_STEPS="${N_STEPS:-61}"

echo "--- H5 Septum Sweep Sensitivity Started: ${SLURM_JOB_ID} ---"
date
echo "  SIM_DIR:  ${SIM_DIR}"
echo "  MANIFEST: ${MANIFEST}"
echo "  T range:  ${T_MIN} to ${T_MAX} mm (${N_STEPS} steps)"

set +u
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
set -u

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd "${SIM_DIR}"

mapfile -t RUN_DIRS < <(
    python - <<'PY'
from pathlib import Path
import csv

root = Path("/home/dtsteene/D1/cardiac-work")
manifest = root / "results/analysis/h5_sweep_submission/h5_corrected_sweep_cases.tsv"
sim_root = root / "results/sims"

with manifest.open(newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        matches = sorted(sim_root.glob(f"*/UKB_6beats_run_{row['job_id']}"))
        if not matches:
            continue
        run = matches[-1]
        if (run / "per_cell_data.npz").exists():
            print(run)
PY
)

if [ "${#RUN_DIRS[@]}" -lt 3 ]; then
    echo "ERROR: need at least 3 completed per-cell cases, found ${#RUN_DIRS[@]}"
    exit 2
fi

echo "  Cases: ${#RUN_DIRS[@]}"
printf '    %s\n' "${RUN_DIRS[@]}"

echo
echo "=== Historical epi-excluded envelope ==="
date
python "${SIM_DIR}/analyze_sweep.py" \
    --output-dir "${OUT_ROOT}/h5_sweep_septum_sensitivity_epi_excluded" \
    --t-min "${T_MIN}" --t-max "${T_MAX}" --n-steps "${N_STEPS}" \
    "${RUN_DIRS[@]}"

echo
echo "=== Epi-inclusive envelope ==="
date
python "${SIM_DIR}/analyze_sweep.py" \
    --include-epi \
    --output-dir "${OUT_ROOT}/h5_sweep_septum_sensitivity_epi_inclusive" \
    --t-min "${T_MIN}" --t-max "${T_MAX}" --n-steps "${N_STEPS}" \
    "${RUN_DIRS[@]}"

echo
echo "--- H5 Septum Sweep Sensitivity Finished ---"
date
