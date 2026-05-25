#!/bin/bash
# Recompute septum-boundary metrics from existing per_cell_data.npz files.
#
# Analysis only: no FEM replay and no checkpoint postprocessing.
#
# Env:
#   MANIFEST=/path/to/cases.tsv
#   OUT_DIR=/path/to/analysis/output
#   T_MIN=-10
#   T_MAX=20
#   N_STEPS=61

#SBATCH --job-name=l5_boundary
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j-l5-boundary.out
#SBATCH --error=slurm-%j-l5-boundary.err

set -euo pipefail

SIM_DIR="/home/dtsteene/D1/cardiac-work"
MANIFEST="${MANIFEST:-${SIM_DIR}/results/analysis/capped_shared_l5_sweep_20260510_141015/capped_shared_l5_cases.tsv}"
OUT_DIR="${OUT_DIR:-${SIM_DIR}/results/analysis/capped_shared_l5_sweep_20260510_141015}"
T_MIN="${T_MIN:--10}"
T_MAX="${T_MAX:-20}"
N_STEPS="${N_STEPS:-61}"

echo "--- L5 boundary metrics started: ${SLURM_JOB_ID} ---"
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

cd "${SIM_DIR}"

mapfile -t RUN_DIRS < <(
    python - <<'PY' "${MANIFEST}"
from pathlib import Path
import csv
import sys

manifest = Path(sys.argv[1])
sim_root = Path("/home/dtsteene/D1/cardiac-work/results/sims")
with manifest.open(newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        run = Path(row.get("result_dir", ""))
        if not run.exists() and row.get("job_id"):
            matches = sorted(sim_root.glob(f"*/UKB_6beats_run_{row['job_id']}"))
            run = matches[-1] if matches else Path("")
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
echo "--- L5 boundary metrics finished ---"
date
