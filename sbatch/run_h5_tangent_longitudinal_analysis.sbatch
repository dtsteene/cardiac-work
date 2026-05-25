#!/bin/bash
# Regenerate analysis tables/figures after tangent-longitudinal postprocessing.

#SBATCH --job-name=h5_tangent_analysis
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j-tangent-analysis.out
#SBATCH --error=slurm-%j-tangent-analysis.err

set -euo pipefail

SIM_DIR="/home/dtsteene/D1/cardiac-work"
MANIFEST="${SIM_DIR}/results/analysis/h5_sweep_submission/h5_corrected_sweep_cases.tsv"
OUT_ROOT="${SIM_DIR}/results/analysis"

set +u
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
set -u
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd "${SIM_DIR}"

echo "=== Tangent-longitudinal analysis started: ${SLURM_JOB_ID} ==="
date

echo
echo "--- analyze_h5_sweep_core.py ---"
python "${SIM_DIR}/analyze_h5_sweep_core.py"

echo
echo "--- analyze_h5_strain_directions.py ---"
python "${SIM_DIR}/analyze_h5_strain_directions.py"

echo
echo "--- analyze_h5_septum_mechanism.py ---"
python "${SIM_DIR}/analyze_h5_septum_mechanism.py"

echo
echo "--- plot_h5_thesis_updates.py ---"
python "${SIM_DIR}/plot_h5_thesis_updates.py"

echo
echo "--- septum sweep sensitivity: epi excluded/inclusive ---"
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

python "${SIM_DIR}/analyze_sweep.py" \
    --output-dir "${OUT_ROOT}/h5_sweep_septum_sensitivity_epi_excluded" \
    --t-min -10 --t-max 20 --n-steps 61 \
    "${RUN_DIRS[@]}"

python "${SIM_DIR}/analyze_sweep.py" \
    --include-epi \
    --output-dir "${OUT_ROOT}/h5_sweep_septum_sensitivity_epi_inclusive" \
    --t-min -10 --t-max 20 --n-steps 61 \
    "${RUN_DIRS[@]}"

echo
echo "=== Tangent-longitudinal analysis finished ==="
date
