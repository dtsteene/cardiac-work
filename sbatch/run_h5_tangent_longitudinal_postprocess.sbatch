#!/bin/bash
# Rerun h=5 postprocessing after redefining longitudinal strain as the
# wall-tangent projection of the LDRB apex-to-base direction.

#SBATCH --job-name=h5_tangent_ll
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --array=1-16%4
#SBATCH --output=slurm-%A_%a-tangent-ll.out
#SBATCH --error=slurm-%A_%a-tangent-ll.err

set -euo pipefail

SIM_DIR="/home/dtsteene/D1/cardiac-work"
MANIFEST="${MANIFEST:-${SIM_DIR}/results/analysis/h5_sweep_submission/h5_corrected_sweep_cases.tsv}"

set +u
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
set -u
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

CACHE_BASE="/global/D1/homes/dtsteene/tmp/fenics-cache"
export XDG_CACHE_HOME="${CACHE_BASE}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export TMPDIR="${XDG_CACHE_HOME}/tmp"
mkdir -p "${TMPDIR}"
trap 'rm -rf "${XDG_CACHE_HOME}"' EXIT

LINE=$(awk -v n="${SLURM_ARRAY_TASK_ID}" 'NR == n + 1 {print}' "${MANIFEST}")
if [ -z "${LINE}" ]; then
    echo "No manifest row for array task ${SLURM_ARRAY_TASK_ID}"
    exit 2
fi

CASE=$(echo "${LINE}" | awk '{print $1}')
JOB_ID=$(echo "${LINE}" | awk '{print $2}')
RESULTS_DIR=$(find "${SIM_DIR}/results/sims" -path "*/UKB_6beats_run_${JOB_ID}" -type d -print -quit)

if [ -z "${RESULTS_DIR}" ]; then
    echo "Could not find results dir for ${CASE} job ${JOB_ID}"
    exit 3
fi

GEOM_FIELDS="${RESULTS_DIR}/geometry/geometry_fields.npz"
if [ ! -f "${GEOM_FIELDS}" ]; then
    echo "Missing geometry fields: ${GEOM_FIELDS}"
    exit 4
fi

echo "=== Tangent-longitudinal h=5 postprocess ==="
echo "case=${CASE}"
echo "job_id=${JOB_ID}"
echo "results_dir=${RESULTS_DIR}"
date

cd "${RESULTS_DIR}"

if [ -f per_cell_data.npz ] && [ ! -f per_cell_data_raw_l0.npz ]; then
    cp -p per_cell_data.npz per_cell_data_raw_l0.npz
    echo "Backed up per_cell_data.npz -> per_cell_data_raw_l0.npz"
fi

if [ -d metrics ] && [ ! -d metrics_raw_l0 ]; then
    cp -a metrics metrics_raw_l0
    echo "Backed up metrics/ -> metrics_raw_l0/"
fi

echo
echo "--- compute_per_cell.py with tangent l0 ---"
date
time "${CONDA_PREFIX}/bin/mpirun" -n "${SLURM_NTASKS}" -launcher fork \
    "${CONDA_PREFIX}/bin/python" "${SIM_DIR}/compute_per_cell.py" \
    "${RESULTS_DIR}" \
    --geometry-fields "${GEOM_FIELDS}" \
    > "${RESULTS_DIR}/compute_per_cell_tangent_ll.log" \
    2> "${RESULTS_DIR}/compute_per_cell_tangent_ll.err"

echo
echo "--- postprocess_metrics.py with tangent l0 ---"
date
time "${CONDA_PREFIX}/bin/mpirun" -n "${SLURM_NTASKS}" -launcher fork \
    "${CONDA_PREFIX}/bin/python" "${SIM_DIR}/postprocess_metrics.py" \
    "${RESULTS_DIR}" \
    > "${RESULTS_DIR}/postprocess_metrics_tangent_ll.log" \
    2> "${RESULTS_DIR}/postprocess_metrics_tangent_ll.err"

echo
echo "--- run_postprocessing.py ---"
date
python "${SIM_DIR}/run_postprocessing.py" "${RESULTS_DIR}" \
    > "${RESULTS_DIR}/run_postprocessing_tangent_ll.log" \
    2> "${RESULTS_DIR}/run_postprocessing_tangent_ll.err"

echo
echo "=== Done ${CASE} (${JOB_ID}) ==="
date
