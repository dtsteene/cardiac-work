#!/bin/bash
# Export last-beat positive ParaView figure fields on a compute node.
#
# Optional env vars:
#   CASES="capped5_sPAP60:/path/to/sPAP60 capped5_sPAP95:/path/to/sPAP95"
#   OUT_ROOT=/path/to/paraview_exports/core_figure_fields_positive50
#   FRAMES=50
#   ACCUMULATION_STRIDE=5
#   FORCE=1

#SBATCH --job-name=core_fig_viz
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --output=slurm-core-figure-%j.out
#SBATCH --error=slurm-core-figure-%j.err

SIM_DIR="/home/dtsteene/D1/cardiac-work"
RUN_ROOT="/home/dtsteene/D1/cardiac-work/results/sims/2026-05-10/capped_shared_l5_20260510_141015"

CASES=${CASES:-"capped5_sPAP60:${RUN_ROOT}/sPAP60 capped5_sPAP95:${RUN_ROOT}/sPAP95"}
OUT_ROOT=${OUT_ROOT:-"/home/dtsteene/D1/cardiac-work/paraview_exports/core_figure_fields_positive50"}
FRAMES=${FRAMES:-50}
ACCUMULATION_STRIDE=${ACCUMULATION_STRIDE:-5}
FORCE=${FORCE:-1}

echo "--- Core figure export started: ${SLURM_JOB_ID} ---"
date
echo "Cases: ${CASES}"
echo "Output root: ${OUT_ROOT}"
echo "Frames: ${FRAMES}"
echo "Accumulation stride: ${ACCUMULATION_STRIDE}"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

CACHE_BASE="/global/D1/homes/dtsteene/tmp/fenics-cache"
export XDG_CACHE_HOME="${CACHE_BASE}/${SLURM_JOB_ID}"
export TMPDIR="${XDG_CACHE_HOME}/tmp"
mkdir -p "${TMPDIR}" "${OUT_ROOT}"
trap "rm -rf ${XDG_CACHE_HOME}" EXIT

cd "${SIM_DIR}"

for item in ${CASES}; do
    label="${item%%:*}"
    result_dir="${item#*:}"
    out_dir="${OUT_ROOT}/${label}"
    case_label="${label}_positive${FRAMES}"
    force_args=()
    if [ "${FORCE}" = "1" ]; then
        force_args=(--force)
    fi

    echo
    echo "=== ${label} ==="
    echo "Result: ${result_dir}"
    echo "Output: ${out_dir}"
    date
    time python "${SIM_DIR}/export_core_figure_paraview.py" \
        "${result_dir}" \
        --case-label "${case_label}" \
        --output-dir "${out_dir}" \
        --frames "${FRAMES}" \
        --accumulation-stride "${ACCUMULATION_STRIDE}" \
        "${force_args[@]}"
    exit_code=$?
    if [ "${exit_code}" -ne 0 ]; then
        echo "ERROR: ${label} failed with exit code ${exit_code}"
        exit "${exit_code}"
    fi
done

echo
echo "--- Core figure export finished ---"
date
