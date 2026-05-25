#!/bin/bash
# Replay the h=5 sweep displacement checkpoints for the exploratory
# principal-strain diagnostic. This intentionally runs on a compute node:
# checkpoint replay is I/O-heavy and should not be run on the login node.
#
# Optional env vars:
#   STRIDE=10             Replay every Nth saved checkpoint step.
#   CASES="sPAP55 sPAP70" Restrict to selected case labels.
#   FORCE=1              Recompute existing cache files.
#   SUMMARY_ONLY=1       Regenerate CSV/figures from cache only.
#   H5_PRINCIPAL_MANIFEST=...  Case manifest TSV.
#   H5_PRINCIPAL_OUT=...       Output analysis directory.
#   H5_STRAIN_BEST=...         Optional existing strain-direction summary CSV.

#SBATCH --job-name=h5_principal_strain
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j-h5-principal.out
#SBATCH --error=slurm-%j-h5-principal.err

set -euo pipefail

SIM_DIR="/home/dtsteene/D1/cardiac-work"
STRIDE="${STRIDE:-10}"
CASES="${CASES:-}"
FORCE="${FORCE:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

echo "--- H5 Principal-Strain Job Started: ${SLURM_JOB_ID} ---"
date
echo "  SIM_DIR:      ${SIM_DIR}"
echo "  STRIDE:       ${STRIDE}"
echo "  CASES:        ${CASES:-<all>}"
echo "  FORCE:        ${FORCE}"
echo "  SUMMARY_ONLY: ${SUMMARY_ONLY}"
echo "  MANIFEST:     ${H5_PRINCIPAL_MANIFEST:-<script default>}"
echo "  OUT:          ${H5_PRINCIPAL_OUT:-<script default>}"
echo "  STRAIN_BEST:  ${H5_STRAIN_BEST:-<script default>}"

set +u
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
set -u

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

CACHE_BASE="/global/D1/homes/dtsteene/tmp/fenics-cache"
export XDG_CACHE_HOME="${CACHE_BASE}/${SLURM_JOB_ID}"
export TMPDIR="${XDG_CACHE_HOME}/tmp"
mkdir -p "${TMPDIR}"
trap "rm -rf ${XDG_CACHE_HOME}" EXIT

cd "${SIM_DIR}"

ARGS=(--stride "${STRIDE}")
if [ "${FORCE}" = "1" ]; then
    ARGS+=(--force)
fi
if [ "${SUMMARY_ONLY}" = "1" ]; then
    ARGS+=(--summary-only)
fi
if [ -n "${CASES}" ]; then
    # shellcheck disable=SC2206
    CASE_ARRAY=(${CASES})
    ARGS+=(--cases "${CASE_ARRAY[@]}")
fi

echo
echo "=== analyze_h5_principal_strain.py ==="
date
set +e
time $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
    $CONDA_PREFIX/bin/python "${SIM_DIR}/analyze_h5_principal_strain.py" \
        "${ARGS[@]}"
EXIT=$?
set -e

echo
echo "--- H5 Principal-Strain Job Finished (exit=${EXIT}) ---"
date
exit ${EXIT}
