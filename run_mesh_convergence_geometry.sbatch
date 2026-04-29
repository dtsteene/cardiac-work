#!/bin/bash

# Generate one shared UKB geometry for a mesh-convergence level.
#
# Required environment variables:
#   CHAR_LENGTH  characteristic mesh length in mm, e.g. 10, 7.5, 5
#   OUT_BASE     output directory; geometry will be written to OUT_BASE/ukb/geometry
#
# This job deliberately defaults to mi50q and the submitter refuses habanaq.

#SBATCH --job-name=meshconv_geo
#SBATCH --partition=mi50q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

if [ -z "${CHAR_LENGTH:-}" ]; then
    echo "ERROR: CHAR_LENGTH is not set"
    exit 2
fi

if [ -z "${OUT_BASE:-}" ]; then
    echo "ERROR: OUT_BASE is not set"
    exit 2
fi

SIM_DIR="/home/dtsteene/D1/cardiac-work"

echo "--- Mesh convergence geometry job started: ${SLURM_JOB_ID} ---"
date
echo "  CHAR_LENGTH: ${CHAR_LENGTH} mm"
echo "  OUT_BASE:    ${OUT_BASE}"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export XDG_CACHE_HOME="/global/D1/homes/dtsteene/tmp/fenics-cache/${SLURM_JOB_ID}"
export TMPDIR="${XDG_CACHE_HOME}/tmp"
mkdir -p "${TMPDIR}"
trap "rm -rf ${XDG_CACHE_HOME}" EXIT

cd "${SIM_DIR}"

python geometry_generator.py \
    --single ukb \
    -c "${CHAR_LENGTH}" \
    --output-dir "${OUT_BASE}"

GEOM="${OUT_BASE}/ukb/geometry"
if [ ! -d "${GEOM}/geometry.bp" ]; then
    echo "ERROR: geometry.bp was not created at ${GEOM}"
    exit 3
fi

if [ ! -f "${GEOM}/geometry_fields.npz" ]; then
    echo "ERROR: geometry_fields.npz was not created at ${GEOM}"
    exit 4
fi

echo "OK: generated ${GEOM}"
date
