#!/bin/bash

# Focused basal-boundary sensitivity for thesis mechanics.
#
# Baseline already exists in the mesh-convergence study:
#   BASE_DIRICHLET=x, ALPHA_EPI=1e5, ALPHA_BASE=1e6, h=5 mm
#
# This script submits the most important counterfactual:
#   BASE_DIRICHLET=none, same Robin springs, same 5 mm mesh, same 6 beats.

set -euo pipefail

cd /home/dtsteene/D1/cardiac-work

PARTITION=${PARTITION:-xeonmaxq}
BEATS=${BEATS:-6}
BPM=${BPM:-75}
CHAR_LENGTH=${CHAR_LENGTH:-5}
BASE_DIRICHLET=${BASE_DIRICHLET:-none}
CASES=${CASES:-"sPAP22 sPAP60 sPAP95"}
CIRC_DIR=${CIRC_DIR:-"data/ukb_circ_v12_exp"}
GEOM_DIR="${PWD}/data/mesh_convergence/ukb_L5/ukb/geometry"
GEOMETRY_FIELDS="${GEOM_DIR}/geometry_fields.npz"
ANALYSIS_DIR="results/analysis/base_dirichlet_sensitivity"

if [ ! -d "${GEOM_DIR}/geometry.bp" ] || [ ! -f "${GEOMETRY_FIELDS}" ]; then
    echo "ERROR: missing 5 mm convergence geometry at ${GEOM_DIR}"
    exit 2
fi

mkdir -p "${ANALYSIS_DIR}"
STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="${ANALYSIS_DIR}/submissions_${STAMP}.tsv"

echo -e "timestamp\tbase_dirichlet\tmesh_mm\tcase\tgeometry_dir\tsim_job\tcirc_file\tbeats\tpartition\talpha_epi\talpha_base" > "${MANIFEST}"

echo "=== Base Dirichlet sensitivity submission ==="
echo "Partition:      ${PARTITION}"
echo "Base Dirichlet: ${BASE_DIRICHLET}"
echo "Mesh:           ${CHAR_LENGTH} mm (${GEOM_DIR})"
echo "Cases:          ${CASES}"
echo "Beats:          ${BEATS}"
echo "Manifest:       ${MANIFEST}"
echo

for CASE in ${CASES}; do
    CIRC_FILE="${CIRC_DIR}/optimized_regazzoni_ukb_${CASE}.json"
    if [ ! -f "${CIRC_FILE}" ]; then
        echo "ERROR: missing circulation file ${CIRC_FILE}"
        exit 3
    fi

    MODE_TAG="${BASE_DIRICHLET//[^A-Za-z0-9]/_}"
    COMMENT="base-dirichlet sensitivity: ${BASE_DIRICHLET}; h=${CHAR_LENGTH}; ${CASE}; compare vs baseline x-clamp"
    echo "Submitting ${CASE}: BASE_DIRICHLET=${BASE_DIRICHLET}"
    SIM_JOB=$(sbatch \
        --job-name="bc_${MODE_TAG}_${CASE}" \
        --partition="${PARTITION}" \
        --time="24:00:00" \
        --ntasks=16 \
        --export=ALL,MESH=UKB,BPM="${BPM}",BEATS="${BEATS}",CHAR_LENGTH="${CHAR_LENGTH}",BASE_DIRICHLET="${BASE_DIRICHLET}",CIRCULATION_PARAMS="${PWD}/${CIRC_FILE}",GEOMETRY_DIR="${GEOM_DIR}",GEOMETRY_FIELDS="${GEOMETRY_FIELDS}",COMMENT="${COMMENT}" \
        run_sim_and_post.sbatch | awk '{print $4}')
    echo "  sim job: ${SIM_JOB}"

    echo -e "${STAMP}\t${BASE_DIRICHLET}\t${CHAR_LENGTH}\t${CASE}\t${GEOM_DIR}\t${SIM_JOB}\t${PWD}/${CIRC_FILE}\t${BEATS}\t${PARTITION}\t1e5\t1e6" >> "${MANIFEST}"
done

echo
echo "Submitted base Dirichlet sensitivity study."
echo "Manifest: ${MANIFEST}"
echo "Analyze after completion:"
echo "  python analyze_base_dirichlet_sensitivity.py --variant-manifest ${MANIFEST}"
