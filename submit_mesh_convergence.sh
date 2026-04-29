#!/bin/bash

# Submit a focused mesh-convergence study for the thesis.
#
# Default design:
#   meshes: 10, 7.5, 5 mm characteristic length
#   cases:  sPAP22, sPAP60, sPAP95
#
# This brackets the pressure-loading spectrum while keeping the study small
# enough to finish. It compares the quantities of interest used in the thesis,
# not just a generic displacement norm.

set -euo pipefail

cd /home/dtsteene/D1/cardiac-work

PARTITION=${PARTITION:-mi50q}
BEATS=${BEATS:-6}
BPM=${BPM:-75}
MESHES=${MESHES:-"10 7.5 5"}
CASES=${CASES:-"sPAP22 sPAP60 sPAP95"}
CIRC_DIR=${CIRC_DIR:-"data/ukb_circ_v12_exp"}
OUT_ROOT=${OUT_ROOT:-"data/mesh_convergence"}
ANALYSIS_DIR="results/analysis/mesh_convergence"

if [ "${PARTITION}" = "habanaq" ]; then
    echo "ERROR: refusing to submit mesh-convergence jobs to habanaq."
    echo "Set PARTITION to a CPU queue, for example: PARTITION=mi50q ./submit_mesh_convergence.sh"
    exit 2
fi

if [ "${PARTITION}" = "rome16q" ]; then
    echo "ERROR: refusing to submit mesh-convergence jobs to rome16q."
    echo "Recent convergence runs there hit intermittent shared-filesystem I/O errors during post-processing."
    echo "Use an x86 queue with a cleaner I/O path, for example: PARTITION=mi50q ./submit_mesh_convergence.sh"
    exit 2
fi

mkdir -p "${ANALYSIS_DIR}"
STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="${ANALYSIS_DIR}/submissions_${STAMP}.tsv"

mesh_tag() {
    local h="$1"
    h="${h/./p}"
    echo "L${h}"
}

walltime_for_mesh() {
    local h="$1"
    python3 - "$h" <<'PY'
import sys
h = float(sys.argv[1])
if h <= 5.0:
    print("24:00:00")
elif h <= 7.5:
    print("12:00:00")
else:
    print("8:00:00")
PY
}

ntasks_for_mesh() {
    local h="$1"
    python3 - "$h" <<'PY'
import sys
h = float(sys.argv[1])
print(16 if h <= 5.0 else 8)
PY
}

echo -e "timestamp\tmesh_mm\tcase\tgeometry_dir\tgeometry_job\tsim_job\tcirc_file\tbeats\tpartition" > "${MANIFEST}"

echo "=== Mesh convergence submission ==="
echo "Partition: ${PARTITION}"
echo "Meshes:    ${MESHES}"
echo "Cases:     ${CASES}"
echo "Beats:     ${BEATS}"
echo "Manifest:  ${MANIFEST}"
echo

for H in ${MESHES}; do
    TAG=$(mesh_tag "${H}")
    OUT_BASE="${PWD}/${OUT_ROOT}/ukb_${TAG}"
    GEOM_DIR="${OUT_BASE}/ukb/geometry"

    GEOM_JOB=""
    DEPENDENCY_ARGS=()
    if [ ! -d "${GEOM_DIR}/geometry.bp" ] || [ ! -f "${GEOM_DIR}/geometry_fields.npz" ]; then
        echo "Submitting geometry generation for ${TAG} (${H} mm)"
        GEOM_JOB=$(sbatch \
            --partition="${PARTITION}" \
            --export=ALL,CHAR_LENGTH="${H}",OUT_BASE="${OUT_BASE}" \
            run_mesh_convergence_geometry.sbatch | awk '{print $4}')
        DEPENDENCY_ARGS=(--dependency="afterok:${GEOM_JOB}")
        echo "  geometry job: ${GEOM_JOB}"
    else
        echo "Geometry exists for ${TAG}: ${GEOM_DIR}"
    fi

    for CASE in ${CASES}; do
        CIRC_FILE="${CIRC_DIR}/optimized_regazzoni_ukb_${CASE}.json"
        if [ ! -f "${CIRC_FILE}" ]; then
            echo "ERROR: missing circulation file ${CIRC_FILE}"
            exit 3
        fi

        TIME_LIMIT=$(walltime_for_mesh "${H}")
        NTASKS=$(ntasks_for_mesh "${H}")
        COMMENT="mesh-convergence ${TAG} ${CASE} ${BEATS}beats; v12_exp circulation; no habanaq"
        echo "Submitting ${TAG} ${CASE}: time=${TIME_LIMIT}, ntasks=${NTASKS}"
        SIM_JOB=$(sbatch \
            --job-name="mc_${TAG}_${CASE}" \
            --partition="${PARTITION}" \
            --time="${TIME_LIMIT}" \
            --ntasks="${NTASKS}" \
            "${DEPENDENCY_ARGS[@]}" \
            --export=ALL,MESH=UKB,BPM="${BPM}",BEATS="${BEATS}",CHAR_LENGTH="${H}",CIRCULATION_PARAMS="${PWD}/${CIRC_FILE}",GEOMETRY_DIR="${GEOM_DIR}",GEOMETRY_FIELDS="${GEOM_DIR}/geometry_fields.npz",COMMENT="${COMMENT}" \
            run_sim_and_post.sbatch | awk '{print $4}')
        echo "  sim job: ${SIM_JOB}"

        echo -e "${STAMP}\t${H}\t${CASE}\t${GEOM_DIR}\t${GEOM_JOB:-existing}\t${SIM_JOB}\t${PWD}/${CIRC_FILE}\t${BEATS}\t${PARTITION}" >> "${MANIFEST}"
    done
done

echo
echo "Submitted mesh-convergence study."
echo "Manifest: ${MANIFEST}"
echo "Monitor:  squeue -u ${USER}"
echo "Analyze after completion:"
echo "  python analyze_mesh_convergence.py --manifest ${MANIFEST}"
