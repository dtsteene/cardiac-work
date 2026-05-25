#!/bin/bash
# Submit unloading-only A/B diagnostics for the RV prestress issue.
#
# This is only a dispatcher; the actual mechanics work runs through
# run_sim_and_post.sbatch on compute nodes, using the conda MPI launcher there.

set -euo pipefail

WORK_DIR="/global/D1/homes/dtsteene/cardiac-work"
DATA_DIR="${WORK_DIR}/data/ukb_circ_v12_exp"
GEOMETRY_DIR="${WORK_DIR}/data/mesh_convergence/ukb_L5/ukb/geometry"

PARTITION="${PARTITION:-mi50q}"
NTASKS="${NTASKS:-8}"
TIME_LIMIT="${TIME_LIMIT:-2:00:00}"
CASES="${CASES:-sPAP22 sPAP60 sPAP95}"

STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="${WORK_DIR}/results/unloading_ab_${STAMP}.tsv"

cd "${WORK_DIR}"
mkdir -p "$(dirname "${MANIFEST}")"
printf "timestamp\tcase\tvariant\tjob_id\tpartition\tntasks\ttime_limit\tjson\tcomment\n" > "${MANIFEST}"

submit_one() {
    local case_name="$1"
    local variant="$2"
    local extra_exports="$3"
    local json="${DATA_DIR}/optimized_regazzoni_ukb_${case_name}.json"

    if [ ! -f "${json}" ]; then
        echo "[MISSING] ${json}; skipping ${case_name}/${variant}"
        return 0
    fi

    local comment="unloading_ab_${case_name}_${variant}"
    local export_vars="ALL,MESH=UKB,BPM=75,BEATS=1,POST_FULL=0,STOP_AFTER_UNLOADING=1,CHAR_LENGTH=5.0,METRICS_SPACE=DG1,CIRCULATION_PARAMS=${json},GEOMETRY_DIR=${GEOMETRY_DIR},COMMENT=${comment}"
    if [ -n "${extra_exports}" ]; then
        export_vars="${export_vars},${extra_exports}"
    fi

    echo "Submitting ${case_name}/${variant}"
    local job_id
    job_id=$(sbatch --parsable \
        --job-name="unload_${case_name}_${variant}" \
        --partition="${PARTITION}" \
        --time="${TIME_LIMIT}" \
        --ntasks="${NTASKS}" \
        --export="${export_vars}" \
        "${WORK_DIR}/sbatch/run_sim_and_post.sbatch")

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date -Is)" "${case_name}" "${variant}" "${job_id}" \
        "${PARTITION}" "${NTASKS}" "${TIME_LIMIT}" "${json}" "${comment}" >> "${MANIFEST}"
    echo "  -> ${job_id}"
}

for case_name in ${CASES}; do
    submit_one "${case_name}" "baseline10" "PRE_CIRC_BEATS=10,PRE_CIRC_MAX_BEATS=10"
    submit_one "${case_name}" "preconv40" "PRE_CIRC_BEATS=10,PRE_CIRC_MAX_BEATS=40,PRE_CIRC_CONVERGENCE_TOL=0.005"
    submit_one "${case_name}" "rvmat2" "PRE_CIRC_BEATS=10,PRE_CIRC_MAX_BEATS=10,RV_MATERIAL_SCALE=2.0"
    submit_one "${case_name}" "rvsep2" "PRE_CIRC_BEATS=10,PRE_CIRC_MAX_BEATS=10,RV_MATERIAL_SCALE=2.0,SEPTUM_MATERIAL_SCALE=2.0"
    submit_one "${case_name}" "rvedp_cap8" "PRE_CIRC_BEATS=10,PRE_CIRC_MAX_BEATS=10,RV_EDP_MAX_MMHG=8.0"
done

echo
echo "Manifest: ${MANIFEST}"
echo "Submitted jobs:"
awk 'NR>1 {print "  " $4 "\t" $2 "/" $3}' "${MANIFEST}"
