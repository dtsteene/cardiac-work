#!/bin/bash

# Targeted "insurance" runs while thesis writing continues.
#
# 1. Patient-geometry cap=5 mini-sweep:
#    healthy and PAH geometries at low/mid/severe pressure, using one fixed
#    reference-tag geometry per mesh.
# 2. Capped L3.75 endpoint mesh runs:
#    direct capped fine-mesh endpoints for the numerical appendix.

set -euo pipefail

cd /home/dtsteene/D1/cardiac-work

PARTITION=${PARTITION:-mi50q}
BPM=${BPM:-75}
BEATS=${BEATS:-6}
STAMP=$(date +%Y%m%d_%H%M%S)
DATE_STR=$(date +%F)

SIM_DIR="/home/dtsteene/D1/cardiac-work"
ANALYSIS_DIR="${SIM_DIR}/results/analysis/insurance_runs_${STAMP}"
RESULT_ROOT="${SIM_DIR}/results/sims/${DATE_STR}/insurance_${STAMP}"
MANIFEST="${ANALYSIS_DIR}/submitted_jobs.tsv"

mkdir -p "${ANALYSIS_DIR}" "${RESULT_ROOT}"

printf "group\tmesh_key\tcase\tjob_id\tresult_dir\tjson\tgeometry_dir\tgeometry_fields\tnote\n" > "${MANIFEST}"

submit_patient_case() {
    local mesh_key="$1"
    local run_mesh="$2"
    local case="$3"
    local json="$4"
    local geom_dir="$5"
    local geom_fields="$6"
    local result_dir="${RESULT_ROOT}/patient_cap5_reference_tag/${mesh_key}/${case}"
    local comment="insurance_patient_cap5_reference_tag_${mesh_key}_${case}"

    mkdir -p "$(dirname "${result_dir}")"

    local jid
    jid=$(sbatch --parsable \
        --job-name="patcap5_${mesh_key}_${case}" \
        --partition="${PARTITION}" \
        --time="10:00:00" \
        --ntasks=8 \
        --export=ALL,MESH="${run_mesh}",BPM="${BPM}",BEATS="${BEATS}",POST_FULL=0,RUN_POSTPROCESS=1,CHAR_LENGTH=5.0,METRICS_SPACE=DG1,CIRCULATION_PARAMS="${json}",GEOMETRY_DIR="${geom_dir}",GEOMETRY_FIELDS="${geom_fields}",RV_EDP_MAX_MMHG=5.0,PRE_CIRC_BEATS=30,PRE_CIRC_MAX_BEATS=80,PRE_CIRC_CONVERGENCE_TOL=0.002,RESULTS_DIR_OVERRIDE="${result_dir}",COMMENT="${comment}" \
        run_sim_and_post.sbatch)

    printf "patient_cap5_reference_tag\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${mesh_key}" "${case}" "${jid}" "${result_dir}" "${json}" "${geom_dir}" "${geom_fields}" \
        "patient geometry, cap=5, fixed per-mesh reference tags" >> "${MANIFEST}"
    echo "submitted patient ${mesh_key} ${case}: ${jid}"
}

submit_l3_case() {
    local case="$1"
    local json="${SIM_DIR}/data/ukb_circ_v12_exp/optimized_regazzoni_ukb_${case}.json"
    local geom_dir="${SIM_DIR}/data/mesh_convergence/ukb_L3p75/ukb/geometry"
    local geom_fields="${geom_dir}/geometry_fields.npz"
    local result_dir="${RESULT_ROOT}/capped_l3p75_endpoints/${case}"
    local comment="insurance_capped_l3p75_${case}_cap5_reference_tag"

    mkdir -p "$(dirname "${result_dir}")"

    local jid
    jid=$(sbatch --parsable \
        --job-name="capL3p75_${case}" \
        --partition="${PARTITION}" \
        --time="24:00:00" \
        --ntasks=16 \
        --export=ALL,MESH=UKB,BPM="${BPM}",BEATS="${BEATS}",POST_FULL=0,RUN_POSTPROCESS=1,CHAR_LENGTH=3.75,METRICS_SPACE=DG1,CIRCULATION_PARAMS="${json}",GEOMETRY_DIR="${geom_dir}",GEOMETRY_FIELDS="${geom_fields}",RV_EDP_MAX_MMHG=5.0,PRE_CIRC_BEATS=30,PRE_CIRC_MAX_BEATS=80,PRE_CIRC_CONVERGENCE_TOL=0.002,RESULTS_DIR_OVERRIDE="${result_dir}",COMMENT="${comment}" \
        run_sim_and_post.sbatch)

    printf "capped_l3p75_endpoint\tukb\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${case}" "${jid}" "${result_dir}" "${json}" "${geom_dir}" "${geom_fields}" \
        "direct capped fine-mesh endpoint" >> "${MANIFEST}"
    echo "submitted L3.75 ${case}: ${jid}"
}

HEALTHY_GEOM="${SIM_DIR}/results/sims/2026-04-29/HEALTHY_6beats_run_1081742/geometry"
HEALTHY_FIELDS="${HEALTHY_GEOM}/geometry_fields.npz"
PAH_GEOM="${SIM_DIR}/results/sims/2026-04-29/PAH_6beats_run_1081750/geometry"
PAH_FIELDS="${PAH_GEOM}/geometry_fields.npz"

for case in sPAP22 sPAP65 sPAP95; do
    submit_patient_case \
        healthy HEALTHY "${case}" \
        "${SIM_DIR}/data/patient_mesh_circ_v12_exp/healthy/optimized_regazzoni_patient_healthy_${case}.json" \
        "${HEALTHY_GEOM}" "${HEALTHY_FIELDS}"

    submit_patient_case \
        pah PAH "${case}" \
        "${SIM_DIR}/data/patient_mesh_circ_v12_exp/pah/optimized_regazzoni_patient_pah_${case}.json" \
        "${PAH_GEOM}" "${PAH_FIELDS}"
done

for case in sPAP22 sPAP60 sPAP95; do
    submit_l3_case "${case}"
done

echo "Manifest: ${MANIFEST}"
echo "Result root: ${RESULT_ROOT}"
