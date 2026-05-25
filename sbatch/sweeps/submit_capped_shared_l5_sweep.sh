#!/usr/bin/env bash
# Submit the capped RV-EDP production sweep on the shared h=5 UKB mesh.
#
# This is the strict-canonical rerun: the same L5 geometry is used at
# simulation time and the same geometry_fields.npz is used for per-cell
# postprocessing, so the resulting cases should all have identical cell counts
# and labels.

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
DATA_DIR="${WORK_DIR}/data/ukb_circ_v12_exp"
GEOM_DIR="${WORK_DIR}/data/mesh_convergence/ukb_L5/ukb/geometry"
GEOM_FIELDS="${GEOM_DIR}/geometry_fields.npz"

CASES="${CASES:-sPAP22 sPAP25 sPAP30 sPAP35 sPAP45 sPAP50 sPAP55 sPAP60 sPAP65 sPAP70 sPAP75 sPAP80 sPAP85 sPAP87 sPAP92 sPAP95}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${WORK_DIR}/results/sims/$(date +%F)/capped_shared_l5_${STAMP}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${WORK_DIR}/results/analysis/capped_shared_l5_sweep_${STAMP}}"
MANIFEST="${MANIFEST:-${ANALYSIS_DIR}/capped_shared_l5_cases.tsv}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -d "${GEOM_DIR}/geometry.bp" ]; then
    echo "Missing shared geometry.bp: ${GEOM_DIR}/geometry.bp" >&2
    exit 2
fi
if [ ! -f "${GEOM_FIELDS}" ]; then
    echo "Missing shared geometry_fields.npz: ${GEOM_FIELDS}" >&2
    exit 3
fi

mkdir -p "${RESULTS_ROOT}" "${ANALYSIS_DIR}"

{
    printf "case\tjob_id\tsource\tnotes\tresult_dir\tjson\tgeometry_dir\tgeometry_fields\n"
} > "${MANIFEST}"

echo "============================================================"
echo "  Capped RV-EDP shared-L5 canonical sweep"
echo "  Started:      $(date)"
echo "  Cases:        ${CASES}"
echo "  Data dir:     ${DATA_DIR}"
echo "  Geometry:     ${GEOM_DIR}"
echo "  Fields:       ${GEOM_FIELDS}"
echo "  Results root: ${RESULTS_ROOT}"
echo "  Manifest:     ${MANIFEST}"
echo "  Dry run:      ${DRY_RUN}"
echo "============================================================"

JOB_IDS=()
FAILURES=0

for case_name in ${CASES}; do
    json="${DATA_DIR}/optimized_regazzoni_ukb_${case_name}.json"
    result_dir="${RESULTS_ROOT}/${case_name}"
    comment="capped_shared_l5_cap5_${case_name}_h5_6bt"

    if [ ! -f "${json}" ]; then
        echo "  [MISSING] ${json} -- skipping ${case_name}"
        printf "%s\t\tmissing\tmissing circulation JSON\t%s\t%s\t%s\t%s\n" \
            "${case_name}" "${result_dir}" "${json}" "${GEOM_DIR}" "${GEOM_FIELDS}" >> "${MANIFEST}"
        FAILURES=$((FAILURES + 1))
        continue
    fi

    echo "  Submitting ${case_name} -> ${result_dir}"
    if [ "${DRY_RUN}" = "1" ]; then
        jid="DRYRUN"
    else
        jid=$(sbatch --parsable \
            --job-name="cap5L5_${case_name}" \
            --time=10:00:00 \
            --export=ALL,MESH=UKB,BPM=75,BEATS=6,POST_FULL=0,RUN_POSTPROCESS=1,CHAR_LENGTH=5.0,METRICS_SPACE=DG1,CIRCULATION_PARAMS="${json}",GEOMETRY_DIR="${GEOM_DIR}",GEOMETRY_FIELDS="${GEOM_FIELDS}",RV_EDP_MAX_MMHG=5.0,PRE_CIRC_BEATS=30,PRE_CIRC_MAX_BEATS=80,PRE_CIRC_CONVERGENCE_TOL=0.002,RESULTS_DIR_OVERRIDE="${result_dir}",COMMENT="${comment}" \
            "${WORK_DIR}/sbatch/jobs/run_sim_and_post.sbatch")
    fi

    if [ -n "${jid}" ]; then
        JOB_IDS+=("${jid}")
        printf "%s\t%s\tcapped_shared_l5_submission_%s\tshared L5 geometry, RV EDP cap 5 mmHg, strict canonical postprocessing\t%s\t%s\t%s\t%s\n" \
            "${case_name}" "${jid}" "${STAMP}" "${result_dir}" "${json}" "${GEOM_DIR}" "${GEOM_FIELDS}" >> "${MANIFEST}"
        echo "    -> job ${jid}"
    else
        echo "    ! submission failed for ${case_name}"
        printf "%s\t\tsubmission_failed\tsbatch returned empty job id\t%s\t%s\t%s\t%s\n" \
            "${case_name}" "${result_dir}" "${json}" "${GEOM_DIR}" "${GEOM_FIELDS}" >> "${MANIFEST}"
        FAILURES=$((FAILURES + 1))
    fi
done

if [ "${#JOB_IDS[@]}" -gt 0 ] && [ "${DRY_RUN}" != "1" ]; then
    dep_ids=$(IFS=:; echo "${JOB_IDS[*]}")
    printf "%s\n" "${dep_ids}" > "${ANALYSIS_DIR}/job_dependency.txt"
    printf "%s\n" "${JOB_IDS[@]}" > "${ANALYSIS_DIR}/job_ids.txt"
fi

echo ""
echo "============================================================"
echo "  Submitted: ${#JOB_IDS[@]}"
echo "  Failures:  ${FAILURES}"
echo "  Job IDs:   ${JOB_IDS[*]:-<none>}"
echo "  Manifest:  ${MANIFEST}"
echo "  Finished:  $(date)"
echo "============================================================"
