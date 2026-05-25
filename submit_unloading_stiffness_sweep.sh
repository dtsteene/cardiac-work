#!/bin/bash
# Submit an unloading-only regional passive stiffness calibration sweep.
#
# This is intentionally a dispatcher only. Each mechanics solve runs through
# run_sim_and_post.sbatch on a Slurm compute node with the existing conda/MPI
# launcher pattern.

set -euo pipefail

WORK_DIR="/global/D1/homes/dtsteene/cardiac-work"
DATA_DIR="${WORK_DIR}/data/ukb_circ_v12_exp"
GEOMETRY_DIR="${WORK_DIR}/data/mesh_convergence/ukb_L5/ukb/geometry"

PARTITION="${PARTITION:-mi50q}"
NTASKS="${NTASKS:-8}"
TIME_LIMIT="${TIME_LIMIT:-2:00:00}"
CASES="${CASES:-sPAP22 sPAP60 sPAP95}"
RV_SCALES="${RV_SCALES:-2 4 8 12 16}"
SEPTUM_SCALES="${SEPTUM_SCALES:-1 2 4 8}"
PRE_CIRC_BEATS="${PRE_CIRC_BEATS:-10}"
PRE_CIRC_MAX_BEATS="${PRE_CIRC_MAX_BEATS:-10}"

STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="${WORK_DIR}/results/unloading_stiffness_sweep_${STAMP}.tsv"

cd "${WORK_DIR}"
mkdir -p "$(dirname "${MANIFEST}")"
printf "timestamp\tcase\trv_scale\tseptum_scale\tvariant\tjob_id\tpartition\tntasks\ttime_limit\tjson\tcomment\n" > "${MANIFEST}"

fmt_scale() {
    printf "%s" "$1" | sed 's/[.]/p/g'
}

submit_one() {
    local case_name="$1"
    local rv_scale="$2"
    local septum_scale="$3"
    local json="${DATA_DIR}/optimized_regazzoni_ukb_${case_name}.json"

    if [ ! -f "${json}" ]; then
        echo "[MISSING] ${json}; skipping ${case_name}/RV=${rv_scale}/Septum=${septum_scale}"
        return 0
    fi

    local rv_tag
    local septum_tag
    rv_tag="$(fmt_scale "${rv_scale}")"
    septum_tag="$(fmt_scale "${septum_scale}")"

    local variant="rv${rv_tag}_sep${septum_tag}"
    local comment="unloading_calib_${case_name}_${variant}"
    local export_vars="ALL,MESH=UKB,BPM=75,BEATS=1,POST_FULL=0,STOP_AFTER_UNLOADING=1,CHAR_LENGTH=5.0,METRICS_SPACE=DG1,CIRCULATION_PARAMS=${json},GEOMETRY_DIR=${GEOMETRY_DIR},COMMENT=${comment},PRE_CIRC_BEATS=${PRE_CIRC_BEATS},PRE_CIRC_MAX_BEATS=${PRE_CIRC_MAX_BEATS},RV_MATERIAL_SCALE=${rv_scale},SEPTUM_MATERIAL_SCALE=${septum_scale}"

    echo "Submitting ${case_name}/${variant}"
    local job_id
    job_id=$(sbatch --parsable \
        --job-name="unldcal_${case_name}_${variant}" \
        --partition="${PARTITION}" \
        --time="${TIME_LIMIT}" \
        --ntasks="${NTASKS}" \
        --export="${export_vars}" \
        run_sim_and_post.sbatch)

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date -Is)" "${case_name}" "${rv_scale}" "${septum_scale}" \
        "${variant}" "${job_id}" "${PARTITION}" "${NTASKS}" \
        "${TIME_LIMIT}" "${json}" "${comment}" >> "${MANIFEST}"
    echo "  -> ${job_id}"
}

for case_name in ${CASES}; do
    for rv_scale in ${RV_SCALES}; do
        for septum_scale in ${SEPTUM_SCALES}; do
            submit_one "${case_name}" "${rv_scale}" "${septum_scale}"
        done
    done
done

echo
echo "Manifest: ${MANIFEST}"
echo "Submitted jobs:"
awk 'NR>1 {print "  " $6 "\t" $2 "/" $5}' "${MANIFEST}"
