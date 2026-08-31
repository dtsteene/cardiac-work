#!/usr/bin/env bash
# PAH pulmonary-windkessel batch: 8 linear-EDPVR circulation cases that differ
# ONLY in the pulmonary windkessel (R_AR_PUL up, C_AR_PUL down), run twice --
# once without Frank-Starling (constant Ta=100, the thesis model) and once with
# Frank-Starling (live g(lambda), Ta=200). 16 jobs total.
#
# Paradigm: ONE shared UKB L5 mesh + fibres + ONE shared inverse-unloaded
# reference (sPAP30 prestress_inverse.bp) for every job. Only the 0D circulation
# JSON and the active-stress model change. Unloading is never recomputed.
#
# Frank-Starling is selected purely by env (USE_FRANK_STARLING) -- no branch
# switch -- from one committed code state.

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
BATCH_DIR="${WORK_DIR}/pah_pulmonary_batch"
PARAM_DIR="${BATCH_DIR}/circ_params"
GEOM_DIR="${GEOM_DIR:-${WORK_DIR}/data/mesh_convergence/ukb_L5/ukb/geometry}"
GEOM_FIELDS="${GEOM_FIELDS:-${GEOM_DIR}/geometry_fields.npz}"

# Resolution / beat count. Defaults reproduce the canonical h=5, 6-beat sweep;
# override for a cheaper pilot (e.g. CHAR_LENGTH=10.0 BEATS=1 with the L10 mesh
# and the matching shared_unloaded_L10 reference).
BEATS="${BEATS:-6}"
CHAR_LENGTH="${CHAR_LENGTH:-5.0}"
# Job-name / comment prefix, so a non-pulmonary campaign is identifiable in squeue.
JOB_PREFIX="${JOB_PREFIX:-pahpulm}"
# Mesh tag used in the job comment: 5.0 -> h5, 10.0 -> h10
MESH_TAG="h${CHAR_LENGTH%%.*}"

# Char-5 shared inverse-unloaded reference, freshly rebuilt 2026-06-08 to a FIXED
# ED target (LV 7.77 / RV 5.00 mmHg) from baseline_linear_v2's 0D warm-up — one
# mesh for all cases (the old _CURRENT_H5_PRODUCTION/sPAP30 symlink was dangling).
DEFAULT_SHARED_UNLOADED="${WORK_DIR}/pah_pulmonary_batch/shared_unloaded_L5/ref/solver/prestress_inverse.bp"
SHARED_UNLOADED_FROM="${SHARED_UNLOADED_FROM:-${DEFAULT_SHARED_UNLOADED}}"
SHARED_UNLOADED_NOTE="${SHARED_UNLOADED_NOTE:-char-5 inverse-unloaded reference, fixed ED target LV7.77/RV5.00 mmHg (baseline_linear_v2); purely passive (Ta=0) so valid for both FS and no-FS bundles.}"

# 8 linear-EDPVR cases on baseline_linear_v2 (even in COUPLED RV systolic 25->95
# mmHg; PUL R up / C down at conserved RC~0.33; shared 8/5 unloaded reference)
CASES_GLOB="${CASES_GLOB:-${PARAM_DIR}/case*_rv*.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${WORK_DIR}/results/sims/$(date +%F)/pah_pulmonary_${STAMP}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${WORK_DIR}/results/analysis/pah_pulmonary_sweep_${STAMP}}"
MANIFEST="${MANIFEST:-${ANALYSIS_DIR}/pah_pulmonary_cases.tsv}"
DRY_RUN="${DRY_RUN:-0}"

# bundle definitions:  label | extra sbatch env
#   no_frank_starling       : USE_FRANK_STARLING=0, constant Ta=100 (thesis model)
#   frank_starling_preload  : g frozen at ED stretch, Ta=220 (effective ~ no-FS); the defensible static FS
#   frank_starling_relax    : activation-lag g, Ta=220, tau=250 ms (sensitivity arm, sits between preload and instantaneous)
# Override the set with BUNDLES_OVERRIDE="a b c" to fire a subset.
BUNDLES=(${BUNDLES_OVERRIDE:-no_frank_starling frank_starling_preload frank_starling_relax})

bundle_env() {  # echo the bundle-specific --export additions
    case "$1" in
        no_frank_starling)      echo "USE_FRANK_STARLING=0,TA_PEAK_KPA=100.0" ;;
        frank_starling_preload) echo "USE_FRANK_STARLING=1,TA_PEAK_KPA=220.0,FS_PRELOAD_ONLY=1" ;;
        frank_starling_relax)   echo "USE_FRANK_STARLING=1,TA_PEAK_KPA=220.0,FS_PRELOAD_ONLY=0,FS_RELAX_TAU_MS=250" ;;
    esac
}

# --- sanity checks ---
[ -d "${GEOM_DIR}/geometry.bp" ] || { echo "Missing geometry.bp: ${GEOM_DIR}/geometry.bp" >&2; exit 2; }
[ -f "${GEOM_FIELDS}" ]          || { echo "Missing geometry_fields.npz: ${GEOM_FIELDS}" >&2; exit 3; }
[ -e "${SHARED_UNLOADED_FROM}" ] || { echo "Missing shared unloaded reference: ${SHARED_UNLOADED_FROM}" >&2; exit 4; }
shopt -s nullglob
CASE_FILES=(${CASES_GLOB})
shopt -u nullglob
[ "${#CASE_FILES[@]}" -gt 0 ] || { echo "No case JSONs match ${CASES_GLOB}" >&2; exit 5; }

mkdir -p "${RESULTS_ROOT}" "${ANALYSIS_DIR}"
# A dry run must never touch a real campaign's manifest: re-running with an
# existing STAMP to preview a resubmission would otherwise truncate the record
# of the jobs actually submitted.
if [ "${DRY_RUN}" = "1" ]; then
    MANIFEST="${MANIFEST}.dryrun"
fi
printf "bundle\tcase\tjob_id\tresult_dir\tjson\tshared_unloaded_from\n" > "${MANIFEST}"

echo "============================================================"
echo "  ${CAMPAIGN_LABEL:-PAH pulmonary-windkessel sweep (linear EDPVR)}"
echo "  Started:          $(date)"
echo "  Cases (${#CASE_FILES[@]}):       $(for f in "${CASE_FILES[@]}"; do basename "$f" .json; done | tr '\n' ' ')"
echo "  Bundles:          ${BUNDLES[*]}"
echo "  Geometry:         ${GEOM_DIR}"
echo "  Resolution:       char_length=${CHAR_LENGTH} (${MESH_TAG})"
echo "  Beats:            ${BEATS}"
echo "  Fixed ratios:     LV=${FIXED_RATIO_LV:-<per-case>} RV=${FIXED_RATIO_RV:-<per-case>}"
echo "  Shared unloaded:  ${SHARED_UNLOADED_FROM}"
echo "  Results root:     ${RESULTS_ROOT}"
echo "  Manifest:         ${MANIFEST}"
echo "  Dry run:          ${DRY_RUN}"
echo "============================================================"

JOB_IDS=()
FAILURES=0

# Round-robin node pinning: char-5 FEM is memory-bandwidth-bound, so piling all
# jobs on one node (what timed out the previous attempt) starves every job.
# Cycle through NODES so ~equal counts land on each milanq node.
NODES_ARR=(${NODES:-n013 n014 n015 n016})
node_i=0
WALLTIME="${WALLTIME:-10:00:00}"

for bundle in "${BUNDLES[@]}"; do
    extra_env="$(bundle_env "${bundle}")"
    for json in "${CASE_FILES[@]}"; do
        case_name="$(basename "${json}" .json)"
        result_dir="${RESULTS_ROOT}/${bundle}/${case_name}"
        comment="${JOB_PREFIX}_${bundle}_${case_name}_${MESH_TAG}_${BEATS}bt_shared_unloaded"
        echo "  [${bundle}] ${case_name} -> ${result_dir}"

        if [ "${DRY_RUN}" = "1" ]; then
            jid="DRYRUN"
            echo "      (dry) export: ALL,MESH=UKB,BPM=75,BEATS=${BEATS},POST_FULL=0,RUN_POSTPROCESS=1,CHAR_LENGTH=${CHAR_LENGTH},METRICS_SPACE=DG1,${extra_env},CIRCULATION_PARAMS=${json},GEOMETRY_DIR=${GEOM_DIR},LOAD_UNLOADED_FROM=...,RESULTS_DIR_OVERRIDE=${result_dir}"
        else
            # Node pinning is optional: set NODES=none to let Slurm schedule freely
            # (used when moving to mi50q/habanaq, which have few but large nodes).
            if [ "${NODES:-}" = "none" ]; then
                NODELIST_ARG=()
            else
                pin_node="${NODES_ARR[$(( node_i % ${#NODES_ARR[@]} ))]}"
                node_i=$(( node_i + 1 ))
                NODELIST_ARG=(--nodelist="${pin_node}")
            fi
            jid=$(sbatch --parsable \
                --partition="${PARTITION:-milanq}" \
                "${NODELIST_ARG[@]}" \
                --signal=B:TERM@300 \
                --job-name="${JOB_PREFIX}_${bundle:0:4}_${case_name}" \
                --time="${WALLTIME}" \
                --export=ALL,MESH=UKB,BPM=75,BEATS=${BEATS},POST_FULL=0,RUN_POSTPROCESS=1,CHAR_LENGTH=${CHAR_LENGTH},METRICS_SPACE=DG1,${extra_env},FIXED_RATIO_LV="${FIXED_RATIO_LV:-}",FIXED_RATIO_RV="${FIXED_RATIO_RV:-}",CIRCULATION_PARAMS="${json}",GEOMETRY_DIR="${GEOM_DIR}",GEOMETRY_FIELDS="${GEOM_FIELDS}",LOAD_UNLOADED_FROM="${SHARED_UNLOADED_FROM}",PRE_CIRC_BEATS=30,PRE_CIRC_MAX_BEATS=80,PRE_CIRC_CONVERGENCE_TOL=0.002,RESULTS_DIR_OVERRIDE="${result_dir}",COMMENT="${comment}" \
                "${WORK_DIR}/sbatch/jobs/run_sim_and_post.sbatch")
            echo "      -> ${pin_node:-${PARTITION:-milanq}}, walltime ${WALLTIME}"
        fi

        if [ -n "${jid}" ]; then
            JOB_IDS+=("${jid}")
            printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${bundle}" "${case_name}" "${jid}" "${result_dir}" "${json}" "${SHARED_UNLOADED_FROM}" >> "${MANIFEST}"
            echo "      -> job ${jid}"
        else
            echo "      ! submission failed for ${bundle}/${case_name}"
            printf "%s\t%s\t\t%s\t%s\t%s\n" "${bundle}" "${case_name}" "${result_dir}" "${json}" "${SHARED_UNLOADED_FROM}" >> "${MANIFEST}"
            FAILURES=$((FAILURES + 1))
        fi
    done
done

if [ "${#JOB_IDS[@]}" -gt 0 ] && [ "${DRY_RUN}" != "1" ]; then
    printf "%s\n" "${JOB_IDS[@]}" > "${ANALYSIS_DIR}/job_ids.txt"
fi

echo ""
echo "============================================================"
echo "  Submitted: ${#JOB_IDS[@]}   Failures: ${FAILURES}"
echo "  Job IDs:   ${JOB_IDS[*]:-<none>}"
echo "  Manifest:  ${MANIFEST}"
echo "  Finished:  $(date)"
echo "============================================================"
