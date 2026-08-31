#!/usr/bin/env bash
# Softer-material pilot (2026-07-08): gauge whether reducing passive stiffness
# opens up LV (and RV/septum) dynamic range in the pure-PAH sweep.
#
# Motivation: in the fixed-ratio sweep the LV end-diastolic fiber stretch varies
# only ~0.4% across the 8 afterload cases, so Frank-Starling (a gain on ED stretch)
# has essentially nothing to amplify. Softer material is the only proposed lever
# that could grow the *signal* (larger stretch, more preload sensitivity). This
# pilot measures that directly on cheap L10 single-beat runs before committing.
#
# Matrix: scale {1.00 control, 0.50, 0.33} x case {baseline rv25, severe rv95}
#         = 6 jobs. Whole-heart softening (LV=RV=Septum). FS-preload bundle.
#         L10 mesh, 1 beat. All runs share the sweep's baseline-anchored fixed
#         ratio so the ONLY variable across a case pair is passive stiffness.
#
# Usage:  bash pah_pulmonary_batch/submit_softmat_pilot.sh          # submit
#         DRY=1 bash pah_pulmonary_batch/submit_softmat_pilot.sh    # print only
set -euo pipefail

SIM_DIR="${SIM_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export SIM_DIR
cd "$SIM_DIR"

GEOM_DIR="${SIM_DIR}/data/mesh_convergence/ukb_L10/ukb/geometry"
GEOM_FIELDS="${GEOM_DIR}/geometry_fields.npz"
UNLOADED="${SIM_DIR}/pah_pulmonary_batch/shared_unloaded_L10/ref/solver/prestress_inverse.bp"
# Baseline-anchored coupling ratio from the fixed-ratio sweep (case0, scale 1.0).
# Identical for every pilot run so preload spread across cases is preserved, not clamped.
FIXED_RATIO_LV="1.02479"
FIXED_RATIO_RV="0.88262"
# frank_starling_preload bundle (the group's preferred active model).
BUNDLE_ENV="USE_FRANK_STARLING=1,TA_PEAK_KPA=220.0,FS_PRELOAD_ONLY=1"

OUT_ROOT="${SIM_DIR}/results/sims/2026-07-08/softmat_pilot_L10"

for f in "${GEOM_DIR}/geometry.bp" "$GEOM_FIELDS" "$UNLOADED"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 2; }
done

declare -A SCALE_TAG=( [1.00]=100 [0.50]=050 [0.33]=033 )
CASES=( case0_rv25 case7_rv95 )

for scale in 1.00 0.50 0.33; do
    tag="${SCALE_TAG[$scale]}"
    for case in "${CASES[@]}"; do
        json="${SIM_DIR}/pah_pulmonary_batch/circ_params/${case}.json"
        [ -e "$json" ] || { echo "MISSING circ json: $json" >&2; exit 2; }
        result_dir="${OUT_ROOT}/scale${tag}/${case}"
        comment="softmat_pilot_L10_scale${tag}_${case}"
        export_str="ALL,MESH=UKB,BPM=75,BEATS=1,POST_FULL=0,RUN_POSTPROCESS=1,CHAR_LENGTH=10.0,METRICS_SPACE=DG1,${BUNDLE_ENV},LV_MATERIAL_SCALE=${scale},RV_MATERIAL_SCALE=${scale},SEPTUM_MATERIAL_SCALE=${scale},FIXED_RATIO_LV=${FIXED_RATIO_LV},FIXED_RATIO_RV=${FIXED_RATIO_RV},CIRCULATION_PARAMS=${json},GEOMETRY_DIR=${GEOM_DIR},GEOMETRY_FIELDS=${GEOM_FIELDS},LOAD_UNLOADED_FROM=${UNLOADED},PRE_CIRC_BEATS=30,PRE_CIRC_MAX_BEATS=80,PRE_CIRC_CONVERGENCE_TOL=0.002,RESULTS_DIR_OVERRIDE=${result_dir},COMMENT=${comment}"
        if [ "${DRY:-0}" = "1" ]; then
            echo "(dry) scale=${scale} ${case} -> ${result_dir}"
        else
            jid=$(sbatch --parsable --job-name="softmat_${tag}_${case}" \
                  --export="${export_str}" sbatch/jobs/run_sim_and_post.sbatch)
            echo "submitted scale=${scale} ${case}  job=${jid}  -> ${result_dir}"
        fi
    done
done
