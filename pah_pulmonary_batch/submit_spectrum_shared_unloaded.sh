#!/usr/bin/env bash
# PAH severity spectrum re-run with ONE shared inverse-unloaded reference.
#
# Why: the original spectrum campaign (figures archived under
# results/_thesis_reference_2026-05-04/analysis/spectrum/) gave every case its
# OWN inverse-unloaded reference. That re-normalises each case back to the mesh
# ED volume, hides the across-case preload spread, and is what produced the
# superseded "transmural is best" septal result. This re-run holds the reference
# geometry fixed across all seven cases so the FEM feels the real preload spread.
#
# Circulation params: data/ukb_circ_v2/ -- confirmed as the set behind the
# original spectrum (all 7 metrics_achieved.RV_ESP match spectrum_raw.npz to
# <0.05 mmHg). Symlinked in severity order under spectrum_v2_params/.
#
# Unlike the pulmonary fixed-ratio sweep (LV loading held constant), this set
# varies BOTH ventricles: RV_ESP 30.6 -> 88.4 mmHg while LV_ESP falls
# 120.4 -> 91.5. That anti-correlation partially breaks the P_LV/P_RV
# collinearity that stops Pearson r from separating the proxy candidates.
#
# Usage:
#   ./submit_spectrum_shared_unloaded.sh pilot        # 1 beat, h=10, no-FS
#   ./submit_spectrum_shared_unloaded.sh production   # 6 beats, h=5, no-FS
# Prefix with DRY_RUN=1 to print the sbatch lines without submitting.
#
# To resubmit a SUBSET into an existing campaign directory (e.g. after jobs
# were starved by node pinning), pass CASES_GLOB as space-separated paths plus
# the original STAMP, and NODES=none to let Slurm schedule freely:
#   NODES=none STAMP=<original> \
#     CASES_GLOB="$PWD/spectrum_v2_params/sev0_healthy.json $PWD/..." \
#     ./submit_spectrum_shared_unloaded.sh pilot
# Cases already completed must be EXCLUDED -- a resubmitted case overwrites
# its results directory.

set -euo pipefail

STAGE="${1:-}"
BATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORK_DIR="$(cd "${BATCH_DIR}/.." && pwd -P)"

case "${STAGE}" in
    pilot)
        RES_TAG="L10";  export CHAR_LENGTH=10.0; export BEATS=1
        # Coupling ratios pinned to the L10 shared reference's own values, so the
        # anchor is the reference geometry rather than any single case.
        export FIXED_RATIO_LV=0.9989672019420158
        export FIXED_RATIO_RV=0.8573073293664611
        export WALLTIME="${WALLTIME:-2:00:00}"
        ;;
    production)
        RES_TAG="L5";   export CHAR_LENGTH=5.0;  export BEATS=6
        export FIXED_RATIO_LV=1.0229115525247285
        export FIXED_RATIO_RV=0.8862529544840637
        export WALLTIME="${WALLTIME:-10:00:00}"
        ;;
    *)
        echo "usage: $0 {pilot|production}" >&2; exit 1 ;;
esac

export GEOM_DIR="${WORK_DIR}/data/mesh_convergence/ukb_${RES_TAG}/ukb/geometry"
export GEOM_FIELDS="${GEOM_DIR}/geometry_fields.npz"
export SHARED_UNLOADED_FROM="${BATCH_DIR}/shared_unloaded_${RES_TAG}/ref/solver/prestress_inverse.bp"
export SHARED_UNLOADED_NOTE="char-${RES_TAG} shared inverse-unloaded reference, fixed ED target LV7.77/RV5.00 mmHg (baseline_linear_v2); one reference for all 7 spectrum cases."
export CASES_GLOB="${CASES_GLOB:-${BATCH_DIR}/spectrum_v2_params/sev*.json}"
export BUNDLES_OVERRIDE="${BUNDLES_OVERRIDE:-no_frank_starling}"
export JOB_PREFIX="spec${RES_TAG}"
export CAMPAIGN_LABEL="PAH severity spectrum, SHARED unloaded reference (${STAGE}: ${RES_TAG}, ${BEATS} beat(s))"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export STAMP
export RESULTS_ROOT="${RESULTS_ROOT:-${WORK_DIR}/results/sims/$(date +%F)/spectrum_shared_unloaded_${STAGE}_${STAMP}}"
export ANALYSIS_DIR="${ANALYSIS_DIR:-${WORK_DIR}/results/analysis/spectrum_shared_unloaded_${STAGE}_${STAMP}}"

exec "${BATCH_DIR}/submit_pah_pulmonary_sweep.sh"
