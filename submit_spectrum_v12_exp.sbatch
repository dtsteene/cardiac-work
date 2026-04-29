#!/bin/bash
# ==============================================================================
# Auto-submit the FEM spectrum after v12 EXP circulation optimisation finishes
# ==============================================================================
# Submit with:
#   sbatch --dependency=afterany:<v12e_jobid> submit_spectrum_v12_exp.sbatch
#
# Identical logic to submit_spectrum_v12.sbatch, but:
#  - reads circulation JSONs from data/ukb_circ_v12_exp/ (EXP filenames,
#    no _linear suffix)
#  - passes POST_FULL=1 so postprocess_metrics runs with
#    --no-skip-regional-internal, restoring the engineering_debug.png
# ==============================================================================

#SBATCH --job-name=fem_disp_exp
#SBATCH --partition=habanaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --output=fem_dispatch_exp_%j.out
#SBATCH --error=fem_dispatch_exp_%j.err

WORK_DIR="/global/D1/homes/dtsteene/cardiac-work"
DATA_DIR="${WORK_DIR}/data/ukb_circ_v12_exp"
CASES="sPAP22 sPAP30 sPAP45 sPAP55 sPAP65 sPAP75 sPAP85 sPAP95"

cd "$WORK_DIR"

echo "============================================================"
echo "  FEM spectrum dispatch (v12 EXP circulation params, 6 beats)"
echo "  Started : $(date)"
echo "  Data dir: $DATA_DIR"
echo "  POST_FULL=1 (regional internal work enabled)"
echo "============================================================"

JOB_IDS=()
MISSING=0

for sev in $CASES; do
    JSON="$DATA_DIR/optimized_regazzoni_ukb_${sev}.json"
    if [ ! -f "$JSON" ]; then
        echo "  [MISSING] $JSON — skipping $sev"
        MISSING=$((MISSING+1))
        continue
    fi

    echo "  Submitting FEM $sev"
    JID=$(sbatch --parsable \
        --export=ALL,MESH=UKB,BEATS=6,POST_FULL=1,\
CIRCULATION_PARAMS="$JSON",COMMENT="spectrum_v12exp_${sev}_6bt" \
        run_sim_and_post.sbatch)
    if [ -n "$JID" ]; then
        JOB_IDS+=("$JID")
        echo "    → job $JID"
    else
        echo "    ! submission failed for $sev"
        MISSING=$((MISSING+1))
    fi
done

echo ""
echo "============================================================"
echo "  Submitted ${#JOB_IDS[@]} FEM jobs, $MISSING missing / failed"
echo "  Job IDs : ${JOB_IDS[@]}"
echo "  Finished: $(date)"
echo "============================================================"
