#!/bin/bash
# ==============================================================================
# FEM dispatcher for the v12 EXP extra 8 cases (sPAP25..sPAP92 interleaved)
# ==============================================================================
# Submit after the circ-optim extra job with:
#   sbatch --dependency=afterany:<extra_jid> submit_spectrum_v12_exp_extra.sbatch
#
# Submits a 6-beat FEM run per case with POST_FULL=1 (so regional internal
# work is computed and engineering_debug.png draws correctly).
# ==============================================================================

#SBATCH --job-name=fem_disp_ex
#SBATCH --partition=habanaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --output=fem_dispatch_exp_extra_%j.out
#SBATCH --error=fem_dispatch_exp_extra_%j.err

WORK_DIR="/global/D1/homes/dtsteene/cardiac-work"
DATA_DIR="${WORK_DIR}/data/ukb_circ_v12_exp"
CASES="sPAP25 sPAP35 sPAP50 sPAP60 sPAP70 sPAP80 sPAP87 sPAP92"

cd "$WORK_DIR"

echo "============================================================"
echo "  FEM spectrum dispatch (v12 EXP extra, 6 beats)"
echo "  Started : $(date)"
echo "  Data dir: $DATA_DIR"
echo "  POST_FULL=1"
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
CIRCULATION_PARAMS="$JSON",COMMENT="spectrum_v12exp_extra_${sev}_6bt" \
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
