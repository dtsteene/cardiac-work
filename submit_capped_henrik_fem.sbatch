#!/bin/bash
# ==============================================================================
# FEM dispatcher — CAPPED-RVEDP variant of Henrik's simple PUL sweep
# ==============================================================================
# Counterpart to submit_henrik_simple_fem.sbatch, with RV_EDP_MAX_MMHG=5.0
# applied during inverse unloading. Same five Regazzoni-defaults parameter
# sets (pulmonary C_AR/R_AR scaled by single factor f ∈ {1.00,0.75,0.50,
# 0.25,0.10}); same coupled-FEM driver; same 6 beats with POST_FULL=1.
#
# Purpose: close the 2×2 design (calibration architecture × cap). If the
# simple uncapped henrik sweep gives transmural correlation r≈+0.28 and
# the capped version gives transmural correlation that matches the
# production-capped pattern (negative or near-zero), the cap is shown to
# be the operative variable independent of calibration architecture.
# ==============================================================================

#SBATCH --job-name=fem_capdsph
#SBATCH --partition=habanaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --output=fem_dispatch_capped_henrik_%j.out
#SBATCH --error=fem_dispatch_capped_henrik_%j.err

WORK_DIR="/global/D1/homes/dtsteene/cardiac-work"
DATA_DIR="${WORK_DIR}/data/henrik_simple_pul"
CASES="pul_f100 pul_f075 pul_f050 pul_f025 pul_f010"
CAP_MMHG="${CAP_MMHG:-5.0}"

cd "$WORK_DIR"

echo "============================================================"
echo "  FEM dispatch — CAPPED henrik simple PUL sweep"
echo "  Cap RV_EDP_MAX_MMHG = $CAP_MMHG"
echo "  Started : $(date)"
echo "  POST_FULL=1"
echo "============================================================"

JOB_IDS=()
for sev in $CASES; do
    JSON="$DATA_DIR/henrik_simple_pul_${sev}.json"
    [ ! -f "$JSON" ] && { echo "MISSING: $JSON"; continue; }
    echo "  Submitting FEM $sev (cap=$CAP_MMHG mmHg)"
    JID=$(sbatch --parsable \
        --export=ALL,MESH=UKB,BEATS=6,POST_FULL=1,RV_EDP_MAX_MMHG="$CAP_MMHG",\
CIRCULATION_PARAMS="$JSON",COMMENT="capped_henrik_simple_${sev}_6bt_cap${CAP_MMHG%.*}mmHg" \
        run_sim_and_post.sbatch)
    [ -n "$JID" ] && { JOB_IDS+=("$JID"); echo "    → $JID"; }
done

echo ""
echo "Submitted ${#JOB_IDS[@]} FEM jobs: ${JOB_IDS[@]}"
echo "Finished: $(date)"
