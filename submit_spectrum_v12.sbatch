#!/bin/bash
# ==============================================================================
# Auto-submit the FEM spectrum after v12 circulation optimisation finishes
# ==============================================================================
# Submit with:
#   sbatch --dependency=afterany:<v12_jobid> submit_spectrum_v12.sbatch
#
# This is a tiny dispatcher job: it waits for v12 to complete (any exit
# status), then checks each v12 output JSON exists and submits one
# run_sim_and_post.sbatch per case. If a JSON is missing (e.g. one Optuna
# run failed), that case is skipped — the rest still go through.
#
# Each submitted FEM job runs on mi50q, 8 MPI ranks, 6 beats.
# ==============================================================================

#SBATCH --job-name=fem_dispatch
#SBATCH --partition=habanaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --output=fem_dispatch_%j.out
#SBATCH --error=fem_dispatch_%j.err

WORK_DIR="/global/D1/homes/dtsteene/cardiac-work"
DATA_DIR="${WORK_DIR}/data/ukb_circ_v12_linear"
CASES="sPAP22 sPAP30 sPAP45 sPAP55 sPAP65 sPAP75 sPAP85 sPAP95"

cd "$WORK_DIR"

echo "============================================================"
echo "  FEM spectrum dispatch (v12 circulation params, 6 beats)"
echo "  Started : $(date)"
echo "  Data dir: $DATA_DIR"
echo "============================================================"

JOB_IDS=()
MISSING=0

for sev in $CASES; do
    JSON="$DATA_DIR/optimized_regazzoni_ukb_linear_${sev}.json"
    if [ ! -f "$JSON" ]; then
        echo "  [MISSING] $JSON — skipping $sev"
        MISSING=$((MISSING+1))
        continue
    fi

    echo "  Submitting FEM $sev"
    JID=$(sbatch --parsable \
        --export=ALL,MESH=UKB,BEATS=6,CIRCULATION_PARAMS="$JSON",COMMENT="spectrum_v12_${sev}_6bt" \
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
echo ""
echo "Monitor with: squeue -u \$USER"
