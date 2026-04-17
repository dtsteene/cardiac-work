#!/bin/bash
# --- Slurm Job Configuration ---
#SBATCH --job-name=cardiac_sim_milanq
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# --- Race check against a paired habanaq job (if any) ---
# Environment:
#   HABANAQ_PAIR_ID  — job id on habanaq that this job is "racing". Required.
#
# Behaviour at the moment this milanq job starts RUNNING:
#   habanaq pair state == RUNNING  → self-cancel (habanaq got there first)
#   habanaq pair state == PENDING  → scancel the habanaq pair, run here
#   habanaq pair state empty/other → self-cancel (something ran already, or
#                                     the pair was cancelled externally)
if [ -z "$HABANAQ_PAIR_ID" ]; then
    echo "ERROR: HABANAQ_PAIR_ID not set — refusing to run to avoid duplication"
    exit 1
fi

PAIR_STATE=$(squeue -j $HABANAQ_PAIR_ID -h -o "%T" 2>/dev/null | tr -d ' ')
echo "--- Race check ---"
echo "  This milanq job: $SLURM_JOB_ID"
echo "  Paired habanaq job: $HABANAQ_PAIR_ID"
echo "  Pair state: '${PAIR_STATE:-<not in queue>}'"

case "$PAIR_STATE" in
    RUNNING)
        echo "  → habanaq got the spot first. Self-cancelling this milanq job."
        scancel $SLURM_JOB_ID
        exit 0
        ;;
    PENDING)
        echo "  → habanaq still pending. Cancelling the habanaq pair and running here."
        scancel $HABANAQ_PAIR_ID
        ;;
    "")
        echo "  → habanaq pair not in queue (already done or cancelled)."
        echo "    Self-cancelling to avoid an unpaired duplicate run."
        scancel $SLURM_JOB_ID
        exit 0
        ;;
    *)
        echo "  → habanaq pair in unexpected state '$PAIR_STATE'. Cancelling it and running here."
        scancel $HABANAQ_PAIR_ID
        ;;
esac
echo "--- Race check done — proceeding ---"
echo ""

# --- Now run the main sim-and-post logic from run_sim_and_post.sbatch ---
# We bash-exec it: the #SBATCH directives inside are only honoured at
# submit time, so at runtime they are just comments and everything runs
# inside THIS job's allocation. All env vars (MESH, BEATS, CIRCULATION_PARAMS,
# GEOMETRY_DIR, GEOMETRY_FIELDS, COMMENT, etc.) are inherited.

bash /home/dtsteene/D1/cardiac-work/run_sim_and_post.sbatch
