#!/bin/bash
# ==============================================================================
# v12 EXP extra — 8 interleaved sPAP cases to bring the spectrum to n = 16
# ==============================================================================
# Cases: sPAP25, 35, 50, 60, 70, 80, 87, 92 (interleaved with the existing
# sPAP22/30/45/55/65/75/85/95 so the final n=16 sweep samples the spectrum
# more densely). Same optimiser, same targets, same exponential EDPVR as
# run_circ_optim_v12_exp.sbatch.
#
# Outputs added to the existing data/ukb_circ_v12_exp/ alongside the first 8.
# ==============================================================================

#SBATCH --job-name=circ_v12e_ex
#SBATCH --partition=habanaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=6:00:00
#SBATCH --output=circ_v12_exp_extra_%j.out
#SBATCH --error=circ_v12_exp_extra_%j.err

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

N_TRIALS=${N_TRIALS:-3000}
SEED=${SEED:-42}
LV_EDV=${LV_EDV:-111.46}
RV_EDV=${RV_EDV:-76.83}
MESH_NAME=${MESH_NAME:-"ukb"}
BSA=${BSA:-1.75}

CASES="sPAP25 sPAP35 sPAP50 sPAP60 sPAP70 sPAP80 sPAP87 sPAP92"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV

WORK_DIR="/global/D1/homes/dtsteene/cardiac-work"
cd "$WORK_DIR"

echo "============================================================"
echo "  v12 EXP extra — 8 interleaved sPAP cases"
echo "  Node     : $(hostname) [$SLURM_CPUS_PER_TASK cores allocated]"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Started  : $(date)"
echo "  Trials   : $N_TRIALS per optimization"
echo "  Cases    : $CASES"
echo "============================================================"

mkdir -p logs_v12_exp_extra
PIDS=()
for SEV in $CASES; do
    LOG="logs_v12_exp_extra/exp_${SEV}.log"
    echo "Launching EXP $SEV → $LOG"
    python "$WORK_DIR/optimize_mesh_circ.py" \
        --lv-edv "$LV_EDV" \
        --rv-edv "$RV_EDV" \
        --mesh-name "$MESH_NAME" \
        --severity "$SEV" \
        --n-trials "$N_TRIALS" \
        --seed "$SEED" \
        --bsa "$BSA" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} jobs; waiting..."
FAILED=0
for PID in "${PIDS[@]}"; do wait "$PID" || FAILED=$((FAILED+1)); done

echo ""
echo "All optimisations finished. Failed (non-zero exit): $FAILED"
echo "NOTE: non-zero exit from evaluate_and_plot diagnostics does not imply"
echo "a failed JSON. Check data directory contents below."

# Copy into the existing v12_exp data directory
SRC="$WORK_DIR/results_mesh_${MESH_NAME}"
DST="$WORK_DIR/data/ukb_circ_v12_exp"
mkdir -p "$DST"

for SEV in $CASES; do
    for ext in json png; do
        FN="optimized_regazzoni_${MESH_NAME}_${SEV}.${ext}"
        [ -f "$SRC/$FN" ] && cp "$SRC/$FN" "$DST/" && echo "copied $FN"
        FN="pv_evaluation_${MESH_NAME}_${SEV}.${ext}"
        [ -f "$SRC/$FN" ] && cp "$SRC/$FN" "$DST/" && echo "copied $FN"
    done
done

echo ""
echo "Finished: $(date)"
exit $FAILED
