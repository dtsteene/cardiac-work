#!/bin/bash
# ==============================================================================
# Slurm job: Optimize 0D circulation parameters for a single severity level
# Usage:
#   sbatch --export=SEVERITY=healthy run_circ_optim.sbatch
#   sbatch --export=SEVERITY=mild,N_TRIALS=5000 run_circ_optim.sbatch
#
# Submit all severities at once with the companion script:
#   bash submit_all_circ_optim.sh
# ==============================================================================

#SBATCH --job-name=circ_optim
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --output=circ_optim_%j.out
#SBATCH --error=circ_optim_%j.err

# --- Defaults ---
SEVERITY=${SEVERITY:-"healthy"}
N_TRIALS=${N_TRIALS:-2500}
MESH=${MESH:-"ukb"}
SEED=${SEED:-42}
# UKB mesh cavity volumes (pre-computed, avoids needing MPI on compute node)
LV_EDV=${LV_EDV:-111.46}
RV_EDV=${RV_EDV:-76.83}
MESH_NAME=${MESH_NAME:-"ukb"}

# --- Environment ---
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV

WORK_DIR="/global/D1/homes/dtsteene/cardiac-work"

# Output goes next to the optimization script (results_mesh_<name>/ under WORK_DIR)
cd "$WORK_DIR"

echo "============================================================"
echo "  Circulation parameter optimization"
echo "  Severity : $SEVERITY"
echo "  LV EDV   : $LV_EDV mL"
echo "  RV EDV   : $RV_EDV mL"
echo "  Mesh name: $MESH_NAME"
echo "  Trials   : $N_TRIALS"
echo "  Seed     : $SEED"
echo "  Node     : $(hostname)"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Started  : $(date)"
echo "============================================================"

# --- Run optimization (use --lv-edv/--rv-edv to skip MPI mesh reading) ---
python "$WORK_DIR/optimize_mesh_circ.py" \
    --lv-edv "$LV_EDV" \
    --rv-edv "$RV_EDV" \
    --mesh-name "$MESH_NAME" \
    --severity "$SEVERITY" \
    --n-trials "$N_TRIALS" \
    --seed "$SEED"

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  Finished : $(date)"
echo "  Exit code: $EXIT_CODE"
echo "============================================================"

# --- Copy results to cardiac-work data directory ---
if [ $EXIT_CODE -eq 0 ]; then
    RESULTS_DIR="$WORK_DIR/results_mesh_${MESH_NAME}"
    TARGET_DIR="$WORK_DIR/data/ukb_circ_v2"
    mkdir -p "$TARGET_DIR"

    for f in "$RESULTS_DIR"/optimized_regazzoni_${MESH_NAME}_${SEVERITY}.json \
             "$RESULTS_DIR"/pv_evaluation_${MESH_NAME}_${SEVERITY}.png; do
        if [ -f "$f" ]; then
            cp "$f" "$TARGET_DIR/"
            echo "Copied: $(basename $f) -> $TARGET_DIR/"
        fi
    done

    # Quick sanity check: print EF from the result
    python -c "
import json, sys
f = '$TARGET_DIR/optimized_regazzoni_${MESH_NAME}_${SEVERITY}.json'
try:
    d = json.load(open(f))
    h = d.get('derived_hemodynamics', {})
    m = d.get('metrics_achieved', {})
    print()
    print('=== QUICK SANITY CHECK ===')
    print(f'  LV EF  : {h.get(\"LV_EF_pct\", \"?\"):.1f}%')
    print(f'  RV EF  : {h.get(\"RV_EF_pct\", \"?\"):.1f}%')
    print(f'  CO     : {h.get(\"CO_Lpm\", \"?\"):.2f} L/min')
    print(f'  SV     : {m.get(\"SV\", \"?\"):.1f} mL')
    print(f'  LV ESP : {m.get(\"LV_ESP\", \"?\"):.1f} mmHg')
    print(f'  RV ESP : {m.get(\"RV_ESP\", \"?\"):.1f} mmHg')
except Exception as e:
    print(f'Could not read results: {e}', file=sys.stderr)
"
fi

exit $EXIT_CODE
