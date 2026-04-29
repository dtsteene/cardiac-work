#!/bin/bash
# ==============================================================================
# Slurm job: LINEAR EDPVR sanity-check — same as run_circ_optim.sbatch but
# with the --linear flag so kE is held at 0 for the entire search.
#
# Purpose:
#   Test whether the stock linear Regazzoni passive elastance can reach the
#   same 7 severity targets that the exponential EDPVR reaches. Results land
#   in data/ukb_circ_linear/ so they don't clobber ukb_circ_v2/.
#
# Usage:
#   sbatch --export=SEVERITY=healthy run_circ_optim_linear.sbatch
#   sbatch --export=SEVERITY=end_stage,N_TRIALS=5000 run_circ_optim_linear.sbatch
# Companion submitter: bash submit_all_circ_optim_linear.sh
# ==============================================================================

#SBATCH --job-name=circ_lin
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --output=circ_optim_linear_%j.out
#SBATCH --error=circ_optim_linear_%j.err

# Prevent numpy/scipy BLAS from spawning worker threads that thrash the node
# when multiple jobs share resources. Pin each process to 1 thread — CMA-ES
# is single-threaded and the 0D integrator doesn't benefit from BLAS parallelism
# at this problem size.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# --- Defaults ---
SEVERITY=${SEVERITY:-"healthy"}
N_TRIALS=${N_TRIALS:-2000}
SEED=${SEED:-42}
# UKB mesh cavity volumes (same as run_circ_optim.sbatch to stay apples-to-apples)
LV_EDV=${LV_EDV:-111.46}
RV_EDV=${RV_EDV:-76.83}
MESH_NAME=${MESH_NAME:-"ukb"}          # --linear appends "_linear" internally

# --- Environment ---
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV

WORK_DIR="/global/D1/homes/dtsteene/cardiac-work"
cd "$WORK_DIR"

echo "============================================================"
echo "  Linear-only circulation optimization (--linear)"
echo "  Severity : $SEVERITY"
echo "  LV EDV   : $LV_EDV mL"
echo "  RV EDV   : $RV_EDV mL"
echo "  Mesh name: $MESH_NAME (becomes ${MESH_NAME}_linear inside script)"
echo "  Trials   : $N_TRIALS"
echo "  Seed     : $SEED"
echo "  Node     : $(hostname)"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Started  : $(date)"
echo "============================================================"

python "$WORK_DIR/optimize_mesh_circ.py" \
    --lv-edv "$LV_EDV" \
    --rv-edv "$RV_EDV" \
    --mesh-name "$MESH_NAME" \
    --severity "$SEVERITY" \
    --n-trials "$N_TRIALS" \
    --seed "$SEED" \
    --linear

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  Finished : $(date)"
echo "  Exit code: $EXIT_CODE"
echo "============================================================"

# --- Copy results to a separate linear directory ---
if [ $EXIT_CODE -eq 0 ]; then
    RESULTS_DIR="$WORK_DIR/results_mesh_${MESH_NAME}_linear"
    TARGET_DIR="$WORK_DIR/data/ukb_circ_linear"
    mkdir -p "$TARGET_DIR"

    JSON="$RESULTS_DIR/optimized_regazzoni_${MESH_NAME}_linear_${SEVERITY}.json"
    PNG="$RESULTS_DIR/pv_evaluation_${MESH_NAME}_linear_${SEVERITY}.png"
    for f in "$JSON" "$PNG"; do
        if [ -f "$f" ]; then
            cp "$f" "$TARGET_DIR/"
            echo "Copied: $(basename $f) -> $TARGET_DIR/"
        fi
    done

    # Quick sanity check: print achieved hemodynamics
    python -c "
import json, sys
f = '$TARGET_DIR/optimized_regazzoni_${MESH_NAME}_linear_${SEVERITY}.json'
try:
    d = json.load(open(f))
    h = d.get('derived_hemodynamics', {})
    m = d.get('metrics_achieved', {})
    t = d.get('pressure_targets', {})
    print()
    print('=== QUICK SANITY CHECK (LINEAR) ===')
    print(f'  LV EF  : {h.get(\"LV_EF_pct\", 0):.1f}%')
    print(f'  RV EF  : {h.get(\"RV_EF_pct\", 0):.1f}%')
    print(f'  CO     : {h.get(\"CO_Lpm\", 0):.2f} L/min')
    print(f'  SV     : {m.get(\"SV\", 0):.1f} mL')
    print(f'  LV ESP : {m.get(\"LV_ESP\", 0):.1f} mmHg   (target {t.get(\"LV_ESP\", 0)})')
    print(f'  LV EDP : {m.get(\"LV_EDP\", 0):.1f} mmHg   (target {t.get(\"LV_EDP\", 0)})')
    print(f'  RV ESP : {m.get(\"RV_ESP\", 0):.1f} mmHg   (target {t.get(\"RV_ESP\", 0)})')
    print(f'  RV EDP : {m.get(\"RV_EDP\", 0):.1f} mmHg   (target {t.get(\"RV_EDP\", 0)})')
except Exception as e:
    print(f'Could not read results: {e}', file=sys.stderr)
"
fi

exit $EXIT_CODE
