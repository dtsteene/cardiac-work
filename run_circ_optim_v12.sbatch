#!/bin/bash
# ==============================================================================
# v12 circulation optimization — two-sided RV_EF / CI penalties (LINEAR EDPVR)
# ==============================================================================
# v11 used one-sided floor penalties for RV_EF and CI, which let the
# optimiser land healthy-level RV_EF (~70 %) and CI (~2.6) in intermediate
# and high-risk cases. v12 replaces those with two-sided penalties toward
# the Humbert T16 band-centre values (see optimize_mesh_circ.py).
#
# LINEAR EDPVR only — the v11 A/B comparison showed exponential gave
# essentially identical achieved hemodynamics.
#
# Outputs:
#   results_mesh_ukb_linear/optimized_regazzoni_ukb_linear_<sev>.json
#     → copied to data/ukb_circ_v12_linear/
#
# v11 LINEAR results in data/ukb_circ_v11_linear/ are preserved for A/B
# comparison with v12.
#
# Usage:
#   sbatch run_circ_optim_v12.sbatch
#   sbatch --export=N_TRIALS=5000 run_circ_optim_v12.sbatch
# ==============================================================================

#SBATCH --job-name=circ_v12
#SBATCH --partition=habanaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=6:00:00
#SBATCH --output=circ_v12_%j.out
#SBATCH --error=circ_v12_%j.err

# Pin each optuna process to 1 thread so 9 parallel processes don't
# thrash the node with BLAS workers. CMA-ES is single-threaded.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# --- Defaults ---
N_TRIALS=${N_TRIALS:-3000}
SEED=${SEED:-42}
LV_EDV=${LV_EDV:-111.46}
RV_EDV=${RV_EDV:-76.83}
MESH_NAME=${MESH_NAME:-"ukb"}
BSA=${BSA:-1.75}

SEVERITIES="sPAP22 sPAP30 sPAP45 sPAP55 sPAP65 sPAP75 sPAP85 sPAP95"

# --- Environment ---
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV

WORK_DIR="/global/D1/homes/dtsteene/cardiac-work"
cd "$WORK_DIR"

echo "============================================================"
echo "  v12 circulation optimization (two-sided RV_EF / CI)"
echo "  Node     : $(hostname) [$SLURM_CPUS_PER_TASK cores allocated]"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Started  : $(date)"
echo "  Trials   : $N_TRIALS per optimization"
echo "  Seed     : $SEED"
echo "  Mesh     : $MESH_NAME (LV_EDV=$LV_EDV, RV_EDV=$RV_EDV mL)"
echo "  BSA      : $BSA m²"
echo "  Severities: $SEVERITIES"
echo "============================================================"

mkdir -p logs_v12

PIDS=()

for SEV in $SEVERITIES; do
    LOG="logs_v12/linear_${SEV}.log"
    echo "Launching LINEAR $SEV → $LOG"
    python "$WORK_DIR/optimize_mesh_circ.py" \
        --lv-edv "$LV_EDV" \
        --rv-edv "$RV_EDV" \
        --mesh-name "$MESH_NAME" \
        --severity "$SEV" \
        --n-trials "$N_TRIALS" \
        --seed "$SEED" \
        --bsa "$BSA" \
        --linear \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All 8 optimizations launched. PIDs: ${PIDS[@]}"
echo "Waiting for completion..."
echo ""

FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=$((FAILED + 1))
done

echo ""
echo "============================================================"
echo "  All optimizations finished"
echo "  Failed   : $FAILED / ${#PIDS[@]}"
echo "  Finished : $(date)"
echo "============================================================"

# --- Copy results into v12 data directory ---
LIN_SRC="$WORK_DIR/results_mesh_${MESH_NAME}_linear"
LIN_DST="$WORK_DIR/data/ukb_circ_v12_linear"

mkdir -p "$LIN_DST"

for SEV in $SEVERITIES; do
    for ext in json png; do
        SRC="$LIN_SRC/optimized_regazzoni_${MESH_NAME}_linear_${SEV}.${ext}"
        [ -f "$SRC" ] && cp "$SRC" "$LIN_DST/" && echo "LINEAR $(basename "$SRC") → $LIN_DST/"

        SRC="$LIN_SRC/pv_evaluation_${MESH_NAME}_linear_${SEV}.${ext}"
        [ -f "$SRC" ] && cp "$SRC" "$LIN_DST/" && echo "LINEAR $(basename "$SRC") → $LIN_DST/"
    done
done

# --- Summary: v12 achieved hemodynamics ---
echo ""
echo "============================================================"
echo "  v12 SUMMARY — achieved hemodynamics"
echo "============================================================"

python3 << 'PYEOF'
import json
import os

CASES = ["sPAP22", "sPAP30", "sPAP45", "sPAP55", "sPAP65",
         "sPAP75", "sPAP85", "sPAP95"]
V12_DIR = "/global/D1/homes/dtsteene/cardiac-work/data/ukb_circ_v12_linear"

def load(sev):
    p = os.path.join(V12_DIR, f"optimized_regazzoni_ukb_linear_{sev}.json")
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None

print(f"{'CASE':<8} | {'sPAP':>6} | {'RAP':>5} | {'SBP':>6} | {'PAWP':>5} | "
      f"{'CI':>5} | {'LVEF':>5} | {'RVEF':>5}")
print("-" * 72)

for sev in CASES:
    d = load(sev)
    if d is None:
        print(f"{sev:<8} | (missing)")
        continue
    m = d.get("metrics_achieved", {})
    h = d.get("derived_hemodynamics", {})
    print(f"{sev:<8} | {m.get('RV_ESP', 0):>6.1f} | {m.get('RV_EDP', 0):>5.1f} | "
          f"{m.get('LV_ESP', 0):>6.1f} | {m.get('LV_EDP', 0):>5.1f} | "
          f"{h.get('CI_Lpm_m2', 0):>5.2f} | "
          f"{h.get('LV_EF_pct', 0):>4.1f}% | {h.get('RV_EF_pct', 0):>4.1f}%")

print()
print(f"Results: {V12_DIR}/")
print()
print("v11 (one-sided RVEF/CI) results remain in:")
print("  /global/D1/homes/dtsteene/cardiac-work/data/ukb_circ_v11_linear/")
print("Legacy case names used: healthy, borderline, mild, moderate,")
print("moderate_severe, severe, very_severe, end_stage.")
PYEOF

exit $FAILED
