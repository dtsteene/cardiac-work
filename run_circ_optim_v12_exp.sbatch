#!/bin/bash
# ==============================================================================
# v12 EXP — circulation optimization with exponential Klotz EDPVR
# ==============================================================================
# Parallel to v12 (linear), same targets, same two-sided CI/RVEF penalties,
# same 8 sPAP cases — only difference is kE is a free parameter instead of
# being pinned at 0.
#
# Rationale: testing whether exponential EDPVR gives tighter 0D↔FEM
# coupling than linear.
#
# Outputs:
#   results_mesh_ukb/optimized_regazzoni_ukb_<sev>.json
#     → copied to data/ukb_circ_v12_exp/
#
# v12 (linear) results remain untouched in data/ukb_circ_v12_linear/.
#
# Usage:
#   sbatch run_circ_optim_v12_exp.sbatch
#   sbatch --export=N_TRIALS=5000 run_circ_optim_v12_exp.sbatch
# ==============================================================================

#SBATCH --job-name=circ_v12e
#SBATCH --partition=habanaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=6:00:00
#SBATCH --output=circ_v12_exp_%j.out
#SBATCH --error=circ_v12_exp_%j.err

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

CASES="sPAP22 sPAP30 sPAP45 sPAP55 sPAP65 sPAP75 sPAP85 sPAP95"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV

WORK_DIR="/global/D1/homes/dtsteene/cardiac-work"
cd "$WORK_DIR"

echo "============================================================"
echo "  v12 EXP circulation optimization (exponential Klotz EDPVR)"
echo "  Node     : $(hostname) [$SLURM_CPUS_PER_TASK cores allocated]"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Started  : $(date)"
echo "  Trials   : $N_TRIALS per optimization"
echo "  Seed     : $SEED"
echo "  Mesh     : $MESH_NAME (LV_EDV=$LV_EDV, RV_EDV=$RV_EDV mL)"
echo "  BSA      : $BSA m²"
echo "  Cases    : $CASES"
echo "============================================================"

mkdir -p logs_v12_exp

PIDS=()

for SEV in $CASES; do
    LOG="logs_v12_exp/exp_${SEV}.log"
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
echo "  Failed   : $FAILED / ${#PIDS[@]}  (NOTE: non-zero exit codes"
echo "             from the post-opt evaluate_and_plot diagnostic"
echo "             do not mean the JSON failed to save — check the"
echo "             data directory below.)"
echo "  Finished : $(date)"
echo "============================================================"

# --- Copy results into v12 exp data directory ---
EXP_SRC="$WORK_DIR/results_mesh_${MESH_NAME}"
EXP_DST="$WORK_DIR/data/ukb_circ_v12_exp"

mkdir -p "$EXP_DST"

for SEV in $CASES; do
    for ext in json png; do
        SRC="$EXP_SRC/optimized_regazzoni_${MESH_NAME}_${SEV}.${ext}"
        [ -f "$SRC" ] && cp "$SRC" "$EXP_DST/" && echo "EXP $(basename "$SRC") → $EXP_DST/"

        SRC="$EXP_SRC/pv_evaluation_${MESH_NAME}_${SEV}.${ext}"
        [ -f "$SRC" ] && cp "$SRC" "$EXP_DST/" && echo "EXP $(basename "$SRC") → $EXP_DST/"
    done
done

# --- Summary: v12 exp achieved hemodynamics ---
echo ""
echo "============================================================"
echo "  v12 EXP SUMMARY — achieved hemodynamics"
echo "============================================================"

python3 << 'PYEOF'
import json, os

CASES = ["sPAP22", "sPAP30", "sPAP45", "sPAP55", "sPAP65",
         "sPAP75", "sPAP85", "sPAP95"]
V12E_DIR = "/global/D1/homes/dtsteene/cardiac-work/data/ukb_circ_v12_exp"

def load(sev):
    p = os.path.join(V12E_DIR, f"optimized_regazzoni_ukb_{sev}.json")
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None

print(f"{'CASE':<8} | {'sPAP':>6} | {'RAP':>5} | {'SBP':>6} | {'PAWP':>5} | "
      f"{'CI':>5} | {'LVEF':>5} | {'RVEF':>5} | {'LV_kE':>6} | {'RV_kE':>6}")
print("-" * 92)

for sev in CASES:
    d = load(sev)
    if d is None:
        print(f"{sev:<8} | (missing)")
        continue
    m = d.get("metrics_achieved", {})
    h = d.get("derived_hemodynamics", {})
    p = d.get("parameters", {})
    lv_kE = (p.get("chambers", {}).get("LV", {}) or {}).get("kE", float("nan"))
    rv_kE = (p.get("chambers", {}).get("RV", {}) or {}).get("kE", float("nan"))
    print(f"{sev:<8} | {m.get('RV_ESP', 0):>6.1f} | {m.get('RV_EDP', 0):>5.1f} | "
          f"{m.get('LV_ESP', 0):>6.1f} | {m.get('LV_EDP', 0):>5.1f} | "
          f"{h.get('CI_Lpm_m2', 0):>5.2f} | "
          f"{h.get('LV_EF_pct', 0):>4.1f}% | {h.get('RV_EF_pct', 0):>4.1f}% | "
          f"{lv_kE:>6.4f} | {rv_kE:>6.4f}")

print()
print(f"Results: {V12E_DIR}/")
print()
print("v12 LINEAR results remain in:")
print("  /global/D1/homes/dtsteene/cardiac-work/data/ukb_circ_v12_linear/")
PYEOF

exit $FAILED
