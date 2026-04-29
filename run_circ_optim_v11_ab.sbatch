#!/bin/bash
# ==============================================================================
# v11 A/B circulation optimization — exponential vs linear Klotz EDPVR
# ==============================================================================
# Runs 18 optimizations in parallel on one habanaq node (h001, 144 cores):
#   9 severities × 2 EDPVR variants (exponential kE-free, linear kE=0).
#
# Each optimization is single-threaded CMA-ES via Optuna. h001 has 144 cores so
# all 18 optimizations can share the node comfortably (~8 cores per process in
# reserve for BLAS threads, though we pin each to 1 thread below for fairness).
#
# Outputs (under cardiac-work/):
#   Exponential: results_mesh_ukb/optimized_regazzoni_ukb_<sev>.json
#                → copied to data/ukb_circ_v11/
#   Linear:      results_mesh_ukb_linear/optimized_regazzoni_ukb_linear_<sev>.json
#                → copied to data/ukb_circ_v11_linear/
#
# Usage:
#   sbatch run_circ_optim_v11_ab.sbatch
#   sbatch --export=N_TRIALS=5000 run_circ_optim_v11_ab.sbatch
#
# Status:
#   squeue -u $USER
#   tail -f circ_v11_ab_<jobid>.out
# ==============================================================================

#SBATCH --job-name=circ_v11_ab
#SBATCH --partition=habanaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=6:00:00
#SBATCH --output=circ_v11_ab_%j.out
#SBATCH --error=circ_v11_ab_%j.err

# Pin each python/optuna process to 1 thread so 18 parallel processes don't
# thrash the node with BLAS workers. CMA-ES is single-threaded.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# --- Defaults ---
N_TRIALS=${N_TRIALS:-3000}
SEED=${SEED:-42}
# UKB mesh cavity volumes (pre-computed so we don't need MPI/FEniCSx on compute node)
LV_EDV=${LV_EDV:-111.46}
RV_EDV=${RV_EDV:-76.83}
MESH_NAME=${MESH_NAME:-"ukb"}
BSA=${BSA:-1.75}

SEVERITIES="healthy_low healthy borderline mild moderate moderate_severe severe very_severe end_stage"

# --- Environment ---
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV

WORK_DIR="/global/D1/homes/dtsteene/cardiac-work"
cd "$WORK_DIR"

echo "============================================================"
echo "  v11 A/B circulation optimization"
echo "  Node     : $(hostname) [$SLURM_CPUS_PER_TASK cores allocated]"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Started  : $(date)"
echo "  Trials   : $N_TRIALS per optimization"
echo "  Seed     : $SEED"
echo "  Mesh     : $MESH_NAME (LV_EDV=$LV_EDV, RV_EDV=$RV_EDV mL)"
echo "  BSA      : $BSA m²"
echo "  Severities: $SEVERITIES"
echo "============================================================"

# Launch all 18 optimizations as background processes.
# Per-run log files under logs/ so we can inspect progress individually.
mkdir -p logs_v11_ab

PIDS=()

for SEV in $SEVERITIES; do
    # Exponential EDPVR (default — kE is a free parameter)
    LOG="logs_v11_ab/exp_${SEV}.log"
    echo "Launching EXP    $SEV → $LOG"
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

    # Linear EDPVR (kE pinned at 0)
    LOG="logs_v11_ab/linear_${SEV}.log"
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
echo "All 18 optimizations launched. PIDs: ${PIDS[@]}"
echo "Waiting for completion..."
echo ""

# Wait for all and track failures
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

# --- Copy results into versioned data directories ---
EXP_SRC="$WORK_DIR/results_mesh_${MESH_NAME}"
LIN_SRC="$WORK_DIR/results_mesh_${MESH_NAME}_linear"
EXP_DST="$WORK_DIR/data/ukb_circ_v11"
LIN_DST="$WORK_DIR/data/ukb_circ_v11_linear"

mkdir -p "$EXP_DST" "$LIN_DST"

for SEV in $SEVERITIES; do
    for ext in json png; do
        SRC="$EXP_SRC/optimized_regazzoni_${MESH_NAME}_${SEV}.${ext}"
        [ -f "$SRC" ] && cp "$SRC" "$EXP_DST/" && echo "EXP    $(basename "$SRC") → $EXP_DST/"

        SRC="$EXP_SRC/pv_evaluation_${MESH_NAME}_${SEV}.${ext}"
        [ -f "$SRC" ] && cp "$SRC" "$EXP_DST/" && echo "EXP    $(basename "$SRC") → $EXP_DST/"

        SRC="$LIN_SRC/optimized_regazzoni_${MESH_NAME}_linear_${SEV}.${ext}"
        [ -f "$SRC" ] && cp "$SRC" "$LIN_DST/" && echo "LINEAR $(basename "$SRC") → $LIN_DST/"

        SRC="$LIN_SRC/pv_evaluation_${MESH_NAME}_linear_${SEV}.${ext}"
        [ -f "$SRC" ] && cp "$SRC" "$LIN_DST/" && echo "LINEAR $(basename "$SRC") → $LIN_DST/"
    done
done

# --- Final A/B summary ---
echo ""
echo "============================================================"
echo "  A/B SUMMARY — achieved vs target (last beat)"
echo "============================================================"

python3 << 'PYEOF'
import json
import os

SEVERITIES = ["healthy_low", "healthy", "borderline", "mild", "moderate",
              "moderate_severe", "severe", "very_severe", "end_stage"]
EXP_DIR = "/global/D1/homes/dtsteene/cardiac-work/data/ukb_circ_v11"
LIN_DIR = "/global/D1/homes/dtsteene/cardiac-work/data/ukb_circ_v11_linear"

def load(base, name, sev):
    p = os.path.join(base, f"optimized_regazzoni_{name}_{sev}.json")
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None

print(f"{'SEVERITY':<18} | {'Variant':>8} | {'LV_ESP':>7} | {'RV_ESP':>7} | "
      f"{'LV_EDP':>7} | {'RV_EDP':>7} | {'CI':>5} | {'LV_EF':>6} | {'RV_EF':>6}")
print("-" * 110)

for sev in SEVERITIES:
    for (label, base, name) in [("EXP", EXP_DIR, "ukb"), ("LINEAR", LIN_DIR, "ukb_linear")]:
        d = load(base, name, sev)
        if d is None:
            print(f"{sev:<18} | {label:>8} | (missing)")
            continue
        m = d.get("metrics_achieved", {})
        h = d.get("derived_hemodynamics", {})
        print(f"{sev:<18} | {label:>8} | "
              f"{m.get('LV_ESP', 0):>7.1f} | {m.get('RV_ESP', 0):>7.1f} | "
              f"{m.get('LV_EDP', 0):>7.1f} | {m.get('RV_EDP', 0):>7.1f} | "
              f"{h.get('CI_Lpm_m2', 0):>5.2f} | "
              f"{h.get('LV_EF_pct', 0):>5.1f}% | {h.get('RV_EF_pct', 0):>5.1f}%")

print()
print(f"EXP    results: {EXP_DIR}/")
print(f"LINEAR results: {LIN_DIR}/")
PYEOF

exit $FAILED
