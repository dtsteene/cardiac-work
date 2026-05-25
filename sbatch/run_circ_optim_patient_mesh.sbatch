#!/bin/bash
# ==============================================================================
# Patient-mesh circulation optimization for the small mesh x pressure check
# ==============================================================================
# Usage:
#   sbatch --export=ALL,MESH_KEY=healthy run_circ_optim_patient_mesh.sbatch
#   sbatch --export=ALL,MESH_KEY=pah     run_circ_optim_patient_mesh.sbatch
#
# This only optimizes the 0D circulation JSONs. FEM jobs are submitted by
# submit_patient_mesh_fem.sbatch after this job finishes.
# ==============================================================================

#SBATCH --job-name=circ_pat
#SBATCH --partition=habanaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=6:00:00
#SBATCH --output=circ_patient_%x_%j.out
#SBATCH --error=circ_patient_%x_%j.err

set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

MESH_KEY=${MESH_KEY:-healthy}
N_TRIALS=${N_TRIALS:-3000}
SEED=${SEED:-42}
BSA=${BSA:-1.75}
CASES=${CASES:-"sPAP22 sPAP30 sPAP45 sPAP55 sPAP65 sPAP75 sPAP85 sPAP95"}

case "$MESH_KEY" in
    healthy)
        MESH_NAME="patient_healthy"
        LV_EDV=94.964
        RV_EDV=94.404
        ;;
    pah)
        MESH_NAME="patient_pah"
        LV_EDV=83.688
        RV_EDV=74.223
        ;;
    *)
        echo "Unknown MESH_KEY=$MESH_KEY. Use healthy or pah."
        exit 2
        ;;
esac

set +u
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
set -u

WORK_DIR="/global/D1/homes/dtsteene/cardiac-work"
DEST_DIR="${WORK_DIR}/data/patient_mesh_circ_v12_exp/${MESH_KEY}"
LOG_DIR="${WORK_DIR}/logs_patient_mesh_circ/${MESH_KEY}"

cd "$WORK_DIR"
mkdir -p "$DEST_DIR" "$LOG_DIR"

echo "============================================================"
echo "  Patient mesh circulation optimization"
echo "  Mesh key : $MESH_KEY"
echo "  Mesh name: $MESH_NAME"
echo "  Volumes  : LV_EDV=$LV_EDV mL, RV_EDV=$RV_EDV mL"
echo "  Cases    : $CASES"
echo "  Trials   : $N_TRIALS"
echo "  Started  : $(date)"
echo "============================================================"

PIDS=()
for SEV in $CASES; do
    LOG="${LOG_DIR}/${MESH_NAME}_${SEV}.log"
    echo "Launching $MESH_NAME $SEV -> $LOG"
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

FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=$((FAILED + 1))
done

echo ""
echo "Optimizations finished. Non-zero exits: $FAILED / ${#PIDS[@]}"
echo "Copying JSON/PNG outputs to $DEST_DIR"

SRC_DIR="${WORK_DIR}/results_mesh_${MESH_NAME}"
for SEV in $CASES; do
    for STEM in "optimized_regazzoni_${MESH_NAME}_${SEV}" "pv_evaluation_${MESH_NAME}_${SEV}"; do
        for EXT in json png; do
            SRC="${SRC_DIR}/${STEM}.${EXT}"
            if [ -f "$SRC" ]; then
                cp "$SRC" "$DEST_DIR/"
                echo "  copied $(basename "$SRC")"
            fi
        done
    done
done

python3 << PYEOF
import json
from pathlib import Path

dest = Path("$DEST_DIR")
cases = "$CASES".split()
mesh_name = "$MESH_NAME"
print()
print("Achieved hemodynamics:")
print(f"{'case':<8} {'sPAP':>6} {'RAP':>6} {'SBP':>6} {'LVEDV':>7} {'RVEDV':>7} {'CI':>5}")
for case in cases:
    path = dest / f"optimized_regazzoni_{mesh_name}_{case}.json"
    if not path.exists():
        print(f"{case:<8} missing")
        continue
    data = json.loads(path.read_text())
    m = data.get("metrics_achieved", {})
    h = data.get("derived_hemodynamics", {})
    print(
        f"{case:<8} {m.get('RV_ESP', 0):6.1f} {m.get('RV_EDP', 0):6.1f} "
        f"{m.get('LV_ESP', 0):6.1f} {m.get('LV_EDV', 0):7.1f} "
        f"{m.get('RV_EDV', 0):7.1f} {h.get('CI_Lpm_m2', 0):5.2f}"
    )
PYEOF

echo "Finished: $(date)"
exit $FAILED
