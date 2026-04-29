#!/bin/bash
# ==============================================================================
# Diagnostic: measure trial pace under different conditions.
#
# We don't interfere with the main circ_v11_ab job running on h001.
# This runs on slowq (8-core nodes, idle) and times 60 trials under four modes:
#   1. SQLite on BeeGFS (/global/D1/...), BLAS pinned            (baseline)
#   2. SQLite on BeeGFS, BLAS NOT pinned                         (contention?)
#   3. SQLite on node-local /tmp, BLAS pinned                    (NFS theory)
#   4. In-memory SQLite (RDBStorage=None), BLAS pinned           (no DB at all)
#
# Usage: sbatch bench_circ_v11.sbatch
# ==============================================================================

#SBATCH --job-name=bench_circ
#SBATCH --partition=slowq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:45:00
#SBATCH --output=bench_circ_%j.out
#SBATCH --error=bench_circ_%j.err

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV

WORK_DIR="/global/D1/homes/dtsteene/cardiac-work"
cd "$WORK_DIR"

N_TRIALS=60
LV_EDV=111.46
RV_EDV=76.83

echo "============================================================"
echo "  Bench: $N_TRIALS-trial circulation optimization (4 modes)"
echo "  Node     : $(hostname)"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Started  : $(date)"
echo "============================================================"
echo ""

run_mode() {
    local label="$1"
    local extra_env="$2"
    local out_dir="$3"

    echo "----- $label -----"
    local t0=$(date +%s)

    (
        eval "$extra_env"
        python "$WORK_DIR/optimize_mesh_circ.py" \
            --lv-edv "$LV_EDV" --rv-edv "$RV_EDV" \
            --mesh-name "bench_${SLURM_JOB_ID}" \
            --severity healthy \
            --n-trials "$N_TRIALS" \
            --seed 42 \
            --bsa 1.75 \
            2>&1 | tail -5
    )

    local t1=$(date +%s)
    local elapsed=$((t1 - t0))
    local per_trial
    per_trial=$(python3 -c "print(f'{$elapsed / $N_TRIALS:.3f}')")
    echo "$label → elapsed ${elapsed}s for $N_TRIALS trials = ${per_trial}s/trial"
    echo ""
}

# Mode 1: BeeGFS SQLite + BLAS pinned (same as sweep)
run_mode "1. BeeGFS SQLite, BLAS=1" \
    "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1"

# Clean up generated DB so mode 2 starts fresh
rm -f results_mesh_bench_${SLURM_JOB_ID}/v11_mesh_bench_*.db

# Mode 2: BeeGFS SQLite + BLAS unbounded
run_mode "2. BeeGFS SQLite, BLAS default" \
    "unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS BLIS_NUM_THREADS"

# Clean up generated DB
rm -f results_mesh_bench_${SLURM_JOB_ID}/v11_mesh_bench_*.db

# Mode 3: symlink results dir to /tmp so SQLite lives on node-local disk
# (We use a tiny hack: copy the script to a new location where outdir is /tmp)
TMP_RESULTS="/tmp/bench_${SLURM_JOB_ID}/results_mesh_bench_${SLURM_JOB_ID}"
mkdir -p "$TMP_RESULTS"
rm -rf "results_mesh_bench_${SLURM_JOB_ID}" 2>/dev/null
ln -s "$TMP_RESULTS" "results_mesh_bench_${SLURM_JOB_ID}"

run_mode "3. /tmp SQLite, BLAS=1" \
    "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1"

# Clean up symlinked dir
rm -rf "$TMP_RESULTS" 2>/dev/null
unlink "results_mesh_bench_${SLURM_JOB_ID}" 2>/dev/null

# Mode 4: patch storage to in-memory via env var and rerun
echo "----- 4. In-memory storage, BLAS=1 -----"
t0=$(date +%s)
(
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
    python3 << PYEOF 2>&1 | tail -8
import sys
sys.path.insert(0, "$WORK_DIR")
import time
import optuna
from optuna.samplers import CmaEsSampler
from optimize_mesh_circ import (
    ModelInterface, SEVERITY_PRESSURE_TARGETS, SEVERITY_PHYSIOLOGY,
    PARAMETER_CONFIG, make_objective,
)

interface = ModelInterface(linear=False)
pressure_targets = SEVERITY_PRESSURE_TARGETS["healthy"]
physiology = SEVERITY_PHYSIOLOGY["healthy"]

study = optuna.create_study(
    direction="minimize",
    sampler=CmaEsSampler(seed=42, warn_independent_sampling=False),
    # storage=None → RDBStorage not used; trials kept in memory only
)

t0 = time.perf_counter()
study.optimize(
    make_objective(interface, pressure_targets, $LV_EDV, $RV_EDV,
                   physiology=physiology, bsa=1.75),
    n_trials=$N_TRIALS,
    n_jobs=1,
    show_progress_bar=False,
)
t1 = time.perf_counter()
print(f"In-memory elapsed: {t1 - t0:.2f}s for $N_TRIALS trials")
print(f"  → {(t1 - t0) / $N_TRIALS:.3f}s/trial")
PYEOF
)
t1=$(date +%s)
echo "4. Elapsed wall: $((t1 - t0))s"

echo ""
echo "============================================================"
echo "  Finished : $(date)"
echo "============================================================"
