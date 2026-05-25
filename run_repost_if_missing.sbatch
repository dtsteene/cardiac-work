#!/bin/bash

# Guarded postprocessing recovery.
#
# This job is intended to be submitted with an afterany dependency on a
# simulation+postprocessing job. If the normal postprocessing outputs are
# already present, it exits immediately. If they are missing, it runs the
# existing scratch-based recovery postprocessing job body inline.

#SBATCH --job-name=repost_guard
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --output=repost_guard_%j.out
#SBATCH --error=repost_guard_%j.err

set -euo pipefail

SIM_DIR=${SIM_DIR:-"/home/dtsteene/D1/cardiac-work"}
RESULTS_DIR=${RESULTS_DIR:?"Set RESULTS_DIR to the simulation results directory"}
GEOMETRY_FIELDS=${GEOMETRY_FIELDS:-""}

RESULTS_DIR=$(readlink -f "$RESULTS_DIR")

echo "--- Guarded repostprocess check: ${SLURM_JOB_ID} ---"
date
echo "Results dir: $RESULTS_DIR"

missing=()

check_file() {
    local rel="$1"
    if [ ! -s "${RESULTS_DIR}/${rel}" ]; then
        missing+=("$rel")
    fi
}

check_any_file() {
    local label="$1"
    shift
    local found=0
    local rel
    for rel in "$@"; do
        if [ -s "${RESULTS_DIR}/${rel}" ]; then
            found=1
            break
        fi
    done
    if [ "$found" = "0" ]; then
        missing+=("$label")
    fi
}

check_file "per_cell_data.npz"
check_file "solver/solver_cavity_pressure_mmHg.npy"
check_any_file "metrics_downsample_1.npy" \
    "metrics/metrics_downsample_1.npy" \
    "metrics_downsample_1.npy"
check_file "analysis/last_beat/engineering_debug.png"
check_file "analysis/last_beat/hemodynamics.json"

if [ "${#missing[@]}" -eq 0 ]; then
    echo "Postprocessing outputs already present; no-op."
    date
    exit 0
fi

echo "Missing postprocessing outputs:"
printf '  - %s\n' "${missing[@]}"

if [ ! -f "${RESULTS_DIR}/simulation_params.json" ] || [ ! -d "${RESULTS_DIR}/solver/checkpoint.bp" ]; then
    echo "ERROR: cannot recover postprocessing; missing simulation_params.json or solver/checkpoint.bp" >&2
    exit 2
fi

export SIM_DIR RESULTS_DIR GEOMETRY_FIELDS
echo "Running scratch-based recovery postprocessing..."
bash "${SIM_DIR}/recover_postprocess_run.sbatch"
