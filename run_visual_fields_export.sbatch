#!/bin/bash
# Export derived ParaView fields on compute nodes.
#
# Required/optional env vars:
#   CASES="label:/path/to/results label2:/path/to/results"
#   OUT_ROOT=/path/to/paraview_exports/production_h5_core/derived_visual_fields
#   STRIDE=20
#   MAX_FRAMES=60
#   INCLUDE_LEGACY_DIFF=1

#SBATCH --job-name=viz_fields
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

SIM_DIR="/home/dtsteene/D1/cardiac-work"

CASES=${CASES:-"sPAP22:/home/dtsteene/D1/cardiac-work/results/sims/2026-05-10/capped_shared_l5_20260510_141015/sPAP22 sPAP60:/home/dtsteene/D1/cardiac-work/results/sims/2026-05-10/capped_shared_l5_20260510_141015/sPAP60 sPAP95:/home/dtsteene/D1/cardiac-work/results/sims/2026-05-10/capped_shared_l5_20260510_141015/sPAP95"}
OUT_ROOT=${OUT_ROOT:-"/home/dtsteene/D1/cardiac-work/paraview_exports/production_h5_core/derived_visual_fields"}
STRIDE=${STRIDE:-20}
MAX_FRAMES=${MAX_FRAMES:-60}
INCLUDE_LEGACY_DIFF=${INCLUDE_LEGACY_DIFF:-0}

echo "--- Visual field export started: $SLURM_JOB_ID ---"
date
echo "Cases: $CASES"
echo "Output root: $OUT_ROOT"
echo "Stride: $STRIDE"
echo "Max frames: $MAX_FRAMES"
echo "Include legacy diff: $INCLUDE_LEGACY_DIFF"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

CACHE_BASE="/global/D1/homes/dtsteene/tmp/fenics-cache"
export XDG_CACHE_HOME="${CACHE_BASE}/${SLURM_JOB_ID}"
export TMPDIR="${XDG_CACHE_HOME}/tmp"
mkdir -p "$TMPDIR" "$OUT_ROOT"
trap "rm -rf ${XDG_CACHE_HOME}" EXIT

cd "$SIM_DIR"

for item in $CASES; do
    label="${item%%:*}"
    result_dir="${item#*:}"
    out_dir="${OUT_ROOT}/${label}"
    mkdir -p "$out_dir"
    echo
    echo "=== $label ==="
    echo "Result: $result_dir"
    echo "Output: $out_dir"
    date
    extra_args=()
    if [ "$INCLUDE_LEGACY_DIFF" = "1" ]; then
        extra_args+=(--include-legacy-diff)
    fi
    time $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
        $CONDA_PREFIX/bin/python "$SIM_DIR/export_visual_fields.py" \
        "$result_dir" \
        --output-dir "$out_dir" \
        --stride "$STRIDE" \
        --max-frames "$MAX_FRAMES" \
        "${extra_args[@]}"
    exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        echo "ERROR: $label failed with exit code $exit_code"
        exit "$exit_code"
    fi
done

echo
echo "--- Visual field export finished ---"
date
