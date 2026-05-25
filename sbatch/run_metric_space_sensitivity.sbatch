#!/bin/bash
# Postprocessing-only sensitivity of regional work to MetricsCalculator state space.
#
# This job intentionally runs sequentially on a compute node. It opens ADIOS
# checkpoints and replays the final beat, so do not run this workflow on a login
# node.

#SBATCH --job-name=metric_space
#SBATCH --partition=mi50q
#SBATCH --exclude=n052
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --output=metric-space-%j.out
#SBATCH --error=metric-space-%j.err

set -eo pipefail

SIM_DIR="/home/dtsteene/D1/cardiac-work"
WORK_DIR="/work/$USER/metric_space_${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
export XDG_CACHE_HOME="${WORK_DIR}/.cache"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

cd "$SIM_DIR"

SPACES=${SPACES:-"Quadrature6 DG0 DG1"}
FORCE=${FORCE:-0}

RUNS=(
  "sPAP22:/home/dtsteene/D1/cardiac-work/results/sims/2026-04-23/UKB_6beats_run_1047450"
  "sPAP60:/home/dtsteene/D1/cardiac-work/results/sims/2026-04-24/UKB_6beats_run_1048197"
  "sPAP95:/home/dtsteene/D1/cardiac-work/results/sims/2026-04-23/UKB_6beats_run_1047457"
)

echo "============================================================"
echo "Metric-space sensitivity"
echo "Node      : $(hostname)"
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Started   : $(date)"
echo "Spaces    : ${SPACES}"
echo "MPI tasks : ${SLURM_NTASKS}"
echo "============================================================"

for item in "${RUNS[@]}"; do
    label="${item%%:*}"
    run_dir="${item#*:}"
    if [ ! -d "${run_dir}/solver/checkpoint.bp" ]; then
        echo "ERROR: missing checkpoint for ${label}: ${run_dir}/solver/checkpoint.bp" >&2
        exit 1
    fi

    for space in ${SPACES}; do
        space_tag="$(echo "$space" | tr '[:upper:]' '[:lower:]')"
        metrics_subdir="metrics_space_${space_tag}"
        output_file="${run_dir}/${metrics_subdir}/metrics_downsample_1.npy"

        if [ "$FORCE" != "1" ] && [ -f "$output_file" ]; then
            echo "SKIP ${label} ${space}: ${output_file} already exists"
            continue
        fi

        echo "----- ${label} ${space} -----"
        date
        set +e
        "$CONDA_PREFIX/bin/mpirun" -n "${SLURM_NTASKS}" -launcher fork \
            "$CONDA_PREFIX/bin/python" "$SIM_DIR/postprocess_metrics.py" "$run_dir" \
                --last-beat \
                --no-skip-regional-internal \
                --metrics-space "$space" \
                --metrics-subdir "$metrics_subdir" \
            > "${run_dir}/postprocess_metric_space_${space_tag}.log" \
            2> "${run_dir}/postprocess_metric_space_${space_tag}.err"
        rc=$?
        set -e
        if [ "$rc" -eq 0 ]; then
            echo "DONE ${label} ${space}"
        else
            echo "FAILED ${label} ${space} with exit code ${rc}" >&2
        fi
    done
done

set +e
"$CONDA_PREFIX/bin/python" "$SIM_DIR/analyze_metric_space_sensitivity.py"
ANALYZE_RC=$?
set -e
if [ "$ANALYZE_RC" -ne 0 ]; then
    echo "WARNING: summary analysis failed with exit code ${ANALYZE_RC}" >&2
fi

echo "Finished: $(date)"
