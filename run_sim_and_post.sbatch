#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=cardiac_sim
#SBATCH --partition=mi50q
#SBATCH --exclude=n052
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# --- 1. Variable Defaults ---
MESH=${MESH:-"UKB"}
BPM=${BPM:-75}
BEATS=${BEATS:-1}
CI=${CI:-0}
CHAR_LENGTH=${CHAR_LENGTH:-10.0}
METRICS_SPACE=${METRICS_SPACE:-"DG1"}
CIRCULATION_PARAMS=${CIRCULATION_PARAMS:-""}
ALPHA_EPI=${ALPHA_EPI:-""}
ALPHA_BASE=${ALPHA_BASE:-""}
INCOMPRESSIBLE=${INCOMPRESSIBLE:-0}
ONE_SIDED_ROBIN=${ONE_SIDED_ROBIN:-0}
COMMENT=${COMMENT:-"No description provided"}
GEOMETRY_DIR=${GEOMETRY_DIR:-""}
RESTART_FROM=${RESTART_FROM:-""}

# --- 2. Paths & Environment ---
SIM_DIR="/home/dtsteene/D1/cardiac-work"
DATA_DIR="${SIM_DIR}/data"

# --- 3. Geometry Logic & Parameter Checks ---

if [ "$MESH" == "UKB" ]; then
    ACTUAL_MESH_PATH=""
    GEO_TAG="UKB"
    echo "Geometry: Standard UKB (Generation)"

elif [ "$MESH" == "PAH" ]; then
    ACTUAL_MESH_PATH="${DATA_DIR}/pah.xdmf"
    GEO_TAG="PAH"
    PAH_PARAMS_FILE="${DATA_DIR}/ph_circulation_params.json"

    echo "Geometry: PAH Preset (${ACTUAL_MESH_PATH})"

    if [ "$CIRCULATION_PARAMS" == "DEFAULT" ]; then
        echo "  ! Override: User forced DEFAULT circulation params (Ignoring PAH JSON)."
        CIRCULATION_PARAMS=""
    elif [ -n "$CIRCULATION_PARAMS" ]; then
        echo "  ! Override: User provided custom circulation params: $CIRCULATION_PARAMS"
    else
        if [ -f "$PAH_PARAMS_FILE" ]; then
            CIRCULATION_PARAMS="$PAH_PARAMS_FILE"
            echo "  Auto-loaded PAH Circulation Params: $PAH_PARAMS_FILE"
        else
            echo "CRITICAL ERROR: Missing $PAH_PARAMS_FILE"
            exit 1
        fi
    fi

elif [ "$MESH" == "HEALTHY" ]; then
    ACTUAL_MESH_PATH="${DATA_DIR}/healthy.xdmf"
    GEO_TAG="HEALTHY"
    HEALTHY_PARAMS_FILE="${DATA_DIR}/healthy_circulation_params.json"

    echo "Geometry: Healthy Preset (${ACTUAL_MESH_PATH})"

    if [ "$CIRCULATION_PARAMS" == "DEFAULT" ]; then
        echo "  ! Override: User forced DEFAULT circulation params (Ignoring Healthy JSON)."
        CIRCULATION_PARAMS=""
    elif [ -n "$CIRCULATION_PARAMS" ]; then
        echo "  ! Override: User provided custom circulation params: $CIRCULATION_PARAMS"
    else
        if [ -f "$HEALTHY_PARAMS_FILE" ]; then
            CIRCULATION_PARAMS="$HEALTHY_PARAMS_FILE"
            echo "  Auto-loaded Healthy Circulation Params: $HEALTHY_PARAMS_FILE"
        else
            echo "CRITICAL ERROR: Missing $HEALTHY_PARAMS_FILE"
            exit 1
        fi
    fi

else
    ACTUAL_MESH_PATH="$MESH"
    if [[ "$ACTUAL_MESH_PATH" != /* ]]; then
        ACTUAL_MESH_PATH="${SLURM_SUBMIT_DIR}/${ACTUAL_MESH_PATH}"
    fi
    GEO_TAG=$(basename "$ACTUAL_MESH_PATH" .xdmf)
    echo "Geometry: Custom File (${ACTUAL_MESH_PATH})"
fi

# Resolve Circulation Params Path
if [ -n "$CIRCULATION_PARAMS" ] && [[ "$CIRCULATION_PARAMS" != /* ]]; then
    CIRCULATION_PARAMS="${SLURM_SUBMIT_DIR}/${CIRCULATION_PARAMS}"
fi

# --- 4. Prepare Arguments for Python ---
ARGS_STR=""

if [ -n "$ACTUAL_MESH_PATH" ]; then
    ARGS_STR="$ARGS_STR --mesh $ACTUAL_MESH_PATH"
else
    ARGS_STR="$ARGS_STR --char_length $CHAR_LENGTH"
fi

ARGS_STR="$ARGS_STR --beats $BEATS"
ARGS_STR="$ARGS_STR --metrics_space $METRICS_SPACE"

[ -n "$CIRCULATION_PARAMS" ] && ARGS_STR="$ARGS_STR --circulation_params $CIRCULATION_PARAMS"
[ "$CI" = "1" ] && ARGS_STR="$ARGS_STR --ci"
[ -n "$ALPHA_EPI" ] && ARGS_STR="$ARGS_STR --alpha_epi $ALPHA_EPI"
[ -n "$ALPHA_BASE" ] && ARGS_STR="$ARGS_STR --alpha_base $ALPHA_BASE"
[ "$INCOMPRESSIBLE" = "1" ] && ARGS_STR="$ARGS_STR --incompressible"
[ "$ONE_SIDED_ROBIN" = "1" ] && ARGS_STR="$ARGS_STR --one-sided-robin"
[ -n "$GEOMETRY_DIR" ] && ARGS_STR="$ARGS_STR --geometry-dir $GEOMETRY_DIR"
[ -n "$RESTART_FROM" ] && ARGS_STR="$ARGS_STR --restart-from $RESTART_FROM"

# --- 5. Setup Directories & Logging ---
echo "--- Job Started: $SLURM_JOB_ID ---"
date

DATE_STR=$(date +%F)
RESULTS_BASE="${SIM_DIR}/results/sims/${DATE_STR}"
RESULTS_DIR="${RESULTS_BASE}/${GEO_TAG}_${BEATS}beats_run_${SLURM_JOB_ID}"
LOG_DIR="${SIM_DIR}/results/log"

mkdir -p ${RESULTS_BASE} ${LOG_DIR} ${RESULTS_DIR}

echo "$COMMENT" > "${RESULTS_DIR}/run_description.txt"

# Log file paths (for copying into results dir at end)
LOG_FILE="${LOG_DIR}/${GEO_TAG}_sim_${SLURM_JOB_ID}.out"
ERR_FILE="${LOG_DIR}/${GEO_TAG}_sim_${SLURM_JOB_ID}.err"
# NOTE: Do NOT use exec redirects here — they cause MPI ranks to block on
# shared file descriptor writes. Slurm --output/--error handles this natively.

# Work directory on local scratch
WORK_DIR="/work/$USER/cardiac_${GEO_TAG}_${SLURM_JOB_ID}"
mkdir -p ${WORK_DIR}
export XDG_CACHE_HOME="${WORK_DIR}/.cache"

# --- 6. Environment ---
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

# --- 7. Execution ---
cd ${WORK_DIR}
echo "Running Simulation..."
echo "  Geometry:  $GEO_TAG"
echo "  Beats:     $BEATS"
echo "  Params:    ${CIRCULATION_PARAMS:-'Default (Internal)'}"
echo "  Restart:   ${RESTART_FROM:-'None (fresh run)'}"
echo "  Save Dir:  $RESULTS_DIR"

time $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork $CONDA_PREFIX/bin/python ${SIM_DIR}/complete_cycle.py $BPM $ARGS_STR
SIM_EXIT=$?

if [ $SIM_EXIT -ne 0 ]; then
    echo "SIMULATION FAILED (exit code $SIM_EXIT) — skipping post-processing"
    rm -rf ${WORK_DIR}
    exit $SIM_EXIT
fi

echo "Simulation completed"

# --- 8. Data Transfer ---
echo "Transferring data..."
RESULTS_WORK=$(find . -maxdepth 1 -type d -name "results_*" | grep -v "geometry" | head -1)

if [ -d "$RESULTS_WORK" ]; then
    cp -r "$RESULTS_WORK"/* ${RESULTS_DIR}/ 2>/dev/null || true
    echo "Data saved to ${RESULTS_DIR}/"
else
    echo "CRITICAL WARNING: No results found in work folder."
fi

# --- 9. Post-Processing ---
echo "Running Offline Metrics Computation..."
cd ${RESULTS_DIR}

if [ -f "simulation_params.json" ] && [ -d "solver/checkpoint.bp" ]; then
    $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork $CONDA_PREFIX/bin/python ${SIM_DIR}/postprocess_metrics.py .
    echo "Metrics computed"

    python3 ${SIM_DIR}/run_postprocessing.py .
    echo "Analysis complete"
else
    echo "Skipping post-processing (missing checkpoint data)."
fi

# Copy slurm logs inside the results folder for easy access
SLURM_OUT="${SIM_DIR}/slurm-${SLURM_JOB_ID}.out"
SLURM_ERR="${SIM_DIR}/slurm-${SLURM_JOB_ID}.err"
cp ${SLURM_OUT} ${RESULTS_DIR}/simulation.log 2>/dev/null
cp ${SLURM_ERR} ${RESULTS_DIR}/simulation.err 2>/dev/null
# Also copy to log dir
cp ${SLURM_OUT} ${LOG_FILE} 2>/dev/null
cp ${SLURM_ERR} ${ERR_FILE} 2>/dev/null

rm -rf ${WORK_DIR}
echo "--- Job Finished ---"
date
