#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=cardiac_buggy_active
#SBATCH --partition=mi50q
#SBATCH --exclude=n052
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
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
BASE_DIRICHLET=${BASE_DIRICHLET:-"x"}
INCOMPRESSIBLE=${INCOMPRESSIBLE:-0}
ONE_SIDED_ROBIN=${ONE_SIDED_ROBIN:-0}
COMMENT=${COMMENT:-"No description provided"}
GEOMETRY_DIR=${GEOMETRY_DIR:-""}
RESTART_FROM=${RESTART_FROM:-""}
LOAD_UNLOADED_FROM=${LOAD_UNLOADED_FROM:-""}
RESTART_PRE_CIRC=${RESTART_PRE_CIRC:-0}
RESTART_RAMP_STEPS=${RESTART_RAMP_STEPS:-""}
PRE_CIRC_BEATS=${PRE_CIRC_BEATS:-""}
PRE_CIRC_MAX_BEATS=${PRE_CIRC_MAX_BEATS:-""}
PRE_CIRC_CONVERGENCE_TOL=${PRE_CIRC_CONVERGENCE_TOL:-""}
RUN_POSTPROCESS=${RUN_POSTPROCESS:-1}
RESULTS_DIR_OVERRIDE=${RESULTS_DIR_OVERRIDE:-""}
STOP_AFTER_UNLOADING=${STOP_AFTER_UNLOADING:-0}
LV_MATERIAL_SCALE=${LV_MATERIAL_SCALE:-""}
RV_MATERIAL_SCALE=${RV_MATERIAL_SCALE:-""}
SEPTUM_MATERIAL_SCALE=${SEPTUM_MATERIAL_SCALE:-""}
RV_EDP_SCALE=${RV_EDP_SCALE:-""}
RV_EDP_MAX_MMHG=${RV_EDP_MAX_MMHG:-""}
RV_EDP_OVERRIDE_MMHG=${RV_EDP_OVERRIDE_MMHG:-""}
# Canonical atlas geometry_fields.npz for compute_per_cell.py. When set,
# per-cell tags are propagated from the shared atlas, giving identical cell
# labels across every case in a spectrum. Leave empty to use per-case
# tag-at-ED mode (anatomically meaningful but case-specific).
GEOMETRY_FIELDS=${GEOMETRY_FIELDS:-""}

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
if [ -n "$LOAD_UNLOADED_FROM" ] && [[ "$LOAD_UNLOADED_FROM" != /* ]]; then
    LOAD_UNLOADED_FROM="${SLURM_SUBMIT_DIR}/${LOAD_UNLOADED_FROM}"
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
[ -n "$BASE_DIRICHLET" ] && ARGS_STR="$ARGS_STR --base-dirichlet $BASE_DIRICHLET"
[ "$INCOMPRESSIBLE" = "1" ] && ARGS_STR="$ARGS_STR --incompressible"
[ "$ONE_SIDED_ROBIN" = "1" ] && ARGS_STR="$ARGS_STR --one-sided-robin"
[ -n "$GEOMETRY_DIR" ] && ARGS_STR="$ARGS_STR --geometry-dir $GEOMETRY_DIR"
[ -n "$RESTART_FROM" ] && ARGS_STR="$ARGS_STR --restart-from $RESTART_FROM"
[ -n "$LOAD_UNLOADED_FROM" ] && ARGS_STR="$ARGS_STR --load-unloaded-from $LOAD_UNLOADED_FROM"
[ "$RESTART_PRE_CIRC" = "1" ] && ARGS_STR="$ARGS_STR --restart-pre-circ"
[ -n "$RESTART_RAMP_STEPS" ] && ARGS_STR="$ARGS_STR --restart-ramp-steps $RESTART_RAMP_STEPS"
[ -n "$PRE_CIRC_BEATS" ] && ARGS_STR="$ARGS_STR --pre-circ-beats $PRE_CIRC_BEATS"
[ -n "$PRE_CIRC_MAX_BEATS" ] && ARGS_STR="$ARGS_STR --pre-circ-max-beats $PRE_CIRC_MAX_BEATS"
[ -n "$PRE_CIRC_CONVERGENCE_TOL" ] && ARGS_STR="$ARGS_STR --pre-circ-convergence-tol $PRE_CIRC_CONVERGENCE_TOL"
[ "$STOP_AFTER_UNLOADING" = "1" ] && ARGS_STR="$ARGS_STR --stop-after-unloading"
[ -n "$LV_MATERIAL_SCALE" ] && ARGS_STR="$ARGS_STR --lv-material-scale $LV_MATERIAL_SCALE"
[ -n "$RV_MATERIAL_SCALE" ] && ARGS_STR="$ARGS_STR --rv-material-scale $RV_MATERIAL_SCALE"
[ -n "$SEPTUM_MATERIAL_SCALE" ] && ARGS_STR="$ARGS_STR --septum-material-scale $SEPTUM_MATERIAL_SCALE"
[ -n "$RV_EDP_SCALE" ] && ARGS_STR="$ARGS_STR --rv-edp-scale $RV_EDP_SCALE"
[ -n "$RV_EDP_MAX_MMHG" ] && ARGS_STR="$ARGS_STR --rv-edp-max-mmhg $RV_EDP_MAX_MMHG"
[ -n "$RV_EDP_OVERRIDE_MMHG" ] && ARGS_STR="$ARGS_STR --rv-edp-override-mmhg $RV_EDP_OVERRIDE_MMHG"

# --- 5. Setup Directories & Logging ---
echo "--- Job Started: $SLURM_JOB_ID ---"
date

DATE_STR=$(date +%F)
RESULTS_BASE="${SIM_DIR}/results/sims/${DATE_STR}"
RESULTS_DIR="${RESULTS_BASE}/${GEO_TAG}_${BEATS}beats_run_${SLURM_JOB_ID}"
if [ -n "$RESULTS_DIR_OVERRIDE" ]; then
    RESULTS_DIR="$RESULTS_DIR_OVERRIDE"
    RESULTS_BASE="$(dirname "$RESULTS_DIR")"
fi
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
SIM_STDOUT="${WORK_DIR}/simulation_stdout.log"
SIM_STDERR="${WORK_DIR}/simulation_stderr.log"
mkdir -p ${WORK_DIR}
export XDG_CACHE_HOME="${WORK_DIR}/.cache"

# --- 6. Environment ---
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate RV
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

# --- 6b. Safety check: verify circulation library has kE (nonlinear EDPVR) support ---
# The JSONs in data/ukb_circ_v2/ contain kE parameters. If the installed
# circulation library silently ignores them, we get wrong hemodynamics.
# Fail fast rather than running a bad sim for hours.
CIRC_CHECK=$($CONDA_PREFIX/bin/python -c "
import sys
try:
    import circulation, inspect
    from circulation.regazzoni2020 import Regazzoni2020
    src = inspect.getsource(inspect.getmodule(Regazzoni2020))
    has_kE = '_kE_LV' in src
    has_nonlin = '(EB / kE) * (np.exp(kE' in src
    loc = circulation.__file__
    if has_kE and has_nonlin:
        print(f'OK {loc}')
    else:
        print(f'MISSING_KE {loc} (kE_store={has_kE}, nonlin={has_nonlin})')
        sys.exit(2)
except Exception as e:
    print(f'IMPORT_ERROR {e}')
    sys.exit(3)
")
CIRC_STATUS=$?
if [ $CIRC_STATUS -ne 0 ]; then
    echo "FATAL: Circulation library check failed"
    echo "  $CIRC_CHECK"
    echo "  The installed circulation library does NOT have kE nonlinear EDPVR support."
    echo "  Run: pip install -e /home/dtsteene/D1/circulation"
    echo "  Refusing to run simulation — would produce wrong hemodynamics."
    exit 1
fi
echo "Circulation lib check: $CIRC_CHECK"

# --- 7. Execution ---
cd ${WORK_DIR}
echo "Running Simulation..."
echo "  Geometry:  $GEO_TAG"
echo "  Beats:     $BEATS"
echo "  Params:    ${CIRCULATION_PARAMS:-'Default (Internal)'}"
echo "  Restart:   ${RESTART_FROM:-'None (fresh run)'}"
echo "  Unloaded reference: ${LOAD_UNLOADED_FROM:-'computed per case'}"
echo "  Restart pre-0D: $RESTART_PRE_CIRC ramp_steps=${RESTART_RAMP_STEPS:-default}"
echo "  Stop unload only: $STOP_AFTER_UNLOADING"
echo "  Run postprocess: $RUN_POSTPROCESS"
echo "  Pre-0D:    beats=${PRE_CIRC_BEATS:-default} max=${PRE_CIRC_MAX_BEATS:-default} tol=${PRE_CIRC_CONVERGENCE_TOL:-off}"
echo "  Material scales: LV=${LV_MATERIAL_SCALE:-1} RV=${RV_MATERIAL_SCALE:-1} Sept=${SEPTUM_MATERIAL_SCALE:-1}"
echo "  RV EDP knobs: scale=${RV_EDP_SCALE:-1} max=${RV_EDP_MAX_MMHG:-none} override=${RV_EDP_OVERRIDE_MMHG:-none}"
echo "  Save Dir:  $RESULTS_DIR"

# Trap EXIT so data transfer happens even on SIGTERM/timeout.
# SLURM sends SIGTERM → 30s grace → SIGKILL. This trap runs on SIGTERM,
# giving us time to copy results before SIGKILL arrives.
TRANSFER_DONE=0
transfer_and_cleanup() {
    if [ "$TRANSFER_DONE" = "1" ]; then
        return
    fi
    TRANSFER_DONE=1
    echo "--- Transferring data from scratch to results dir ---"
    if [ -d "${WORK_DIR}" ]; then
        cd ${WORK_DIR}
        RESULTS_WORK=$(find . -maxdepth 1 -type d -name "results_*" | grep -v "geometry" | head -1)
        if [ -d "$RESULTS_WORK" ]; then
            cp -r "$RESULTS_WORK"/* ${RESULTS_DIR}/ 2>/dev/null || true
            if [ -n "${RESTART_FROM:-}" ] && [ ! -e "${RESULTS_DIR}/geometry" ] && [ -e "${RESTART_FROM}/geometry" ]; then
                ln -s "${RESTART_FROM}/geometry" "${RESULTS_DIR}/geometry" 2>/dev/null || true
            fi
            echo "Data saved to ${RESULTS_DIR}/"
        else
            echo "WARNING: No results directory found in scratch."
        fi
        cp "${SIM_STDOUT}" "${RESULTS_DIR}/simulation_stdout.log" 2>/dev/null || true
        cp "${SIM_STDERR}" "${RESULTS_DIR}/simulation_stderr.log" 2>/dev/null || true
    else
        echo "WARNING: Work directory ${WORK_DIR} not found (already cleaned?)."
    fi
    # Copy slurm logs
    cp "${SIM_DIR}/slurm-${SLURM_JOB_ID}.out" ${RESULTS_DIR}/simulation.log 2>/dev/null
    cp "${SIM_DIR}/slurm-${SLURM_JOB_ID}.err" ${RESULTS_DIR}/simulation.err 2>/dev/null
    cp "${SIM_DIR}/slurm-${SLURM_JOB_ID}.out" ${LOG_FILE} 2>/dev/null
    cp "${SIM_DIR}/slurm-${SLURM_JOB_ID}.err" ${ERR_FILE} 2>/dev/null
    # Cleanup scratch AFTER transfer
    rm -rf ${WORK_DIR} 2>/dev/null
    echo "Scratch cleaned."
}
trap transfer_and_cleanup EXIT

echo "  Simulation stdout: ${SIM_STDOUT}"
echo "  Simulation stderr: ${SIM_STDERR}"
{ time $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork $CONDA_PREFIX/bin/python ${SIM_DIR}/complete_cycle_buggy_active.py $BPM $ARGS_STR; } \
    > "${SIM_STDOUT}" 2> "${SIM_STDERR}"
SIM_EXIT=$?

if [ $SIM_EXIT -ne 0 ]; then
    echo "SIMULATION FAILED (exit code $SIM_EXIT) — data will still be transferred"
fi

echo "Simulation completed (exit code $SIM_EXIT)"

# --- 8. Transfer data FIRST, then post-process ---
# Must transfer before post-processing check, otherwise RESULTS_DIR is empty.
# The EXIT trap is the safety net for timeouts/crashes; this is the normal path.
transfer_and_cleanup

if [ $SIM_EXIT -eq 0 ] && [ "$RUN_POSTPROCESS" = "1" ]; then
    echo "Running Offline Metrics Computation..."
    cd ${RESULTS_DIR}

    if [ -f "simulation_params.json" ] && [ -d "solver/checkpoint.bp" ]; then
        # --- A. CANONICAL regional work: compute_per_cell.py ---
        # Replays the last beat and writes per_cell_data.npz with per-cell
        # work density, transventricular coordinate, and region masks. This
        # is the canonical source for all regional internal work downstream.
        PER_CELL_ARGS="."
        if [ -n "$GEOMETRY_FIELDS" ] && [ -f "$GEOMETRY_FIELDS" ]; then
            echo "  Using canonical atlas geometry fields: $GEOMETRY_FIELDS"
            PER_CELL_ARGS="$PER_CELL_ARGS --geometry-fields $GEOMETRY_FIELDS"
        else
            echo "  Using per-case tag-at-ED mode (no canonical atlas NPZ provided)"
        fi
        echo "Running compute_per_cell.py ..."
        $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
            $CONDA_PREFIX/bin/python ${SIM_DIR}/compute_per_cell_buggy_active.py $PER_CELL_ARGS \
            > "${RESULTS_DIR}/compute_per_cell.log" 2> "${RESULTS_DIR}/compute_per_cell.err"
        PER_CELL_EXIT=$?
        if [ $PER_CELL_EXIT -ne 0 ]; then
            echo "WARNING: compute_per_cell.py exited with code $PER_CELL_EXIT"
        else
            echo "Per-cell work computed"
        fi

        # --- B. Boundary + loops: slim postprocess_metrics.py ---
        # With the 2026-04-13 refactor, default flags skip the regional
        # internal work and directional decomposition paths (compute_per_cell
        # owns them). Set POST_FULL=1 to re-enable the regional internal work
        # computation (needed for a working engineering_debug.png); default
        # stays off to match the fast path.
        POST_ARGS="."
        [ "${POST_FULL:-0}" = "1" ] && POST_ARGS=". --no-skip-regional-internal"
        echo "Running postprocess_metrics.py (POST_FULL=${POST_FULL:-0})..."
        $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
            $CONDA_PREFIX/bin/python ${SIM_DIR}/postprocess_metrics_buggy_active.py $POST_ARGS \
            > "${RESULTS_DIR}/postprocess_metrics.log" 2> "${RESULTS_DIR}/postprocess_metrics.err"
        echo "Metrics computed"

        python3 ${SIM_DIR}/run_postprocessing.py . \
            > "${RESULTS_DIR}/run_postprocessing.log" 2> "${RESULTS_DIR}/run_postprocessing.err"
        echo "Analysis complete"
    else
        echo "Skipping post-processing (missing checkpoint data)."
    fi
elif [ $SIM_EXIT -eq 0 ]; then
    echo "Skipping post-processing (RUN_POSTPROCESS=$RUN_POSTPROCESS)."
fi

# Cleanup and data transfer handled by the EXIT trap (transfer_and_cleanup)
echo "--- Job Finished ---"
date
exit $SIM_EXIT
