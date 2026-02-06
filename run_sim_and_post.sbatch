#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=cardiac_sim
#SBATCH --partition=hgx2q  
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --exclude=n052

# --- NEW: Temporary Log Handling ---
# This creates the files, but we will merge and delete them inside the script
#SBATCH --output=init_slurm_%j.out
#SBATCH --error=init_slurm_%j.err

# --- Usage Examples ---
# 1. Default (UKB, 1 Beat):         sbatch run_sim.sbatch
# 2. Multi-beat Run:                sbatch --export=BEATS=10 run_sim.sbatch
# 3. PAH Mesh (Auto-loads JSON):    sbatch --export=MESH=PAH run_sim.sbatch
# 4. PAH Mesh (Force Default):      sbatch --export=MESH=PAH,CIRCULATION_PARAMS="DEFAULT" run_sim.sbatch
# 5. Custom Mesh:                   sbatch --export=MESH="./my_mesh.xdmf" run_sim.sbatch

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
COMMENT=${COMMENT:-"No description provided"}

# --- 2. Paths & Environment ---
SUBMIT_DIR=$SLURM_SUBMIT_DIR
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

    # --- PAH Special Logic ---
    if [ "$CIRCULATION_PARAMS" == "DEFAULT" ]; then
        echo "  ! Override: User forced DEFAULT circulation params (Ignoring PAH JSON)."
        CIRCULATION_PARAMS="" 
        
    elif [ -n "$CIRCULATION_PARAMS" ]; then
        echo "  ! Override: User provided custom circulation params: $CIRCULATION_PARAMS"
        
    else
        if [ -f "$PAH_PARAMS_FILE" ]; then
            CIRCULATION_PARAMS="$PAH_PARAMS_FILE"
            echo "  ✓ Auto-loaded PAH Circulation Params: $PAH_PARAMS_FILE"
        else
            echo ""
            echo "❌ CRITICAL ERROR: Circulation Parameters Missing"
            echo "   Missing: $PAH_PARAMS_FILE"
            echo "   To Fix: Ensure file exists OR run with CIRCULATION_PARAMS='DEFAULT'"
            echo ""
            exit 1
        fi
    fi

elif [ "$MESH" == "HEALTHY" ]; then
    ACTUAL_MESH_PATH="${DATA_DIR}/healthy.xdmf"
    GEO_TAG="HEALTHY"
    echo "Geometry: Healthy Preset (${ACTUAL_MESH_PATH})"

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

# Mesh selection
if [ -n "$ACTUAL_MESH_PATH" ]; then
    ARGS_STR="$ARGS_STR --mesh $ACTUAL_MESH_PATH"
else
    ARGS_STR="$ARGS_STR --char_length $CHAR_LENGTH"
fi

ARGS_STR="$ARGS_STR --beats $BEATS"
ARGS_STR="$ARGS_STR --metrics_space $METRICS_SPACE"

if [ -n "$CIRCULATION_PARAMS" ]; then
    ARGS_STR="$ARGS_STR --circulation_params $CIRCULATION_PARAMS"
fi

if [ "$CI" = "1" ]; then
    ARGS_STR="$ARGS_STR --ci"
fi

if [ -n "$ALPHA_EPI" ]; then
    ARGS_STR="$ARGS_STR --alpha_epi $ALPHA_EPI"
fi

if [ -n "$ALPHA_BASE" ]; then
    ARGS_STR="$ARGS_STR --alpha_base $ALPHA_BASE"
fi

# --- 5. Setup Logging & Directories ---
echo "--- Job Started: $SLURM_JOB_ID ---"
date

DATE_STR=$(date +%F)
RESULTS_BASE="${SIM_DIR}/results/sims/${DATE_STR}"
RESULTS_DIR="${RESULTS_BASE}/${GEO_TAG}_${BEATS}beats_run_${SLURM_JOB_ID}"
LOG_DIR="${SIM_DIR}/results/log"

mkdir -p ${RESULTS_BASE} ${LOG_DIR} ${RESULTS_DIR}

echo "$COMMENT" > "${RESULTS_DIR}/run_description.txt"

# Define the Final Log Files
LOG_FILE="${LOG_DIR}/${GEO_TAG}_sim_${SLURM_JOB_ID}.out"
ERR_FILE="${LOG_DIR}/${GEO_TAG}_sim_${SLURM_JOB_ID}.err"

# --- LOGGING MAGIC STARTS HERE ---
# 1. Flush current buffer so everything printed so far is saved to init_slurm...
sync

# 2. Merge the temporary start-up logs into the final logs
cat "init_slurm_${SLURM_JOB_ID}.out" >> "${LOG_FILE}" 2>/dev/null
cat "init_slurm_${SLURM_JOB_ID}.err" >> "${ERR_FILE}" 2>/dev/null

# 3. Redirect all FUTURE output to the final logs
exec 1>>"${LOG_FILE}"
exec 2>>"${ERR_FILE}"

# 4. Clean up the temporary files from the submit dir
rm -f "init_slurm_${SLURM_JOB_ID}.out" "init_slurm_${SLURM_JOB_ID}.err"
# --- LOGGING MAGIC ENDS HERE ---

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
echo "  Geometry: $GEO_TAG"
echo "  Beats:    $BEATS"
echo "  Params:   ${CIRCULATION_PARAMS:-'Default (Internal)'}"
echo "  Save Dir: $RESULTS_DIR"

time $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork $CONDA_PREFIX/bin/python ${SIM_DIR}/complete_cycle.py $BPM $ARGS_STR

echo "✓ Simulation completed"

# --- 8. Data Transfer ---
echo "Transferring data..."
RESULTS_WORK=$(find . -maxdepth 1 -type d -name "results_*" | grep -v "geometry" | head -1)

if [ -d "$RESULTS_WORK" ]; then
    cp -r "$RESULTS_WORK"/* ${RESULTS_DIR}/ 2>/dev/null || true
    echo "✓ Data saved to ${RESULTS_DIR}/"
else
    echo "CRITICAL WARNING: No results found in work folder."
fi

# --- 9. Post-Processing ---
echo "Running Post-Processing..."
cd ${RESULTS_DIR}

if [ -f "metrics_downsample_1.npy" ]; then
    python3 ${SIM_DIR}/run_postprocessing.py .
    echo "✓ Analysis complete"
else
    echo "Skipping analysis (missing metrics file)."
fi

# Copy logs inside the results folder for easy access
cp ${LOG_FILE} ${RESULTS_DIR}/simulation.log 2>/dev/null
cp ${ERR_FILE} ${RESULTS_DIR}/simulation.err 2>/dev/null

rm -rf ${WORK_DIR}
echo "--- Job Finished ---"
date