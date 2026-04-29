#!/bin/bash
# ==============================================================================
# FEM dispatcher for the small patient-mesh x pressure robustness check
# ==============================================================================
# Submit after the corresponding circulation optimization:
#   sbatch --dependency=afterany:<circ_job> \
#     --export=ALL,MESH_KEY=healthy submit_patient_mesh_fem.sbatch
# ==============================================================================

#SBATCH --job-name=fem_pat
#SBATCH --partition=habanaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --output=fem_patient_%x_%j.out
#SBATCH --error=fem_patient_%x_%j.err

set -u

MESH_KEY=${MESH_KEY:-healthy}
CASES=${CASES:-"sPAP22 sPAP30 sPAP45 sPAP55 sPAP65 sPAP75 sPAP85 sPAP95"}
BEATS=${BEATS:-6}

case "$MESH_KEY" in
    healthy)
        RUN_MESH="HEALTHY"
        MESH_NAME="patient_healthy"
        ;;
    pah)
        RUN_MESH="PAH"
        MESH_NAME="patient_pah"
        ;;
    *)
        echo "Unknown MESH_KEY=$MESH_KEY. Use healthy or pah."
        exit 2
        ;;
esac

WORK_DIR="/global/D1/homes/dtsteene/cardiac-work"
DATA_DIR="${WORK_DIR}/data/patient_mesh_circ_v12_exp/${MESH_KEY}"
MANIFEST_DIR="${WORK_DIR}/results/patient_mesh_sweep"
MANIFEST="${MANIFEST_DIR}/${MESH_KEY}_jobs_${SLURM_JOB_ID}.tsv"

cd "$WORK_DIR"
mkdir -p "$MANIFEST_DIR"

echo "============================================================"
echo "  Patient mesh FEM dispatch"
echo "  Mesh key : $MESH_KEY"
echo "  Run mesh : $RUN_MESH"
echo "  Data dir : $DATA_DIR"
echo "  Cases    : $CASES"
echo "  Beats    : $BEATS"
echo "  Started  : $(date)"
echo "============================================================"

printf "mesh_key\tcase\tjob_id\tjson\n" > "$MANIFEST"

JOB_IDS=()
MISSING=0
for SEV in $CASES; do
    JSON="${DATA_DIR}/optimized_regazzoni_${MESH_NAME}_${SEV}.json"
    if [ ! -f "$JSON" ]; then
        echo "  [MISSING] $JSON — skipping $SEV"
        MISSING=$((MISSING + 1))
        continue
    fi

    COMMENT="patient_mesh_sweep_${MESH_KEY}_${SEV}_${BEATS}bt"
    echo "  Submitting $MESH_KEY $SEV"
    JID=$(sbatch --parsable \
        --export=ALL,MESH="$RUN_MESH",BEATS="$BEATS",POST_FULL=1,\
CIRCULATION_PARAMS="$JSON",COMMENT="$COMMENT" \
        run_sim_and_post.sbatch)
    if [ -n "$JID" ]; then
        JOB_IDS+=("$JID")
        printf "%s\t%s\t%s\t%s\n" "$MESH_KEY" "$SEV" "$JID" "$JSON" >> "$MANIFEST"
        echo "    -> $JID"
    else
        echo "    ! submission failed"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
echo "Submitted ${#JOB_IDS[@]} jobs; missing/failed: $MISSING"
echo "Job IDs: ${JOB_IDS[@]}"
echo "Manifest: $MANIFEST"
echo "Finished: $(date)"
