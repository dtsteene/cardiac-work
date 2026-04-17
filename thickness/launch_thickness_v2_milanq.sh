#!/bin/bash
# Companion to launch_thickness_v2.sh: submits paired milanq jobs that
# race against the existing habanaq ones. Each milanq job knows its
# habanaq pair's job id and, at start time, either kills itself (if
# habanaq beat it) or kills the habanaq pair (if habanaq was still
# waiting).
#
# Usage: paste the 8 habanaq job IDs into the MAP below in the same order
# the launch_thickness_v2.sh loop produces them — rvfw_03mm/healthy first,
# then rvfw_03mm/severe, etc.

cd /home/dtsteene/D1/cardiac-work

# (variant, severity, habanaq_job_id)
declare -a MAP=(
    "rvfw_03mm healthy 1028251"
    "rvfw_03mm severe  1028252"
    "rvfw_06mm healthy 1028253"
    "rvfw_06mm severe  1028254"
    "rvfw_09mm healthy 1028255"
    "rvfw_09mm severe  1028256"
    "rvfw_12mm healthy 1028257"
    "rvfw_12mm severe  1028258"
)

BEATS=1
BPM=75
THICK_ROOT="${PWD}/thickness/warp_meshes/thickness_v2"

for entry in "${MAP[@]}"; do
    read VARIANT SEV PAIR_ID <<< "$entry"
    CIRC_FILE="data/ukb_circ_v2/optimized_regazzoni_ukb_${SEV}.json"
    GEOMETRY_DIR="${THICK_ROOT}/${VARIANT}/ukb/geometry"
    GEOM_FIELDS="${GEOMETRY_DIR}/geometry_fields.npz"

    if [ ! -f "$CIRC_FILE" ]; then
        echo "SKIP ${VARIANT}/${SEV} — circulation file missing: $CIRC_FILE"
        continue
    fi
    if [ ! -d "${GEOMETRY_DIR}/geometry.bp" ]; then
        echo "SKIP ${VARIANT}/${SEV} — geometry.bp missing"
        continue
    fi
    if [ ! -f "$GEOM_FIELDS" ]; then
        echo "SKIP ${VARIANT}/${SEV} — geometry_fields.npz missing"
        continue
    fi

    COMMENT="Phase4v2 thickness ${VARIANT} ${SEV} ${BEATS}beats (milanq race vs ${PAIR_ID})"
    echo "Submitting milanq race for ${VARIANT}/${SEV} vs habanaq job ${PAIR_ID}"
    JOB_ID=$(sbatch \
        --job-name="thickv2_${VARIANT}_${SEV}_m" \
        --export=ALL,MESH=UKB,BPM=$BPM,BEATS=$BEATS,CIRCULATION_PARAMS=$CIRC_FILE,GEOMETRY_DIR=$GEOMETRY_DIR,GEOMETRY_FIELDS=$GEOM_FIELDS,COMMENT="$COMMENT",HABANAQ_PAIR_ID=$PAIR_ID \
        thickness/run_milanq_with_race.sbatch | awk '{print $4}')
    echo "  -> milanq job $JOB_ID  (pair=$PAIR_ID)"
done

echo ""
echo "=== Submitted. Queues now: ==="
squeue -u $USER -o "%.8i %.28j %.2t %.10M %.9P" | head -25
