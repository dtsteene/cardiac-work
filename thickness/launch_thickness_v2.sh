#!/bin/bash
# Phase 4 v2: Thickness variants with fixed epi_outward warp
#
# Rebuilds the rvfw thickness sweep on the clinically-meaningful range
# (3 / 6 / 9 / 12 mm RV FW thickening) after discovering the original
# endo-clamp warp saturated at ~2.7 mm. The new meshes live in
# thickness/warp_meshes/thickness_v2/ and were generated with
# --wall-region rv_fw under the epi_outward mode in
# modify_wall_thickness_regional.
#
# Each variant runs at BOTH healthy and severe circulation so we can
# look at the proxy ranking at two different pressure regimes.
#
# Beats: 1 (fast & dirty — we only look at the last beat, steady state
# will be a second pass once we've ironed out the pipeline on this set).

cd /home/dtsteene/D1/cardiac-work

BEATS=1
BPM=75

VARIANTS=(rvfw_03mm rvfw_06mm rvfw_09mm rvfw_12mm)
CIRCS=(
    "healthy data/ukb_circ_v2/optimized_regazzoni_ukb_healthy.json"
    "severe  data/ukb_circ_v2/optimized_regazzoni_ukb_severe.json"
)

for VARIANT in "${VARIANTS[@]}"; do
    GEOMETRY_DIR="${PWD}/thickness/warp_meshes/thickness_v2/${VARIANT}/ukb/geometry"
    GEOM_FIELDS="${GEOMETRY_DIR}/geometry_fields.npz"

    if [ ! -d "${GEOMETRY_DIR}/geometry.bp" ]; then
        echo "SKIP ${VARIANT} — geometry.bp missing in: $GEOMETRY_DIR"
        continue
    fi
    if [ ! -f "$GEOM_FIELDS" ]; then
        echo "SKIP ${VARIANT} — geometry_fields.npz missing"
        continue
    fi

    for pair in "${CIRCS[@]}"; do
        read SEV CIRC_FILE <<< "$pair"
        if [ ! -f "$CIRC_FILE" ]; then
            echo "SKIP ${VARIANT}/${SEV} — circ file missing: $CIRC_FILE"
            continue
        fi
        COMMENT="Phase4v2 thickness ${VARIANT} ${SEV} ${BEATS}beats"
        echo "Submitting: ${VARIANT} / ${SEV}"
        JOB_ID=$(sbatch \
            --job-name="thickv2_${VARIANT}_${SEV}" \
            --partition=habanaq \
            --time=3:00:00 \
            --export=ALL,MESH=UKB,BPM=$BPM,BEATS=$BEATS,CIRCULATION_PARAMS=$CIRC_FILE,GEOMETRY_DIR=$GEOMETRY_DIR,GEOMETRY_FIELDS=$GEOM_FIELDS,COMMENT="$COMMENT" \
            run_sim_and_post.sbatch | awk '{print $4}')
        echo "  -> Job $JOB_ID"
    done
done

echo ""
echo "=== Submitted ==="
squeue -u $USER -o "%.8i %.26j %.2t %.10M" | head -20
