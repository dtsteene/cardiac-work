#!/bin/bash
# Phase 4: Thickness variants (collaborator request)
#
# Uses pre-built warp meshes in thickness/warp_meshes/thickness_study/
# Each variant = different RV/global wall thickening on the UKB baseline.
# All variants use the same "severe" PAH circulation (RV_ESP=72 mmHg)
# to isolate the effect of wall thickness on proxy accuracy.
#
# Variants:
#   global_1mm, global_2mm  — uniform thickening everywhere
#   rvfw_2mm, rvfw_3mm, rvfw_5mm, rvfw_7mm  — RV free wall only (PAH hypertrophy)

cd /home/dtsteene/D1/cardiac-work

CIRC_FILE="data/ukb_circ_v2/optimized_regazzoni_ukb_severe.json"
BEATS=6
BPM=75

if [ ! -f "$CIRC_FILE" ]; then
    echo "ERROR: circulation file missing: $CIRC_FILE"
    exit 1
fi

echo "=== Phase 4: Thickness Variants (severe PAH circulation) ==="
echo "Circ: $CIRC_FILE"
echo "Beats: $BEATS, BPM: $BPM"
echo ""

VARIANTS=(
    global_1mm
    global_2mm
    rvfw_2mm
    rvfw_3mm
    rvfw_5mm
    rvfw_7mm
)

for VARIANT in "${VARIANTS[@]}"; do
    GEOMETRY_DIR="${PWD}/thickness/warp_meshes/thickness_cl10/${VARIANT}/ukb/geometry"
    GEOM_FIELDS="${GEOMETRY_DIR}/geometry_fields.npz"

    if [ ! -d "$GEOMETRY_DIR" ]; then
        echo "SKIP ${VARIANT} — geometry not found: $GEOMETRY_DIR"
        continue
    fi
    if [ ! -d "${GEOMETRY_DIR}/geometry.bp" ]; then
        echo "SKIP ${VARIANT} — geometry.bp missing in: $GEOMETRY_DIR"
        continue
    fi
    if [ ! -f "$GEOM_FIELDS" ]; then
        echo "SKIP ${VARIANT} — geometry_fields.npz missing: $GEOM_FIELDS"
        echo "  Run: python3 precompute_geometry_fields.py $GEOMETRY_DIR"
        continue
    fi

    echo "Submitting: ${VARIANT} -> ${GEOMETRY_DIR}"

    JOB_ID=$(sbatch \
        --job-name="thick_${VARIANT}" \
        --partition=habanaq \
        --time=6:00:00 \
        --export=ALL,MESH=UKB,BPM=$BPM,BEATS=$BEATS,CIRCULATION_PARAMS=$CIRC_FILE,GEOMETRY_DIR=$GEOMETRY_DIR,GEOMETRY_FIELDS=$GEOM_FIELDS,COMMENT="Phase4 thickness ${VARIANT} severe ${BEATS}beats" \
        run_sim_and_post.sbatch | awk '{print $4}')

    echo "  -> Job $JOB_ID"
done

echo ""
echo "=== Submitted. Monitor with: squeue -u \$USER ==="
