#!/bin/bash
# Phase 1 resubmission: 8 severities on a SINGLE shared UKB mesh
# so that per-cell data is consistent across cases (same cell indexing,
# same region masks, same tau values).
#
# The shared mesh must exist at data/shared_ukb_mesh/ukb/geometry/ — see
# geometry_generator.py --single ukb -c 10 --output-dir data/shared_ukb_mesh
#
# Each sim loads the mesh via adios4dolfinx (verified bit-for-bit correct
# round-trip for serial-write / parallel-read). See test_adios_roundtrip.py.

set -e
cd /home/dtsteene/D1/cardiac-work

SHARED_GEOM="${PWD}/data/shared_ukb_mesh/ukb/geometry"
if [ ! -d "$SHARED_GEOM/geometry.bp" ]; then
    echo "ERROR: shared geometry.bp not found at $SHARED_GEOM"
    echo "Generate it first:"
    echo "  python3 geometry_generator.py --single ukb -c 10 --output-dir data/shared_ukb_mesh"
    exit 1
fi

CIRC_DIR="data/ukb_circ_v2"
BEATS=${BEATS:-6}
BPM=75
PARTITION=${PARTITION:-mi50q}

echo "=== Phase 1 (shared mesh): 8 disease severities ==="
echo "Shared mesh: $SHARED_GEOM"
echo "Partition:   $PARTITION"
echo "Beats:       $BEATS  (use BEATS=6 for publication-quality converged runs)"
echo "BPM:         $BPM"
echo ""

for SEVERITY in healthy_low healthy mild moderate moderate_severe severe very_severe end_stage; do
    CIRC_FILE="${CIRC_DIR}/optimized_regazzoni_ukb_${SEVERITY}.json"
    if [ ! -f "$CIRC_FILE" ]; then
        echo "SKIP $SEVERITY — $CIRC_FILE not found"
        continue
    fi

    RV_ESP=$(python3 -c "import json; d=json.load(open('$CIRC_FILE')); print(d['pressure_targets']['RV_ESP'])" 2>/dev/null)
    echo "Submitting: $SEVERITY (RV_ESP target=${RV_ESP}mmHg)"

    JOB_ID=$(sbatch \
        --job-name="ukb_${SEVERITY}" \
        --partition=$PARTITION \
        --time=6:00:00 \
        --export=ALL,MESH=UKB,BPM=$BPM,BEATS=$BEATS,CIRCULATION_PARAMS=$CIRC_FILE,GEOMETRY_DIR=$SHARED_GEOM,COMMENT="Phase1 shared-mesh v2circ ${SEVERITY} ${BEATS}beats" \
        run_sim_and_post.sbatch | awk '{print $4}')
    echo "  -> Job $JOB_ID"
done

echo ""
echo "Monitor with: squeue -u \$USER"
