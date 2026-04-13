#!/bin/bash
# Submit per-cell computation jobs for Phase 4 thickness variants
# Run AFTER the thickness sims complete (jobs 1017843-1017848)

cd /home/dtsteene/D1/cardiac-work

RESULTS_DATE="2026-04-09"  # adjust if thickness sims finish on a different date

# Find thickness sim directories automatically
echo "=== Submitting per-cell jobs (Phase 4, thickness variants) ==="

for VARIANT in global_1mm global_2mm rvfw_2mm rvfw_3mm rvfw_5mm rvfw_7mm; do
    # Find the most recent result dir matching the variant's comment
    DIR=$(grep -l "thickness ${VARIANT}" results/sims/${RESULTS_DATE}/*/run_description.txt 2>/dev/null | \
          head -1 | xargs -r dirname)

    if [ -z "$DIR" ] || [ ! -d "$DIR/solver/checkpoint.bp" ]; then
        echo "SKIP $VARIANT: not found or incomplete in $RESULTS_DATE"
        continue
    fi

    JOB_ID=$(sbatch \
        --job-name="pc_thick_${VARIANT}" \
        --export=ALL,RESULTS_DIR=${PWD}/${DIR},RETAG_SEPTUM=1 \
        run_per_cell.sbatch | awk '{print $4}')
    echo "  $VARIANT -> Job $JOB_ID ($DIR)"
done

echo ""
echo "Monitor with: squeue -u \$USER"
