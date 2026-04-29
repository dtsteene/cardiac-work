#!/bin/bash
# ==============================================================================
# Submit linear-EDPVR optimization for all 7 handover severities in parallel.
#
# Each severity becomes an independent slurm job on the defq partition.
# Results land in data/ukb_circ_linear/, separate from ukb_circ_v2/ so the
# A/B test never clobbers the reference exponential params.
#
# Usage:
#   bash submit_all_circ_optim_linear.sh            # submit with default N_TRIALS
#   bash submit_all_circ_optim_linear.sh 5000       # override N_TRIALS
#   bash submit_all_circ_optim_linear.sh status     # quick status check
# ==============================================================================

# Only the 7 severities used in the handover spectrum (drop healthy_low, borderline)
SEVERITIES="healthy mild moderate severe moderate_severe very_severe end_stage"
N_TRIALS=${1:-3000}

# ------- status mode -------
if [ "$1" = "status" ]; then
    echo "=== Job status ==="
    squeue -u $USER -o "%.8i %.12j %.2t %.10M %.6D %R" | head -30
    echo ""
    echo "=== Linear-run results so far ==="
    TARGET_DIR="/global/D1/homes/dtsteene/cardiac-work/data/ukb_circ_linear"
    for sev in $SEVERITIES; do
        json="$TARGET_DIR/optimized_regazzoni_ukb_linear_${sev}.json"
        if [ -f "$json" ]; then
            summary=$(python3 -c "
import json
d = json.load(open('$json'))
h = d.get('derived_hemodynamics', {})
m = d.get('metrics_achieved', {})
t = d.get('pressure_targets', {})
print(
    f'LV_EF={h.get(\"LV_EF_pct\",0):.1f}%  '
    f'RV_EF={h.get(\"RV_EF_pct\",0):.1f}%  '
    f'LV_EDP={m.get(\"LV_EDP\",0):.1f}/{t.get(\"LV_EDP\",0)}mmHg  '
    f'RV_EDP={m.get(\"RV_EDP\",0):.1f}/{t.get(\"RV_EDP\",0)}mmHg'
)
" 2>/dev/null)
            printf "  %-18s DONE      %s\n" "$sev" "$summary"
        else
            # Is it running right now?
            running=$(squeue -u $USER -n "circ_lin" -h 2>/dev/null | wc -l)
            if [ "$running" -gt 0 ]; then
                printf "  %-18s (running or pending in a circ_lin job)\n" "$sev"
            else
                printf "  %-18s PENDING\n" "$sev"
            fi
        fi
    done
    exit 0
fi

# ------- submit mode -------
echo "Submitting LINEAR (kE=0) circulation optimization for 7 severities"
echo "  N_TRIALS: $N_TRIALS"
echo "  Target dir: data/ukb_circ_linear/"
echo ""

for sev in $SEVERITIES; do
    jobid=$(sbatch --export=SEVERITY=$sev,N_TRIALS=$N_TRIALS \
            --parsable \
            run_circ_optim_linear.sbatch)
    echo "  Submitted $sev -> Job $jobid"
done

echo ""
echo "Monitor with: bash submit_all_circ_optim_linear.sh status"
echo "Or:           squeue -u \$USER"
