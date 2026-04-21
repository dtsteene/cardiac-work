#!/bin/bash
# ==============================================================================
# Submit all severity levels for circulation optimization
#
# Usage:
#   bash submit_all_circ_optim.sh            # submit all 6 severities
#   bash submit_all_circ_optim.sh 5000       # override N_TRIALS
#
# Monitor:
#   squeue -u $USER -o "%.8i %.12j %.2t %.10M %.6D %R"
#   tail -f circ_optim_circ_healthy_*.out    # watch a specific job
#   bash submit_all_circ_optim.sh status     # quick status check
# ==============================================================================

SEVERITIES="healthy_low healthy mild moderate moderate_severe severe very_severe end_stage"
N_TRIALS=${1:-2500}

# Status check mode
if [ "$1" = "status" ]; then
    echo "=== Job Status ==="
    squeue -u $USER -o "%.8i %.16j %.2t %.10M %.6D %R" | head -20
    echo ""
    echo "=== Results so far ==="
    RESULTS_DIR="/global/D1/homes/dtsteene/cardiac-work/results_mesh_ukb"
    V2_DIR="/global/D1/homes/dtsteene/cardiac-work/data/ukb_circ_v2"
    for sev in $SEVERITIES; do
        json="$V2_DIR/optimized_regazzoni_ukb_${sev}.json"
        if [ -f "$json" ]; then
            ef=$(python3 -c "
import json
d = json.load(open('$json'))
h = d.get('derived_hemodynamics', {})
m = d.get('metrics_achieved', {})
print(f'LV_EF={h.get(\"LV_EF_pct\",0):.1f}%  RV_EF={h.get(\"RV_EF_pct\",0):.1f}%  CO={h.get(\"CO_Lpm\",0):.1f}L/min  SV={m.get(\"SV\",0):.1f}mL')
" 2>/dev/null)
            printf "  %-20s DONE  %s\n" "$sev" "$ef"
        else
            # Check if running
            running=$(squeue -u $USER -n "circ_${sev}" -h 2>/dev/null | wc -l)
            if [ "$running" -gt 0 ]; then
                printf "  %-20s RUNNING\n" "$sev"
            else
                printf "  %-20s PENDING\n" "$sev"
            fi
        fi
    done
    exit 0
fi

echo "Submitting circulation optimization for all severities"
echo "  N_TRIALS: $N_TRIALS"
echo ""

for sev in $SEVERITIES; do
    jobid=$(sbatch --export=SEVERITY=$sev,N_TRIALS=$N_TRIALS \
            --parsable \
            run_circ_optim.sbatch)
    echo "  Submitted $sev -> Job $jobid"
done

echo ""
echo "Monitor with: bash submit_all_circ_optim.sh status"
echo "Or:           squeue -u \$USER"
