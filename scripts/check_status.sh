#!/bin/bash
# SIMPLE STATUS CHECK - Run this to monitor simulations
# Usage: bash check_status.sh

echo "=== CHECKING SIMULATION STATUS ==="
echo ""

# 1. RUNNING JOBS?
echo "1. Running jobs:"
squeue --me | tail -1
if squeue --me | grep -q "RUNNING"; then
    echo "   ✓ Jobs running"
    echo ""
    # Show recent log output
    echo "2. Recent log activity:"
    for f in log/*.out; do
        if [ -f "$f" ]; then
            echo "   $f: $(wc -l < $f) lines"
            tail -2 "$f" | head -1
        fi
    done
else
    echo "   ✗ No running jobs"
fi

echo ""
echo "3. Checking for completed results:"
ls -d results_debug_* 2>/dev/null | while read dir; do
    if [ -f "$dir/output.json" ]; then
        echo "   ✓ $dir - HAS output.json (READY TO POSTPROCESS)"
    else
        echo "   - $dir - empty or incomplete"
    fi
done

echo ""
echo "4. Last sbatch submitted:"
ls -t log/*.out 2>/dev/null | head -1 | xargs -I {} sh -c 'echo "   {}"; grep "Running" {} | tail -1'

echo ""
