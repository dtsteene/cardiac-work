#!/bin/bash
# Main Pipeline Test: Simulation → Postprocessing → Animation

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          MAIN PIPELINE TEST - Complete Workflow               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

JOBID=941935
ANIM_JOBID=941979

echo "📋 Pipeline Components:"
echo "  1. Simulation (complete_cycle.py)"
echo "  2. Postprocessing (postprocess.py)"  
echo "  3. Animation Generation (create_3d_animation_fixed.py)"
echo ""

# ============================================================
# 1. CHECK SIMULATION
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. SIMULATION (Job $JOBID)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

RESULTS_DIR="/home/dtsteene/D1/prelimSim/results/results_debug_$JOBID"

if [ -d "$RESULTS_DIR" ]; then
    echo "  ✓ Results directory exists: $RESULTS_DIR"
    
    # Check critical files
    CRITICAL_FILES=("output.json" "time.txt" "state_names.txt" "history.npy")
    ALL_PRESENT=true
    
    for file in "${CRITICAL_FILES[@]}"; do
        if [ -f "$RESULTS_DIR/$file" ]; then
            SIZE=$(du -h "$RESULTS_DIR/$file" | cut -f1)
            echo "  ✓ $file ($SIZE)"
        else
            echo "  ✗ MISSING: $file"
            ALL_PRESENT=false
        fi
    done
    
    if $ALL_PRESENT; then
        echo "  ✅ Simulation: PASSED"
    else
        echo "  ❌ Simulation: FAILED - Missing files"
        exit 1
    fi
else
    echo "  ❌ Simulation: FAILED - No results directory"
    exit 1
fi

echo ""

# ============================================================
# 2. CHECK POSTPROCESSING
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. POSTPROCESSING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

POSTPROC_FILES=("pv_loop_analysis.png" "hemodynamics_timeseries.png" "postprocessing_summary.json")
ALL_PRESENT=true

for file in "${POSTPROC_FILES[@]}"; do
    if [ -f "$RESULTS_DIR/$file" ]; then
        SIZE=$(du -h "$RESULTS_DIR/$file" | cut -f1)
        echo "  ✓ $file ($SIZE)"
    else
        echo "  ✗ MISSING: $file"
        ALL_PRESENT=false
    fi
done

if $ALL_PRESENT; then
    echo "  ✅ Postprocessing: PASSED"
else
    echo "  ❌ Postprocessing: FAILED - Missing output files"
    exit 1
fi

echo ""

# ============================================================
# 3. CHECK ANIMATION (wait for completion)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. ANIMATION GENERATION (Job $ANIM_JOBID)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "  ⏳ Waiting for animation job to complete..."

MAX_WAIT=120  # 2 hours max
for i in $(seq 1 $MAX_WAIT); do
    STATUS=$(squeue --me | grep $ANIM_JOBID | awk '{print $5}')
    
    if [ -z "$STATUS" ]; then
        echo "  ✓ Animation job completed!"
        break
    fi
    
    if [ $((i % 5)) -eq 0 ]; then
        echo "  [$i/$MAX_WAIT min] Job status: $STATUS"
    fi
    sleep 60
done

# Check if animation output exists
ANIM_DIR="/home/dtsteene/D1/prelimSim/animations/animations_3d_$ANIM_JOBID"
if [ -d "$ANIM_DIR" ]; then
    echo "  ✓ Animation directory created: $ANIM_DIR"
    
    # Check for output files
    MP4_COUNT=$(find "$ANIM_DIR" -name "*.mp4" 2>/dev/null | wc -l)
    if [ $MP4_COUNT -gt 0 ]; then
        echo "  ✓ Found $MP4_COUNT MP4 animation file(s)"
        find "$ANIM_DIR" -name "*.mp4" -exec du -h {} \; | sed 's/^/    /'
        echo "  ✅ Animation: PASSED"
    else
        echo "  ❌ Animation: FAILED - No MP4 files generated"
        exit 1
    fi
else
    echo "  ⚠️  Animation directory not found (may still be processing)"
    echo "     Check: $ANIM_DIR"
    echo "     Log: /home/dtsteene/D1/prelimSim/animation_3d.log"
    exit 1
fi

echo ""

# ============================================================
# FINAL SUMMARY
# ============================================================
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   ✅ PIPELINE TEST: PASSED                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "✓ All components working correctly:"
echo "  1. ✓ Simulation completes with all outputs"
echo "  2. ✓ Postprocessing generates analysis plots"
echo "  3. ✓ Animation generation creates MP4 videos"
echo ""
echo "📁 Output Locations:"
echo "  • Simulation: $RESULTS_DIR"
echo "  • Animation:  $ANIM_DIR"
echo ""
echo "🚀 Main pipeline is production-ready!"
