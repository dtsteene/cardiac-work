# Workspace Organization Guide

## Directory Structure

```
/home/dtsteene/D1/prelimSim/
├── results/                         # All simulation outputs (organized)
│   ├── results_debug_941908/       # 60 BPM simulation
│   ├── results_debug_941909/       # 75 BPM simulation
│   └── results_debug_JOBID/        # Future results go here
│
├── bpm_comparison/                 # BPM comparison study (LEAN & CLEAN)
│   ├── README.md                   # Quick reference
│   ├── ANALYSIS_RESULTS.md         # Full analysis & recommendations
│   ├── compare_bpm.py              # Analysis script
│   ├── differences.json            # Computed metric differences
│   ├── metrics_*bpm.json           # Metric summaries
│   └── *_comparison.png            # Visualizations
│
├── complete_cycle.py               # Main simulation script
├── complete_cycle.sbatch           # SLURM job submission
├── postprocess.py                  # Post-processing script
│
└── [Other utility files]
```

## Updated Paths

- **Simulation Results**: Now go to `results/results_debug_JOBID/`
- **BPM Analysis**: Look in `bpm_comparison/` - it's lean and organized
- **Future Job Outputs**: Will automatically go to `results/` directory

## Key Changes Made

✅ Moved all `results_debug_*` folders into `results/` directory
✅ Pruned `bpm_comparison/` - removed duplicates and monitor scripts
✅ Updated `complete_cycle.sbatch` to output to `results/` directory
✅ Updated `bpm_comparison/compare_bpm.py` to read from new paths

## Quick Commands

**Re-run BPM comparison analysis:**
```bash
cd /home/dtsteene/D1/prelimSim
python3 bpm_comparison/compare_bpm.py
```

**Submit new simulation:**
```bash
sbatch --export=BPM=60 complete_cycle.sbatch  # Results auto-save to results/
```

**View latest results:**
```bash
ls -lh results/ | tail -5
```
