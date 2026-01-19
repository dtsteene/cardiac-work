# Preliminary Cardiac Simulation Studies

Quick navigation guide for the prelimSim workspace.

## 🎯 Core Simulation

**Main scripts:**
- `complete_cycle.py` - Main cardiac cycle simulation
- `complete_cycle.sbatch` - SLURM job submission (milanq partition)
- `postprocess.py` - Post-processing utilities

**Quick run:**
```bash
sbatch --export=BPM=75 complete_cycle.sbatch
```

---

## 📊 Studies & Analysis

### BPM Comparison Study (60 vs 75 BPM)
- **Location**: `bpm_comparison/`
- **Key Result**: 60 BPM acceptable for exploratory work
- **Run**: `python3 bpm_comparison/compare_bpm.py`
- **Read**: `bpm_comparison/README.md` or `bpm_comparison/ANALYSIS_RESULTS.md`

---

## 📁 Directory Layout

```
prelimSim/
├── bpm_comparison/          # 60 vs 75 BPM comparison study
├── results/                 # Simulation outputs (organized by job)
├── animations/              # 3D animation generation
├── docs/                    # Documentation & notes
├── scripts/                 # Utility scripts
├── log/                     # Job logs
│
├── complete_cycle.py        # Main simulation script
├── complete_cycle.sbatch    # Job submission
├── postprocess.py           # Post-processing
└── README.md               # This file
```

---

## 📖 Documentation

- `docs/WORKSPACE_ORG.md` - Workspace organization guide
- `docs/HANDOVER_NOTES.md` - Project handover notes
- `docs/PROJECT_FILES.md` - Original project file listing
- `bpm_comparison/ANALYSIS_RESULTS.md` - Full BPM study analysis

---

## 🚀 Common Tasks

**Submit simulation:**
```bash
sbatch --export=BPM=75 complete_cycle.sbatch
```

**Check job status:**
```bash
squeue --me
```

**View latest results:**
```bash
ls -lh results/ | tail -3
```

**Re-run BPM analysis:**
```bash
python3 bpm_comparison/compare_bpm.py
```

**Generate animations:**
```bash
sbatch animations/generate_3d_animation.sbatch
```

---

## 💾 Data Organization

- **Simulation Results**: `results/results_debug_JOBID/`
  - `output.json` - Hemodynamic data
  - `*.bp/` - Binary parallel data files
  - `*.png` - Diagnostic plots

---

**Last Updated**: January 19, 2026
