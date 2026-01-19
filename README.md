# Cardiac Simulation Workflow

Navigation guide for the cardiac-work repository.

## 🎯 Core Simulation

**Main scripts:**
- `complete_cycle.py` - Main cardiac cycle simulation
- `run_sim_and_post.sbatch` - SLURM job submission (runs sim + post)
- `postprocess.py` - Post-processing utilities

**Quick run:**
```bash
sbatch --export=BPM=75 run_sim_and_post.sbatch
```

---

## 📁 Directory Layout

```
cardiac-work/
├── docs/                    # Documentation & notes
├── results/                 # Outputs (animations, logs, sims)
│   ├── animations/          # 3D animation artifacts
│   ├── bpm_comparison/      # Legacy study outputs (kept out of git)
│   ├── log/                 # Job logs
│   └── sims/                # Simulation runs (results_*)
├── scripts/                 # Utility scripts
│
├── complete_cycle.py        # Main simulation script
├── run_sim_and_post.sbatch  # Job submission
├── postprocess.py           # Post-processing
└── README.md                # This file
```

---

## 📖 Documentation

- `docs/WORKSPACE_ORG.md` - Workspace organization guide
- `docs/HANDOVER_NOTES.md` - Project handover notes
- `docs/PROJECT_FILES.md` - Original project file listing

---

## 🚀 Common Tasks

**Submit simulation:**
```bash
sbatch --export=BPM=75 run_sim_and_post.sbatch
```

**Check job status:**
```bash
squeue --me
```

**View latest results:**
```bash
ls -lh results/sims | tail -3
```

## 💾 Data Organization

- **Simulation Results**: `results/sims/results_debug_JOBID/`
  - `output.json` - Hemodynamic data
  - `*.bp/` - Binary parallel data files
  - `*.png` - Diagnostic plots

---

**Last Updated**: January 19, 2026
