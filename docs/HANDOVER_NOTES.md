# Cardiac FEM Simulation Project - Agent Handover Notes

**Project Status**: Active development with working simulation pipeline
**Last Updated**: January 16, 2026
**Key Contact**: dtsteene (primary developer)

---

## 🎯 Project Overview

This is a **multiscale cardiac electromechanics simulation** combining:
- **3D Mechanics**: FEniCSx/Dolfinx finite element solver with Pulse library
- **0D Circulation**: Regazzoni2020 closed-loop lumped-parameter model  
- **Material Model**: HolzapfelOgden anisotropic hyperelastic constitutive law
- **Fiber Field**: LDRB algorithm for myocardial anisotropy
- **Geometry**: UK Biobank BiV mesh (~2,150 elements)

**Main Objective**: Simulate 3D cardiac mechanics coupled to 0D hemodynamics with validation against physiological parameters.

---

## ⚠️ CRITICAL PITFALLS (Learn from Our Mistakes)

### 1. **MPI LAUNCHER BUG** 🚨 (Already Fixed)

**THE PROBLEM:**
```bash
# ❌ WRONG - This fails silently or crashes
time $CONDA_DEFAULT_ENV/bin/mpirun -n ${MY_CORES} python script.py
# Error: /path/RV/bin/mpirun: No such file or directory
```

**ROOT CAUSE:** After `conda activate RV`, the variable `$CONDA_DEFAULT_ENV` contains only the environment name `"RV"`, NOT the full path.

**THE SOLUTION:**
```bash
# ✅ CORRECT - Use $CONDA_PREFIX instead
time $CONDA_PREFIX/bin/mpirun -np ${MY_CORES} -launcher fork python script.py

# Or add this environment variable
export MPICH_HYDRA_LAUNCHER=fork
```

**WHERE FIXED:**
- `/home/dtsteene/D1/prelimSim/complete_cycle.sbatch` (line 53)
- `/home/dtsteene/D1/sims/run_sims.sbatch` (lines 80, 124)

**VERIFICATION:** Job 941147 ran successfully with this fix.

---

### 2. **Odd Image Dimensions in ffmpeg**

**THE PROBLEM:**
```
[libx264] width not divisible by 2 (1389x1000)
Error initializing output stream
```

**THE SOLUTION:** Use ffmpeg's scale filter:
```bash
ffmpeg -pattern_type glob -i 'frames/*.png' \
  -vf "scale=1388:1000" \
  -c:v libx264 -pix_fmt yuv420p -r 20 output.mp4
```

**KEY LEARNING:** H.264 codec requires even dimensions. Always scale to even numbers.

---

### 3. **XDMF Grid Name Case Sensitivity**

**THE PROBLEM:**
```python
# ❌ FAILS - Looks for grid named "mesh"
with dolfinx.io.XDMFFile(comm, "geometry/mesh.xdmf", "r") as f:
    mesh = f.read_mesh()
# Error: <Grid> with name 'mesh' not found
```

**SOLUTION:** The actual grid is named `"Mesh"` (capitalized) in the XDMF file. Use HDF5 direct reading instead:
```python
import h5py
with h5py.File("geometry/mesh.h5", "r") as f:
    # Parse topology and geometry directly
    pass
```

**LESSON:** Always inspect XDMF structure with `head` before assuming grid names.

---

### 4. **Simulation Speed Expectations**

**REALITY CHECK:** A single 1-beat cardiac cycle with ~800 timesteps takes **~45 minutes** on 8 cores.

**WHY?** Coupled 3D/0D nonlinear solve at each timestep. This is normal, not a bug.

**DO NOT** try to optimize for speed initially - validate physics first. Once validated, profile and optimize.

---

### 5. **Volume Offset Calibration**

**THE PROBLEM:** Initial LV/RV volumes from geometry don't match target physiological volumes.

**THE SOLUTION:** Create calibration scripts that:
1. Run initialization phase only
2. Measure actual chamber volumes  
3. Apply offset: `V_target = V_measured + offset`
4. Store offset in parameters JSON

**FILES:** 
- `verify_passive_filling.py` - Validates volume calibration
- Offset stored in `data/healthy_circulation_params.json`

---

## 📁 Critical File Locations

### Simulation Scripts
```
/home/dtsteene/D1/prelimSim/
├── complete_cycle.py          # Main simulation (3D/0D coupled)
├── complete_cycle.sbatch      # SLURM wrapper for complete_cycle.py
├── postprocess.py             # Generate visualizations (PNG)
├── create_animations_simple.py # Generate animation frames
├── SIMULATION_REPORT.md       # Full technical report
└── INDEX.md                   # File guide and quick reference

/home/dtsteene/D1/sims/
└── run_sims.sbatch            # Full pipeline wrapper (tuning → simulation)
```

### Geometry & Data
```
/home/dtsteene/cardiac-work-fem-comparison/
├── data/
│   ├── healthy.h5 / healthy.xdmf      # BiV mesh
│   ├── healthy_circulation_params.json # 0D model parameters
│   └── [other data files]
└── simpler_geometries/                 # Simplified test geometries
```

### Results Storage
```
/home/dtsteene/D1/prelimSim/results_debug_941147/
├── geometry/          # mesh.h5, mesh.xdmf (206 KB)
├── output.json        # Hemodynamic data (pressures, volumes)
├── output.bp          # 3D displacement field (binary format)
├── time.txt           # Timestep values
└── postprocessing_summary.json # Calculated metrics
```

---

## 🚀 How to Run Simulations

### Quick Start (Single Beat)
```bash
cd /home/dtsteene/D1/prelimSim

# Method 1: Direct submission
sbatch --export=BPM=75 complete_cycle.sbatch

# Method 2: Check status
squeue --me
```

### Full Pipeline (Not Yet Active)
```bash
cd /home/dtsteene/D1/sims
sbatch run_sims.sbatch
# Runs: geometry → tuning → simulation → post-processing
# Currently stops after tuning (see line ~68 in sbatch)
```

### Important sbatch Parameters

In `complete_cycle.sbatch`:
```bash
#SBATCH --ntasks=8          # Number of MPI processes
#SBATCH --time=02:00:00     # 2 hours (enough for 1 beat @ BPM=75)
#SBATCH --partition=gpu     # Which partition (check with `sinfo`)
```

---

## 🔧 Environment Setup

### Conda Environment
```bash
# The RV environment is already set up at:
conda activate /global/D1/homes/dtsteene/conda-envs/RV

# Key packages:
# - fenics-dolfinx (3.0+)
# - pulse (cardiac mechanics)
# - cardiac-geometries (UK Biobank meshes)
# - adios4dolfinx (I/O for large fields)
# - matplotlib, pyvista, ffmpeg, imagemagick
```

### Critical Environment Variables (in sbatch)
```bash
export FFCX_JIT_LOCK_TIMEOUT=60  # FEniCSx JIT compilation timeout
export OPENBLAS_NUM_THREADS=1    # 1 thread per MPI process
export OMP_NUM_THREADS=1         # Prevent thread contention
export MPICH_HYDRA_LAUNCHER=fork # MPI launcher for proper fork
```

---

## 📊 Understanding the Output Files

### `output.json` Structure
```json
{
  "p_LV": [array of 800 LV pressure values (mmHg)],
  "V_LV": [array of 800 LV volume values (mL)],
  "p_RV": [array of 800 RV pressure values],
  "V_RV": [array of 800 RV volume values],
  "p_LA": [array of LA pressures],
  "p_RA": [array of RA pressures]
}
```

### Physiological Validation Ranges
```
LEFT VENTRICLE:
  - Peak systolic: 90-140 mmHg ✓
  - End diastolic: 5-12 mmHg ✓
  - Ejection fraction: 50-70% ✓
  - Stroke volume: 60-100 mL ✓

RIGHT VENTRICLE:
  - Peak systolic: 15-30 mmHg ✓
  - End diastolic: 2-8 mmHg ✓
  - Stroke volume: 40-80 mL ✓
```

**Job 941147 Results** (PASSED all checks):
- LV: 7.1-116.7 mmHg, EF 58.1%, SV 66.5 mL ✅
- RV: 4.2-23.8 mmHg, SV 45.6 mL ✅

---

## 📈 Post-Processing Pipeline

### Step 1: Generate Visualizations (PNG)
```bash
cd /home/dtsteene/D1/prelimSim
python postprocess.py
# Outputs: pv_loop_analysis.png, hemodynamics_timeseries.png
```

### Step 2: Generate Animation Frames (800 PNG frames)
```bash
python create_animations_simple.py
# Generates: animations_941147/pv_loops/*.png (800 frames)
# Runtime: ~8 minutes
```

### Step 3: Create MP4 Video
```bash
cd animations_941147
ffmpeg -pattern_type glob -i 'pv_loops/*.png' \
  -vf "scale=1388:1000" \
  -c:v libx264 -pix_fmt yuv420p -r 20 output.mp4
```

---

## 🛠️ Common Tasks & Solutions

### Task: Run with Different Heart Rate
```bash
sbatch --export=BPM=90 complete_cycle.sbatch
# Or modify in complete_cycle.py: line that sets BPM
```

### Task: Run Multiple Beats
```bash
# Edit /home/dtsteene/D1/prelimSim/complete_cycle.py
# Change: num_beats = 1
# To:     num_beats = 3
# (Warning: 3 beats = ~3.5 hours)

sbatch complete_cycle.sbatch
```

### Task: Monitor Active Job
```bash
# Real-time status
watch -n 5 squeue --me

# View job output
tail -f /home/dtsteene/D1/prelimSim/slurm-<JOBID>.out

# Check stderr
tail -f /home/dtsteene/D1/prelimSim/slurm-<JOBID>.err
```

### Task: Debug a Failed Job
```bash
# 1. Check SLURM output
cat slurm-<JOBID>.out

# 2. Search for common errors
grep -i "error\|failed\|exception" slurm-<JOBID>.out

# 3. Check if it's MPI launcher
grep "mpirun" slurm-<JOBID>.out

# 4. Verify environment variables
grep "CONDA_PREFIX\|CONDA_DEFAULT_ENV" slurm-<JOBID>.out
```

### Task: Check Available Compute Resources
```bash
# View all partitions and their status
sinfo

# View your quota
sinfo --partition=<partition> --long

# View job history
sacct --user=$USER --format=JobID,JobName,State,Elapsed,Partition
```

---

## 📝 Key Code Patterns

### Running MPI Python Job (Correct)
```bash
$CONDA_PREFIX/bin/mpirun -np $NPROCS -launcher fork python script.py
```

### Reading Hemodynamic Output in Python
```python
import json
import numpy as np

with open("output.json") as f:
    data = json.load(f)

lv_pressure = np.array(data["p_LV"])
lv_volume = np.array(data["V_LV"])
```

### Reading 3D Displacement Field (adios4dolfinx)
```python
from mpi4py import MPI
import adios4dolfinx
import dolfinx

comm = MPI.COMM_WORLD

# Read mesh from XDMF
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(
    "geometry/mesh.xdmf", comm, rank=0)

# Read displacement at specific timestep
u = adios4dolfinx.read_function(
    "output.bp", u_function, time=t_specific, name="u")
```

---

## 🎓 Learning Resources

### Key References in Code
- `/home/dtsteene/D1/prelimSim/SIMULATION_REPORT.md` - Full technical details
- `/home/dtsteene/D1/prelimSim/INDEX.md` - File structure and references
- `postprocess.py` - Clean example of reading/plotting output

### External Documentation
- FEniCSx: https://docs.fenicsproject.org/
- Pulse: https://pulse-docker.readthedocs.io/
- ADIOS2: https://adios2.readthedocs.io/

---

## ✅ Pre-Flight Checklist Before Submitting Jobs

- [ ] `$CONDA_PREFIX` resolves correctly (run `echo $CONDA_PREFIX`)
- [ ] MPI launcher uses `-launcher fork` flag
- [ ] Thread environment variables are set (OMP_NUM_THREADS=1)
- [ ] Job time limit is sufficient (1 beat ≈ 45 min on 8 cores)
- [ ] Output directory exists and is writable
- [ ] Geometry files are readable at specified paths
- [ ] Check `squeue` for partition availability before submission

---

## 🔴 Red Flags to Watch For

1. **"mpirun: No such file or directory"** → Check `$CONDA_PREFIX` vs `$CONDA_DEFAULT_ENV`
2. **Simulation progresses but then stops** → Check disk space on /work or /global
3. **Takes >2 hours for 1 beat** → Job may be I/O bound; check cluster load
4. **Output files are 0 bytes** → Check error logs; likely a crash during initialization
5. **ffmpeg fails with odd dimensions** → Use scale filter in command
6. **Pressure values are negative or unrealistic** → Check material parameters and boundary conditions

---

## 📞 When Things Go Wrong

**First Steps:**
1. Read the SLURM output file (`slurm-<JOBID>.out`)
2. Check if it's an MPI issue (try with fewer cores: `-np 1`)
3. Verify environment variables are exported
4. Run a quick test with 1 MPI process on login node (if allowed)

**Escalation:**
- Check `/home/dtsteene/D1/prelimSim/complete_cycle.py` for recent changes
- Compare to last known working version (git log or backup)
- Test with simpler geometry first (ellipsoid or cylinder)

---

## 📋 Project Continuation Ideas

1. **Multi-beat simulation**: Validate model for 2-3 consecutive beats
2. **Parameter sensitivity study**: Vary material parameters systematically
3. **3D animation**: Fix mesh loading to generate displacement animations
4. **Pathology modeling**: Run diseased heart models (PAH parameters in data/)
5. **Optimization**: Auto-tune material parameters to match clinical data

---

## 🎯 Most Important Takeaway

**The MPI launcher bug** is the most likely issue you'll encounter. If a job fails:
1. First check: Is `$CONDA_PREFIX` correct?
2. Second check: Is `-launcher fork` in mpirun?
3. Third check: Are thread variables set to 1?

Everything else is usually physics/I/O related and much easier to debug once the MPI part works.

---

**Good luck with the project! The simulation pipeline is now working and physiologically validated. 🚀**
