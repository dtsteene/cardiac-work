#!/bin/bash
#SBATCH --job-name=sweep_knobs
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=/home/dtsteene/D1/cardiac-work/results/log/slurm_sweep_%j.out

# --- Parameter Sweep Script ---
# Submits 3 configurations to compare Mesh Resolution vs Metrics Space

# Knob 1: Mesh Resolution (char_length)
#   - 10.0: Standard UKB (Fast, ~35 min)
#   - 5.0:  High Fidelity (Slow, ~42 min)

# Knob 2: Metrics Function Space (metrics_space)
#   - DG0: Piecewise Constant (High spatial correlation, lower magnitude)
#   - DG1: Piecewise Linear (High energy magnitude, potentially noisy septum)

# --- Configuration A: Standard Control ---
# Replicates the "Fast & Good Correlation" run we had earlier
# Mesh: 10.0, Metrics: DG0
sbatch --export=BPM=75,CHAR_LENGTH=10.0,METRICS_SPACE="DG0",COMMENT="Control: UKB 10mm + DG0" /home/dtsteene/D1/cardiac-work/run_sim_and_post.sbatch

# --- Configuration B: High Res Stability ---
# Your current setup: Fine mesh, but stable DG0 metrics
# Mesh: 5.0, Metrics: DG0
sbatch --export=BPM=75,CHAR_LENGTH=5.0,METRICS_SPACE="DG0",COMMENT="High Res: 5mm + DG0" /home/dtsteene/D1/cardiac-work/run_sim_and_post.sbatch

# --- Configuration C: High Energy Fidelity ---
# The "Energy" experiment: Fine mesh + Linear metrics
# Mesh: 5.0, Metrics: DG1
sbatch --export=BPM=75,CHAR_LENGTH=5.0,METRICS_SPACE="DG1",COMMENT="High Energy: 5mm + DG1" /home/dtsteene/D1/cardiac-work/run_sim_and_post.sbatch

echo "All 3 sweep jobs submitted."
