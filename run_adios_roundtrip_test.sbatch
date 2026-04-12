#!/bin/bash
# Small test job: verify a serially-generated mesh survives parallel reload.
# Should complete in <5 minutes.

#SBATCH --job-name=adios_roundtrip
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=0:15:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

OUTDIR=${OUTDIR:-/tmp/adios_roundtrip_test}
SIM_DIR="/home/dtsteene/D1/cardiac-work"

# Force unbuffered stdout for everything in the job so we see live progress.
export PYTHONUNBUFFERED=1
exec 1> >(stdbuf -o0 cat)
exec 2> >(stdbuf -o0 cat >&2)

echo "[sbatch] --- adios round-trip test ---"
date
echo "[sbatch] Outdir: $OUTDIR"
echo "[sbatch] Partition: $SLURM_JOB_PARTITION  Node: $SLURM_JOB_NODELIST  Tasks: $SLURM_NTASKS"

echo "[sbatch] module load anaconda3..."
module load anaconda3
echo "[sbatch] conda shell hook..."
eval "$(conda shell.bash hook)"
echo "[sbatch] conda activate RV..."
conda activate RV
echo "[sbatch] conda env: $CONDA_DEFAULT_ENV, prefix: $CONDA_PREFIX"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPICH_HYDRA_LAUNCHER=fork

cd $SIM_DIR
echo "[sbatch] starting mpirun..."
date
time $CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} -launcher fork \
    $CONDA_PREFIX/bin/python3 -u test_adios_roundtrip.py verify --outdir "$OUTDIR"

echo "[sbatch] mpirun returned, exit code: $?"
date
