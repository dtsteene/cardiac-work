#!/bin/bash
# Runs analyze_transventricular.py twice on the 8-case spectrum:
#   - default (per_cell_data.npz, tag-at-ED) -> results/analysis/transventricular_ed/
#   - --input-tag canonical                  -> results/analysis/transventricular_canonical/
# Both produce the same set of figures; the two output dirs can be compared
# side-by-side to check whether the transmural mechanism story is robust to
# the choice of tagging frame.
set -e
cd /home/dtsteene/D1/cardiac-work

CASES=(
  results/sims/2026-04-08/UKB_6beats_run_1017516
  results/sims/2026-04-08/UKB_6beats_run_1017517
  results/sims/2026-04-08/UKB_6beats_run_1017525
  results/sims/2026-04-08/UKB_6beats_run_1017519
  results/sims/2026-04-08/UKB_6beats_run_1017520
  results/sims/2026-04-08/UKB_6beats_run_1017521
  results/sims/2026-04-08/UKB_6beats_run_1017522
  results/sims/2026-04-08/UKB_6beats_run_1017523
)

echo "=============================================================="
echo "RUN 1: ED tagging  (default per_cell_data.npz)"
echo "=============================================================="
python3 analyze_transventricular.py \
    --output-dir results/analysis/transventricular_ed \
    "${CASES[@]}" 2>&1 | tail -30

echo
echo "=============================================================="
echo "RUN 2: Canonical (reference-config) tagging  (--input-tag canonical)"
echo "=============================================================="
python3 analyze_transventricular.py \
    --input-tag canonical \
    --output-dir results/analysis/transventricular_canonical \
    "${CASES[@]}" 2>&1 | tail -30

echo
echo "Done. Outputs:"
echo "  results/analysis/transventricular_ed/"
echo "  results/analysis/transventricular_canonical/"
