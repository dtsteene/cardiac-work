#!/usr/bin/env python3
"""
table_septum_tagging_sensitivity.py — Septum tagging sensitivity, TABLE form.

For each of the 7 spectrum cases, load both the canonical-tagged and the
ED-tagged per_cell_data files and count cells in three septum definitions:

    LDRB-native   region_tags == 3         (LDRB algorithm's own tag)
    LDRB-loose    is_ldrb_septum           (Laplace-based, wider)
    geometric     is_geometric_septum      (max(d_LV,d_RV) < d_epi)

Canonical cell counts are identical across cases by construction. ED cell
counts vary per case because tagging is recomputed on each case's loaded
mesh. The story: geometric is sensitive to preload, LDRB is robust.

TODO — table to produce for the thesis:
  [ ] Cell counts per case for each (tag-mode, septum definition) pair
  [ ] Jaccard overlap between canonical and ED tag sets, per definition
  [ ] Extend tag-modes beyond {canonical, ED-beat5}:
        - ED during the beat at different preload/stress points
        - Unstressed ED (reference config) geometry tagging
      to quantify what actually matters about WHEN you tag.
  [ ] CV of cell count across the severity spectrum, per tag-mode
Output: pure stdout table (no figure). Copy/format for LaTeX later.
"""
import numpy as np
from pathlib import Path

SPECTRUM = [
    ("healthy",         "Borderline PH", "1020849", 30.6),
    ("mild",            "Mild",          "1020851", 38.2),
    ("moderate",        "Moderate",      "1020852", 55.4),
    ("moderate_severe", "Mod–severe",    "1020853", 62.5),
    ("severe",          "Severe",        "1020854", 70.8),
    ("very_severe",     "Very severe",   "1020855", 85.0),
    ("end_stage",       "End-stage",     "1020856", 88.4),
]
ROOT = Path("results/sims/2026-04-12")


def septum_counts(pc):
    return {
        "LDRB-native": int((pc["region_tags"] == 3).sum()),
        "LDRB-loose":  int(pc["is_ldrb_septum"].sum()),
        "geometric":   int(pc["is_geometric_septum"].sum()),
    }


canon = {"LDRB-native": [], "LDRB-loose": [], "geometric": []}
ed    = {"LDRB-native": [], "LDRB-loose": [], "geometric": []}
labels, rv_esps = [], []
for _sev, label, run_id, rve in SPECTRUM:
    pc_c = np.load(ROOT / f"UKB_6beats_run_{run_id}" / "per_cell_data.npz",
                    allow_pickle=True)
    pc_e = np.load(ROOT / f"UKB_6beats_run_{run_id}" / "per_cell_data_ed_beat5.npz",
                    allow_pickle=True)
    cc = septum_counts(pc_c)
    ec = septum_counts(pc_e)
    for k in canon:
        canon[k].append(cc[k])
        ed[k].append(ec[k])
    labels.append(label)
    rv_esps.append(rve)

print(f"{'case':<16} {'LDRB-native (can/ED)':>24} "
      f"{'LDRB-loose (can/ED)':>24} {'geometric (can/ED)':>24}")
for i, lbl in enumerate(labels):
    rn, re = canon["LDRB-native"][i], ed["LDRB-native"][i]
    ln, le = canon["LDRB-loose"][i],  ed["LDRB-loose"][i]
    gn, ge = canon["geometric"][i],   ed["geometric"][i]
    print(f"{lbl:<16} {f'{rn}/{re}':>24} {f'{ln}/{le}':>24} {f'{gn}/{ge}':>24}")

# Relative variation summary — coefficient of variation across the 7 cases
print("\nCoefficient of variation across 7 severities (std / mean):")
for tag_mode, d in [("canonical", canon), ("ED", ed)]:
    for name in ("LDRB-native", "LDRB-loose", "geometric"):
        vals = np.array(d[name])
        cv = vals.std() / vals.mean() if vals.mean() else 0
        print(f"  {tag_mode:<10} {name:<15} mean={vals.mean():.0f}  "
              f"std={vals.std():.1f}  CV={cv*100:.1f}%")
