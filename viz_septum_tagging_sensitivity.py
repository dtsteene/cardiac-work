#!/usr/bin/env python3
"""
viz_septum_tagging_sensitivity.py — Section B, plot 4:
How much does each septum definition move when you tag on the
end-diastolic (loaded) mesh instead of on the canonical (unloaded)
reference mesh?

For each of the 7 spectrum cases, load both the canonical-tagged and the
ED-tagged per_cell_data files and count cells in three septum definitions:

    LDRB-native       region_tags == 3 (LDRB algorithm's own tag)
    LDRB-loose        is_ldrb_septum (Laplace-based, wider)
    geometric         is_geometric_septum  (max(d_LV,d_RV) < d_epi)

Canonical cell counts are identical across cases by construction (they come
from a shared reference). ED cell counts vary per case because the tagging
is recomputed on each case's loaded mesh. The point of this figure is to
show that the geometric definition is sensitive to preload (cell count
moves meaningfully across severities) while the LDRB definitions are
robust.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/analysis/spectrum")
OUT.mkdir(parents=True, exist_ok=True)

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

# ── Figure ──────────────────────────────────────────────────────────────────
X = np.arange(len(SPECTRUM))
xtick_labels = [f"{lbl}\n({rv:.0f} mmHg)" for lbl, rv in zip(labels, rv_esps)]

STYLE = {
    "LDRB-native": dict(color="#2ca02c", marker="o"),
    "LDRB-loose":  dict(color="#9467bd", marker="s"),
    "geometric":   dict(color="#d62728", marker="^"),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                          sharey=True, constrained_layout=True)
for j, (tag_mode, d) in enumerate([("Canonical (reference mesh)", canon),
                                     ("ED (loaded mesh)",           ed)]):
    ax = axes[j]
    for name in ("LDRB-native", "LDRB-loose", "geometric"):
        vals = np.array(d[name])
        ax.plot(X, vals, "-", ms=8, lw=2.0,
                color=STYLE[name]["color"],
                marker=STYLE[name]["marker"],
                label=name)
    ax.set_xticks(X)
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.set_title(tag_mode, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.set_ylim(0, max(max(canon["LDRB-loose"]), max(ed["LDRB-loose"])) * 1.15)
    if j == 0:
        ax.set_ylabel("Septum cell count", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

fig.suptitle("Septum cell count — sensitivity to tagging instant",
              fontsize=12, fontweight="bold")
fig.savefig(OUT / "fig_septum_tagging_sensitivity.png", dpi=160,
             bbox_inches="tight")
fig.savefig(OUT / "fig_septum_tagging_sensitivity.pdf", bbox_inches="tight")
print(f"\nSaved {OUT / 'fig_septum_tagging_sensitivity.png'}")

np.savez(OUT / "septum_tagging_raw.npz",
          canonical_LDRB_native=np.array(canon["LDRB-native"]),
          canonical_LDRB_loose=np.array(canon["LDRB-loose"]),
          canonical_geometric=np.array(canon["geometric"]),
          ED_LDRB_native=np.array(ed["LDRB-native"]),
          ED_LDRB_loose=np.array(ed["LDRB-loose"]),
          ED_geometric=np.array(ed["geometric"]),
          labels=np.array(labels), rv_esp=np.array(rv_esps))
