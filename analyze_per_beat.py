#!/usr/bin/env python3
"""
analyze_per_beat.py — Per-beat convergence analysis for proxy correlations.

For each beat b in 0..N-1, aggregate per-cell data from all cases and compute
r_PLV(b), r_PRV(b), r_Trans(b) across the disease spectrum at several septum
definitions (geometric, LDRB, and the full t-threshold sweep).

Inputs: per_cell_data_beat0.npz ... per_cell_data_beat{N-1}.npz in each case dir.
These are produced by running compute_per_cell.py --beat N for each beat.

Output: per_beat_convergence.png and per_beat_raw.npz

Usage:
    python3 analyze_per_beat.py results/sims/2026-04-12/UKB_6beats_run_*
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("result_dirs", nargs="+", type=Path)
parser.add_argument("--output-dir", type=Path, default=None)
args = parser.parse_args()

out_dir = args.output_dir or Path("results/analysis/per_beat")
out_dir.mkdir(parents=True, exist_ok=True)


def safe_r(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return pearsonr(x, y)[0]


# ── Load per-beat data ──────────────────────────────────────────────────────
# For each case, find all per_cell_data_beat{N}.npz files. All cases must have
# the same set of beats.
cases = []
for d in args.result_dirs:
    d = d.resolve()
    beat_files = sorted(d.glob("per_cell_data_beat*.npz"))
    if len(beat_files) == 0:
        print(f"  SKIP {d.name}: no per_cell_data_beat*.npz")
        continue

    # Determine case label
    desc = (d / "run_description.txt").read_text().strip() if (d / "run_description.txt").exists() else d.name
    for pfx in ["Phase1 shared-mesh v2circ ", "Phase1 v2circ ", "v2circ "]:
        desc = desc.replace(pfx, "")
    label = desc.split()[0]

    # Solver pressure for RV_ESP ordering
    sp_path = d / "solver" / "solver_cavity_pressure_mmHg.npy"
    if not sp_path.exists():
        sp_path = d / "solver" / "pressure_history.npy"
    sp = np.load(sp_path)
    rv_esp = float(sp[:, 1].max())

    beats = {}
    for bf in beat_files:
        # Extract beat number from filename
        try:
            beat_idx = int(bf.stem.replace("per_cell_data_beat", ""))
        except ValueError:
            continue
        pc = np.load(bf, allow_pickle=True)
        beats[beat_idx] = pc

    cases.append({"label": label, "rv_esp": rv_esp, "beats": beats, "dir": d})

if not cases:
    print("ERROR: no cases with per-beat data found")
    exit(1)

cases.sort(key=lambda c: c["rv_esp"])
n = len(cases)
print(f"Loaded {n} cases")
for c in cases:
    print(f"  {c['label']:<18} RV_ESP={c['rv_esp']:.1f} beats={sorted(c['beats'].keys())}")

# All cases must have the same beat indices
all_beats = set()
for c in cases:
    all_beats.update(c["beats"].keys())
all_beats = sorted(all_beats)
print(f"\nBeat indices found: {all_beats}")

# Sanity check: each case must have every beat
for c in cases:
    missing = set(all_beats) - set(c["beats"].keys())
    if missing:
        print(f"  WARNING: {c['label']} missing beats {sorted(missing)}")

# ── Compute per-beat correlations at fixed definitions ──────────────────────
# Use canonical fields: is_geometric_septum, is_ldrb_septum are the same cells
# for every case (because canonical tagging was used).
# Sweep over t from -10 to +15 mm on the envelope.

N_SWEEP = 40
t_mm_sweep = np.linspace(-10, 15, N_SWEEP)

# Arrays: shape (n_beats, n_sweep_points)
r_plv_sweep = np.full((len(all_beats), N_SWEEP), np.nan)
r_prv_sweep = np.full((len(all_beats), N_SWEEP), np.nan)
r_trans_sweep = np.full((len(all_beats), N_SWEEP), np.nan)
n_cells_sweep = np.zeros((len(all_beats), N_SWEEP))

# At each direct definition (geometric, LDRB): shape (n_beats,)
r_plv_geo = np.full(len(all_beats), np.nan)
r_prv_geo = np.full(len(all_beats), np.nan)
r_trans_geo = np.full(len(all_beats), np.nan)
r_plv_ldrb = np.full(len(all_beats), np.nan)
r_prv_ldrb = np.full(len(all_beats), np.nan)
r_trans_ldrb = np.full(len(all_beats), np.nan)

for bi, beat in enumerate(all_beats):
    # Pull one case to get the canonical masks (same for all cases)
    ref = cases[0]["beats"][beat]
    env = ref["envelope"]
    entry_t = ref["entry_t"]
    is_geo = ref["is_geometric_septum"]
    is_ldrb = ref["is_ldrb_septum"]

    # --- Direct geometric ---
    W_true = np.array([c["beats"][beat]["w_total"][c["beats"][beat]["is_geometric_septum"]].sum() for c in cases])
    W_plv = np.array([c["beats"][beat]["proxy_PLV_ll"][c["beats"][beat]["is_geometric_septum"]].sum() for c in cases])
    W_prv = np.array([c["beats"][beat]["proxy_PRV_ll"][c["beats"][beat]["is_geometric_septum"]].sum() for c in cases])
    W_trans = np.array([c["beats"][beat]["proxy_Trans_ll"][c["beats"][beat]["is_geometric_septum"]].sum() for c in cases])
    r_plv_geo[bi] = safe_r(W_true, W_plv)
    r_prv_geo[bi] = safe_r(W_true, W_prv)
    r_trans_geo[bi] = safe_r(W_true, W_trans)

    # --- Direct LDRB ---
    W_true = np.array([c["beats"][beat]["w_total"][c["beats"][beat]["is_ldrb_septum"]].sum() for c in cases])
    W_plv = np.array([c["beats"][beat]["proxy_PLV_ll"][c["beats"][beat]["is_ldrb_septum"]].sum() for c in cases])
    W_prv = np.array([c["beats"][beat]["proxy_PRV_ll"][c["beats"][beat]["is_ldrb_septum"]].sum() for c in cases])
    W_trans = np.array([c["beats"][beat]["proxy_Trans_ll"][c["beats"][beat]["is_ldrb_septum"]].sum() for c in cases])
    r_plv_ldrb[bi] = safe_r(W_true, W_plv)
    r_prv_ldrb[bi] = safe_r(W_true, W_prv)
    r_trans_ldrb[bi] = safe_r(W_true, W_trans)

    # --- Sweep over t ---
    for ti, t_mm in enumerate(t_mm_sweep):
        t_m = t_mm / 1000.0
        W_true = np.zeros(n)
        W_plv = np.zeros(n)
        W_prv = np.zeros(n)
        W_trans = np.zeros(n)
        total_cells = 0
        for j, c in enumerate(cases):
            pcb = c["beats"][beat]
            mask = pcb["envelope"] & (pcb["entry_t"] < t_m)
            total_cells += mask.sum()
            if mask.sum() == 0:
                continue
            W_true[j] = pcb["w_total"][mask].sum()
            W_plv[j] = pcb["proxy_PLV_ll"][mask].sum()
            W_prv[j] = pcb["proxy_PRV_ll"][mask].sum()
            W_trans[j] = pcb["proxy_Trans_ll"][mask].sum()
        n_cells_sweep[bi, ti] = total_cells / n
        r_plv_sweep[bi, ti] = safe_r(W_true, W_plv)
        r_prv_sweep[bi, ti] = safe_r(W_true, W_prv)
        r_trans_sweep[bi, ti] = safe_r(W_true, W_trans)

# ── Print table ──────────────────────────────────────────────────────────────
print()
print("=" * 85)
print(f"PER-BEAT CONVERGENCE — direct definitions (canonical atlas tagging)")
print("=" * 85)
print(f"\n{'Beat':>4} {'Geo r_PLV':>12} {'Geo r_PRV':>12} {'Geo r_Trans':>12} |"
      f" {'LDRB r_PLV':>12} {'LDRB r_PRV':>12} {'LDRB r_Trans':>13}")
print("-" * 85)
for bi, beat in enumerate(all_beats):
    print(f"{beat:>4} {r_plv_geo[bi]:>+12.4f} {r_prv_geo[bi]:>+12.4f} {r_trans_geo[bi]:>+12.4f} |"
          f" {r_plv_ldrb[bi]:>+12.4f} {r_prv_ldrb[bi]:>+12.4f} {r_trans_ldrb[bi]:>+13.4f}")

# ── Plots ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# Panel 1: r vs beat at direct geometric definition
ax = fig.add_subplot(gs[0, 0])
ax.plot(all_beats, r_plv_geo, "-o", color="C0", lw=2, ms=8, label="$P_{LV}$")
ax.plot(all_beats, r_prv_geo, "-s", color="C3", lw=2, ms=8, label="$P_{RV}$")
ax.plot(all_beats, r_trans_geo, "-^", color="C2", lw=2, ms=8, label="$P_{LV}-P_{RV}$")
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_xlabel("Beat number")
ax.set_ylabel(f"Pearson r (across {n} cases)")
ax.set_title("Geometric septum (direct)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
ax.set_ylim(-1.1, 1.1)

# Panel 2: r vs beat at direct LDRB definition
ax = fig.add_subplot(gs[0, 1])
ax.plot(all_beats, r_plv_ldrb, "-o", color="C0", lw=2, ms=8, label="$P_{LV}$")
ax.plot(all_beats, r_prv_ldrb, "-s", color="C3", lw=2, ms=8, label="$P_{RV}$")
ax.plot(all_beats, r_trans_ldrb, "-^", color="C2", lw=2, ms=8, label="$P_{LV}-P_{RV}$")
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_xlabel("Beat number")
ax.set_ylabel("Pearson r")
ax.set_title("LDRB septum (direct)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
ax.set_ylim(-1.1, 1.1)

# Panel 3: Diff between beat 0 and last beat
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
last_b = all_beats[-1]
first_b = all_beats[0]
summary = (
    f"Convergence summary\n"
    f"{'─'*28}\n\n"
    f"{'':<8} {'beat 0':>10} {'beat '+str(last_b):>10} {'Δ':>10}\n"
    f"{'─'*40}\n"
    f"Geometric:\n"
    f"{'  P_LV':<8} {r_plv_geo[0]:>+10.3f} {r_plv_geo[-1]:>+10.3f} {r_plv_geo[-1]-r_plv_geo[0]:>+10.3f}\n"
    f"{'  P_RV':<8} {r_prv_geo[0]:>+10.3f} {r_prv_geo[-1]:>+10.3f} {r_prv_geo[-1]-r_prv_geo[0]:>+10.3f}\n"
    f"{'  Trans':<8} {r_trans_geo[0]:>+10.3f} {r_trans_geo[-1]:>+10.3f} {r_trans_geo[-1]-r_trans_geo[0]:>+10.3f}\n\n"
    f"LDRB:\n"
    f"{'  P_LV':<8} {r_plv_ldrb[0]:>+10.3f} {r_plv_ldrb[-1]:>+10.3f} {r_plv_ldrb[-1]-r_plv_ldrb[0]:>+10.3f}\n"
    f"{'  P_RV':<8} {r_prv_ldrb[0]:>+10.3f} {r_prv_ldrb[-1]:>+10.3f} {r_prv_ldrb[-1]-r_prv_ldrb[0]:>+10.3f}\n"
    f"{'  Trans':<8} {r_trans_ldrb[0]:>+10.3f} {r_trans_ldrb[-1]:>+10.3f} {r_trans_ldrb[-1]-r_trans_ldrb[0]:>+10.3f}\n"
)
ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
        va="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# Panel 4: Sweep at beat 0 vs last beat (r_PLV)
ax = fig.add_subplot(gs[1, 0])
cmap = plt.cm.coolwarm
for bi, beat in enumerate(all_beats):
    color = cmap(bi / max(1, len(all_beats) - 1))
    ax.plot(t_mm_sweep, r_plv_sweep[bi], "-", color=color, lw=1.5, label=f"beat {beat}")
ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5, label="t=0 (geometric)")
ax.set_xlabel("t (mm)")
ax.set_ylabel("r_PLV")
ax.set_title("$P_{LV}$ sweep per beat")
ax.legend(fontsize=7, loc="lower right", ncol=2)
ax.grid(alpha=0.3)
ax.set_ylim(-1.1, 1.1)

# Panel 5: Sweep at beat 0 vs last beat (r_Trans)
ax = fig.add_subplot(gs[1, 1])
for bi, beat in enumerate(all_beats):
    color = cmap(bi / max(1, len(all_beats) - 1))
    ax.plot(t_mm_sweep, r_trans_sweep[bi], "-", color=color, lw=1.5, label=f"beat {beat}")
ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
ax.set_xlabel("t (mm)")
ax.set_ylabel("r_Trans")
ax.set_title("$P_{LV}-P_{RV}$ sweep per beat")
ax.legend(fontsize=7, loc="lower right", ncol=2)
ax.grid(alpha=0.3)
ax.set_ylim(-1.1, 1.1)

# Panel 6: Sweep at last beat — all proxies
ax = fig.add_subplot(gs[1, 2])
bi_last = len(all_beats) - 1
ax.plot(t_mm_sweep, r_plv_sweep[bi_last], "-o", color="C0", lw=2, ms=4, label="$P_{LV}$")
ax.plot(t_mm_sweep, r_prv_sweep[bi_last], "-s", color="C3", lw=2, ms=4, label="$P_{RV}$")
ax.plot(t_mm_sweep, r_trans_sweep[bi_last], "-^", color="C2", lw=2, ms=4, label="$P_{LV}-P_{RV}$")
ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_xlabel("t (mm)")
ax.set_ylabel("Pearson r")
ax.set_title(f"Sweep at last beat ({all_beats[-1]})")
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3)
ax.set_ylim(-1.1, 1.1)

fig.suptitle(f"Per-beat proxy correlation convergence — {n} cases, canonical atlas tagging",
             fontsize=13, fontweight="bold")
fig.savefig(out_dir / "per_beat_convergence.png", dpi=150, bbox_inches="tight")
print(f"\nSaved {out_dir / 'per_beat_convergence.png'}")

# Save raw data
np.savez(out_dir / "per_beat_raw.npz",
         beats=np.array(all_beats),
         t_mm_sweep=t_mm_sweep,
         r_plv_sweep=r_plv_sweep,
         r_prv_sweep=r_prv_sweep,
         r_trans_sweep=r_trans_sweep,
         n_cells_sweep=n_cells_sweep,
         r_plv_geo=r_plv_geo,
         r_prv_geo=r_prv_geo,
         r_trans_geo=r_trans_geo,
         r_plv_ldrb=r_plv_ldrb,
         r_prv_ldrb=r_prv_ldrb,
         r_trans_ldrb=r_trans_ldrb,
         case_labels=np.array([c["label"] for c in cases]),
         case_rv_esp=np.array([c["rv_esp"] for c in cases]),
)
print(f"Saved {out_dir / 'per_beat_raw.npz'}")
print("Done.")
