#!/usr/bin/env python3
"""
viz_best_proxy_per_cell.py — For every wall cell, ask which proxy best
tracks its true work across the severity spectrum. Write the result to an
XDMF file with fields the user can threshold in ParaView to see a
data-driven septum emerge.

For each cell c and each severity s ∈ {7 spectrum sims}, read:
    w_true_c(s)   = w_total[c]   in that sim
    w_PLV_c(s)    = proxy_PLV_ll[c]
    w_PRV_c(s)    = proxy_PRV_ll[c]
    w_Trans_c(s)  = proxy_Trans_ll[c]

Then compute Pearson r across the 7 severities, per cell, per proxy.
Also compute the closed-form optimal linear blend of PLV and PRV (the same
angle search we already use in analyze_spectrum.py, but per cell).

Output fields (all DG0 scalars on the canonical shared UKB mesh):

    r_PLV           per-cell r(W_true, W_PLV) across 7 severities
    r_PRV           per-cell r(W_true, W_PRV) across 7 severities
    r_Trans         per-cell r(W_true, W_Trans) across 7 severities
    r_Opt           per-cell r(W_true, optimal linear blend)
    theta_opt_deg   per-cell optimal blend angle in degrees
    best_proxy      categorical: 1=PLV, 2=PRV, 3=Trans, 0=degenerate
    best_r          the r value of the winning proxy
    W_true_severe   absolute magnitude of W_true in the severe case (sanity)

Load in ParaView → open fig_best_proxy_per_cell.xdmf → threshold on r_Trans
to see where Trans dominates. Or colour by theta_opt_deg and see the
gradient from PLV (θ=0°) through Trans (θ=-45°) to PRV (θ=+90°) across
the wall. The septum should emerge as the band where Trans wins.
"""
import numpy as np
from pathlib import Path
from mpi4py import MPI
import dolfinx
import cardiac_geometries

# ── Spectrum specification (ordered) ─────────────────────────────────────────
SPECTRUM = [
    ("healthy",         "1020849"),
    ("mild",            "1020851"),
    ("moderate",        "1020852"),
    ("moderate_severe", "1020853"),
    ("severe",          "1020854"),
    ("very_severe",     "1020855"),
    ("end_stage",       "1020856"),
]
ROOT = Path("/home/dtsteene/D1/cardiac-work/results/sims/2026-04-12")
OUT = Path("/home/dtsteene/D1/cardiac-work/results/analysis/cascade")
OUT.mkdir(parents=True, exist_ok=True)

SHARED = "/home/dtsteene/D1/cardiac-work/data/shared_ukb_mesh/ukb/geometry"
geo = cardiac_geometries.geometry.Geometry.from_folder(MPI.COMM_SELF, SHARED)
mesh = geo.mesh
n_cg = mesh.topology.index_map(3).size_local
print(f"Canonical shared mesh: {n_cg} cells")

# ── Load all 7 per_cell files and build (n_cells, 7) arrays on canonical mesh ─
def load_cg_field(pc, field_name):
    """Read a per-cell array from per_cell_data.npz (ckpt order) and
    scatter-assign onto canonical cg order using ckpt_to_cg_idx."""
    perm = pc["ckpt_to_cg_idx"]
    vals = np.asarray(pc[field_name], dtype=np.float64)
    cg = np.zeros(n_cg, dtype=np.float64)
    cg[perm] = vals
    return cg

W_true = np.zeros((n_cg, len(SPECTRUM)), dtype=np.float64)
W_PLV = np.zeros_like(W_true)
W_PRV = np.zeros_like(W_true)
W_Trans = np.zeros_like(W_true)

for i, (sev, run_id) in enumerate(SPECTRUM):
    pc_path = ROOT / f"UKB_6beats_run_{run_id}" / "per_cell_data.npz"
    pc = np.load(pc_path, allow_pickle=True)
    W_true[:, i]  = load_cg_field(pc, "w_total")
    W_PLV[:, i]   = load_cg_field(pc, "proxy_PLV_ll")
    W_PRV[:, i]   = load_cg_field(pc, "proxy_PRV_ll")
    W_Trans[:, i] = load_cg_field(pc, "proxy_Trans_ll")
    print(f"  Loaded {sev} — W_true sum {W_true[:, i].sum()*1000:+.2f} mJ")

# ── Per-cell Pearson r across the 7 severities ───────────────────────────────
def per_cell_r(A, B):
    """Row-wise Pearson correlation; returns array of length n_cg."""
    Am = A - A.mean(axis=1, keepdims=True)
    Bm = B - B.mean(axis=1, keepdims=True)
    num = (Am * Bm).sum(axis=1)
    den = np.sqrt((Am ** 2).sum(axis=1) * (Bm ** 2).sum(axis=1))
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-18)
    return r


r_PLV   = per_cell_r(W_true, W_PLV)
r_PRV   = per_cell_r(W_true, W_PRV)
r_Trans = per_cell_r(W_true, W_Trans)

# ── Per-cell optimal blend angle ─────────────────────────────────────────────
# For each cell, sweep θ ∈ [-π, π] at 1° resolution and pick argmax r.
# 361 angles × 2153 cells = 778k small dot products — fast.
thetas = np.linspace(-np.pi, np.pi, 361)
print(f"Computing per-cell optimal blend over {len(thetas)} angles × {n_cg} cells...")

Wt_mean = W_true.mean(axis=1, keepdims=True)
Wt_c = W_true - Wt_mean                                # (n_cells, 7)
Wt_norm = np.sqrt((Wt_c ** 2).sum(axis=1) + 1e-18)    # (n_cells,)

PLV_c = W_PLV - W_PLV.mean(axis=1, keepdims=True)
PRV_c = W_PRV - W_PRV.mean(axis=1, keepdims=True)

best_r_opt = np.full(n_cg, -np.inf, dtype=np.float64)
best_theta = np.zeros(n_cg, dtype=np.float64)
for th in thetas:
    comb = np.cos(th) * PLV_c + np.sin(th) * PRV_c      # (n_cells, 7)
    num = (comb * Wt_c).sum(axis=1)
    comb_norm = np.sqrt((comb ** 2).sum(axis=1) + 1e-18)
    r = num / (comb_norm * Wt_norm)
    r = np.nan_to_num(r, nan=-np.inf)
    better = r > best_r_opt
    best_r_opt[better] = r[better]
    best_theta[better] = th
best_r_opt[~np.isfinite(best_r_opt)] = 0.0

# ── Best canonical proxy per cell ────────────────────────────────────────────
# 1 = PLV, 2 = PRV, 3 = Trans, 0 = degenerate (all three low/NaN).
stack = np.stack([r_PLV, r_PRV, r_Trans], axis=1)       # (n, 3)
argmax = np.argmax(stack, axis=1) + 1                    # 1, 2, 3
best_r = stack.max(axis=1)
# Mask degenerate cells (wildly low correlation OR NaN)
degenerate = (~np.isfinite(best_r)) | (best_r < 0.0)
argmax[degenerate] = 0
best_r[degenerate] = 0.0
print(f"Winner counts: PLV={int((argmax==1).sum())}  "
      f"PRV={int((argmax==2).sum())}  Trans={int((argmax==3).sum())}  "
      f"none={int((argmax==0).sum())}")

# ── Data-driven septum definitions ──────────────────────────────────────────
# Three strengths:
#   loose     — Trans is argmax (noisy, 337 cells)
#   confident — argmax AND margin > 0.10 vs the runner-up proxy (strips
#               out coin-flip decisions; with n=7 per cell the raw Pearson
#               r is noisy so this is the statistically honest version)
#   strict    — argmax AND r_Trans > 0.9 (smallest, highest-confidence)
stack_r   = np.stack([r_PLV, r_PRV, r_Trans], axis=1)
sorted_r  = np.sort(stack_r, axis=1)[:, ::-1]   # descending per cell
margin    = sorted_r[:, 0] - sorted_r[:, 1]      # best minus runner-up

data_septum_loose     = (argmax == 3).astype(np.float64)
data_septum_confident = ((argmax == 3) & (margin > 0.10)).astype(np.float64)
data_septum_strict    = ((argmax == 3) & (r_Trans > 0.9)).astype(np.float64)

# ── Load LDRB + geometric septum masks (from any canonical-tagged sim) ──────
# Note: per_cell_data.npz carries THREE different septum-ish masks that can
# disagree — we compare the data-driven definition against all three.
#
#   region_tags == 3     → "LDRB-native": the hard septum tag emitted by
#                          the LDRB algorithm's markers_mt output (smallest,
#                          most conservative). This is the one
#                          metrics_calculator uses for "Septum" region.
#   is_ldrb_septum       → "LDRB-loose": a wider Laplace-based definition
#                          used for the sweep envelope in compute_per_cell.
#   is_geometric_septum  → geometric distance definition:
#                          max(d_LV,d_RV) < d_epi
_ref_pc = np.load(ROOT / f"UKB_6beats_run_{SPECTRUM[0][1]}" / "per_cell_data.npz",
                   allow_pickle=True)
ldrb_native_bool = load_cg_field(_ref_pc, "region_tags") > 2.5           # ==3
ldrb_loose_bool  = load_cg_field(_ref_pc, "is_ldrb_septum") > 0.5
geo_septum_bool  = load_cg_field(_ref_pc, "is_geometric_septum") > 0.5

ldrb_native = ldrb_native_bool.astype(np.float64)
ldrb_loose  = ldrb_loose_bool.astype(np.float64)
geo_septum  = geo_septum_bool.astype(np.float64)

# ── Set-overlap metrics ─────────────────────────────────────────────────────
def overlap_metrics(A_bool, B_bool, name_a, name_b):
    A = A_bool.astype(bool); B = B_bool.astype(bool)
    nA, nB = int(A.sum()), int(B.sum())
    inter = int((A & B).sum())
    union = int((A | B).sum())
    jaccard = inter / union if union else float("nan")
    dice    = 2 * inter / (nA + nB) if (nA + nB) else float("nan")
    # Recall(A|B) = fraction of B's cells that are also in A
    recall_b = inter / nB if nB else float("nan")
    recall_a = inter / nA if nA else float("nan")
    return {
        "n_A": nA, "n_B": nB, "inter": inter, "union": union,
        "jaccard": jaccard, "dice": dice,
        "recall_A_in_B": recall_a, "recall_B_in_A": recall_b,
        "name_a": name_a, "name_b": name_b,
    }


data_loose_bool  = (argmax == 3)
data_strict_bool = (argmax == 3) & (r_Trans > 0.9)

data_conf_bool = (argmax == 3) & (margin > 0.10)

comparisons = [
    ("data_septum_loose (Trans=argmax)",      data_loose_bool,  ldrb_native_bool, "LDRB-native"),
    ("data_septum_loose (Trans=argmax)",      data_loose_bool,  geo_septum_bool,  "geometric"),
    ("data_septum_confident (margin>0.10)",   data_conf_bool,   ldrb_native_bool, "LDRB-native"),
    ("data_septum_confident (margin>0.10)",   data_conf_bool,   geo_septum_bool,  "geometric"),
    ("data_septum_strict (r>0.9)",             data_strict_bool, ldrb_native_bool, "LDRB-native"),
    ("data_septum_strict (r>0.9)",             data_strict_bool, geo_septum_bool,  "geometric"),
    ("LDRB-native",                             ldrb_native_bool, geo_septum_bool,  "geometric"),
]

print("\nSet-overlap metrics:")
print(f"{'A':<38} {'B':<12} {'|A|':>5} {'|B|':>5} {'|A∩B|':>7} "
      f"{'Jaccard':>8} {'Dice':>6} {'A⊂B %':>6} {'B⊂A %':>6}")
print("-" * 100)
for name_a, A_bool, B_bool, name_b in comparisons:
    m = overlap_metrics(A_bool, B_bool, name_a, name_b)
    print(f"{name_a:<38} {name_b:<12} {m['n_A']:>5d} {m['n_B']:>5d} "
          f"{m['inter']:>7d} {m['jaccard']:>8.3f} {m['dice']:>6.3f} "
          f"{m['recall_A_in_B']*100:>6.1f} {m['recall_B_in_A']*100:>6.1f}")

# ── Write to XDMF ────────────────────────────────────────────────────────────
V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))


def write_field(xf, name, arr):
    f = dolfinx.fem.Function(V0)
    f.name = name
    f.x.array[:n_cg] = arr.astype(np.float64)
    xf.write_function(f, 0.0)


out_xdmf = OUT / "fig_best_proxy_per_cell.xdmf"
with dolfinx.io.XDMFFile(MPI.COMM_SELF, out_xdmf, "w") as xf:
    xf.write_mesh(mesh)
    write_field(xf, "r_PLV",               r_PLV)
    write_field(xf, "r_PRV",               r_PRV)
    write_field(xf, "r_Trans",             r_Trans)
    write_field(xf, "r_Opt",               best_r_opt)
    write_field(xf, "theta_opt_deg",       np.degrees(best_theta))
    write_field(xf, "best_proxy",          argmax.astype(np.float64))
    write_field(xf, "best_r",              best_r)
    write_field(xf, "data_septum_loose",     data_septum_loose)
    write_field(xf, "data_septum_confident", data_septum_confident)
    write_field(xf, "data_septum_strict",    data_septum_strict)
    write_field(xf, "margin",                margin)
    write_field(xf, "ldrb_native_septum",    ldrb_native)
    write_field(xf, "ldrb_loose_septum",     ldrb_loose)
    write_field(xf, "geometric_septum",      geo_septum)
    write_field(xf, "W_true_severe",       W_true[:, 4])   # severe column — sanity

print(f"\nSaved {out_xdmf}")
print(f"  (plus {out_xdmf.with_suffix('.h5')})")

# ── Console summary: candidate data-driven septum thresholds ─────────────────
print("\nThreshold sweep — how big is the 'data-driven septum' "
      "(cells where Trans wins with r_Trans > threshold)?")
print(f"{'threshold':>11} {'Trans-wins cells':>18}  "
      f"{'% of wall':>10}")
for th in [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]:
    mask = (argmax == 3) & (r_Trans > th)
    n = int(mask.sum())
    print(f"  r_Trans>{th:<4}  {n:>12d}     "
          f"{100*n/n_cg:>7.1f}%")

# Save raw arrays too
np.savez(OUT / "best_proxy_raw.npz",
         r_PLV=r_PLV, r_PRV=r_PRV, r_Trans=r_Trans,
         r_Opt=best_r_opt, theta_opt=best_theta,
         best_proxy=argmax, best_r=best_r,
         W_true=W_true, W_PLV=W_PLV, W_PRV=W_PRV, W_Trans=W_Trans,
         severities=np.array([s for s, _ in SPECTRUM]))
print(f"\nSaved {OUT / 'best_proxy_raw.npz'}")
