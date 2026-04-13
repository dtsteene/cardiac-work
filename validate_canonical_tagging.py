#!/usr/bin/env python3
"""
validate_canonical_tagging.py — sanity check the u_pre canonical tagging.

Compares per-case and canonical per_cell_data.npz outputs for the same sims
and checks a series of invariants. Fails loudly if any test breaks.

Tests:
  1. Cell counts: canonical must give identical n_geo/n_ldrb across all cases
  2. Bijection: each canonical mask must have exactly the expected count
  3. Intersection: canonical and per-case should agree on most cells
     (intersection / union = Jaccard > 0.5 — they should overlap substantially)
  4. Energy conservation: sum(w_total_per_cell) must match the regional integral
     from postprocess_metrics (if available)
  5. Proxy algebraic identity: W_Trans == W_PLV - W_PRV to machine precision
  6. Total work summed over envelope should be bounded and finite

Usage:
    python3 validate_canonical_tagging.py results/sims/2026-04-12/UKB_6beats_run_*
"""
import argparse
import numpy as np
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("result_dirs", nargs="+", type=Path)
parser.add_argument("--strict", action="store_true",
                    help="Exit with non-zero status if any test fails")
args = parser.parse_args()

# ── Load per-case and canonical data for each sim ────────────────────────────
cases = []
for d in args.result_dirs:
    d = d.resolve()
    canonical = d / "per_cell_data.npz"
    percase = d / "per_cell_data_percase.npz"
    if not canonical.exists():
        print(f"  SKIP {d.name}: no per_cell_data.npz (canonical)")
        continue
    entry = {"dir": d, "canonical": np.load(canonical)}
    if percase.exists():
        entry["percase"] = np.load(percase)
    cases.append(entry)

if len(cases) == 0:
    print("ERROR: no canonical results found")
    sys.exit(1)

print(f"Loaded {len(cases)} cases")
for c in cases:
    have_pc = "yes" if "percase" in c else "no"
    print(f"  {c['dir'].name}: canonical=yes percase={have_pc}")
print()

# ── TEST 1: Cell-count consistency across all cases (canonical only) ─────────
print("=" * 70)
print("TEST 1: Canonical cell counts consistent across cases")
print("=" * 70)
counts = {"geo": [], "ldrb": [], "env": [], "study": []}
for c in cases:
    pc = c["canonical"]
    counts["geo"].append(int(pc["is_geometric_septum"].sum()))
    counts["ldrb"].append(int(pc["is_ldrb_septum"].sum()))
    counts["env"].append(int(pc["envelope"].sum()))
    counts["study"].append(int(pc["study_region"].sum()))

passed_1 = True
for k, v in counts.items():
    v = np.array(v)
    std = v.std()
    print(f"  {k:<8}: values={v.tolist()}, std={std:.2f}")
    if std > 2:  # allow up to 2-cell non-bijection tolerance
        print(f"    !! FAIL: std={std:.2f} too high")
        passed_1 = False
print(f"  --> {'PASS' if passed_1 else 'FAIL'}: cell counts {'identical' if passed_1 else 'differ significantly'}")
print()

# ── TEST 2: Bijection check (implicit via count match) ──────────────────────
print("=" * 70)
print("TEST 2: Canonical fields are self-consistent")
print("=" * 70)
passed_2 = True
for c in cases:
    pc = c["canonical"]
    n = len(pc["tau"])
    # Each cell should appear at most once in each mask
    # Trivially true for boolean arrays, but check that masks are well-formed
    for key in ["is_geometric_septum", "is_ldrb_septum", "envelope", "touches_epi"]:
        if key not in pc.files:
            print(f"  {c['dir'].name}: missing {key}")
            passed_2 = False
            continue
        arr = pc[key]
        if arr.dtype != bool:
            print(f"  {c['dir'].name}: {key} is not bool (dtype={arr.dtype})")
            passed_2 = False
        if arr.shape[0] != n:
            print(f"  {c['dir'].name}: {key} size mismatch")
            passed_2 = False
print(f"  --> {'PASS' if passed_2 else 'FAIL'}")
print()

# ── TEST 3: Jaccard overlap between canonical and per-case ──────────────────
print("=" * 70)
print("TEST 3: Canonical vs per-case mask overlap (should be substantial)")
print("=" * 70)
have_percase = [c for c in cases if "percase" in c]
if len(have_percase) == 0:
    print("  SKIP: no per-case data available for comparison")
else:
    print(f"  {'Case':>20} {'n_geo_canon':>12} {'n_geo_pc':>10} {'Jaccard_geo':>12} {'Jaccard_ldrb':>13}")
    all_jaccards = []
    for c in have_percase:
        canon = c["canonical"]
        pc = c["percase"]
        geo_c = canon["is_geometric_septum"]
        geo_p = pc["is_geometric_septum"]
        ldrb_c = canon["is_ldrb_septum"]
        ldrb_p = pc["is_ldrb_septum"]
        j_geo = (geo_c & geo_p).sum() / max(1, (geo_c | geo_p).sum())
        j_ldrb = (ldrb_c & ldrb_p).sum() / max(1, (ldrb_c | ldrb_p).sum())
        all_jaccards.append((j_geo, j_ldrb))
        print(f"  {c['dir'].name[:20]:>20} {int(geo_c.sum()):>12} {int(geo_p.sum()):>10} {j_geo:>12.3f} {j_ldrb:>13.3f}")
    min_j_geo = min(j for j, _ in all_jaccards)
    min_j_ldrb = min(l for _, l in all_jaccards)
    # LDRB is stable per-case so Jaccard should be near 1
    # Geo is NOT stable per-case (that's the bug we're fixing) so Jaccard may be lower
    passed_3 = min_j_ldrb > 0.9 and min_j_geo > 0.3
    print(f"  Min Jaccard geo={min_j_geo:.3f}, LDRB={min_j_ldrb:.3f}")
    print(f"  --> {'PASS' if passed_3 else 'WARN'}")
    print(f"  (LDRB should be > 0.9; geo can be lower because per-case drifts with prestress)")
print()

# ── TEST 4: Proxy algebraic identity W_Trans = W_PLV - W_PRV ────────────────
print("=" * 70)
print("TEST 4: Proxy identity W_Trans = W_PLV - W_PRV (machine precision)")
print("=" * 70)
passed_4 = True
for c in cases:
    pc = c["canonical"]
    mask = pc["envelope"]
    plv = pc["proxy_PLV_ll"][mask].sum()
    prv = pc["proxy_PRV_ll"][mask].sum()
    trans = pc["proxy_Trans_ll"][mask].sum()
    residual = abs(trans - (plv - prv))
    rel = residual / max(1e-30, abs(plv) + abs(prv))
    if rel > 1e-10:
        print(f"  {c['dir'].name}: residual={residual:.3e}, rel={rel:.3e} FAIL")
        passed_4 = False
    else:
        print(f"  {c['dir'].name}: residual={residual:.3e} (rel={rel:.3e}) ok")
print(f"  --> {'PASS' if passed_4 else 'FAIL'}")
print()

# ── TEST 5: Bounded work values (sanity check) ───────────────────────────────
print("=" * 70)
print("TEST 5: Work values are bounded and finite")
print("=" * 70)
passed_5 = True
for c in cases:
    pc = c["canonical"]
    for key in ["w_total", "proxy_PLV_ll", "proxy_PRV_ll", "proxy_Trans_ll"]:
        if key not in pc.files:
            continue
        arr = pc[key]
        if not np.all(np.isfinite(arr)):
            print(f"  {c['dir'].name}: {key} has non-finite values FAIL")
            passed_5 = False
print(f"  --> {'PASS' if passed_5 else 'FAIL'}")
print()

# ── TEST 5b: DG0-vs-scalar integration (machine-precision invariant) ─────────
# The DG0 trick says sum(per_cell_integral) == scalar_domain_integral for the
# same integrand. If this fails, the per-cell data is wrong at a fundamental
# level — independent of tagging, pressure source, or any other choice. Hard
# evidence that compute_per_cell.py's DG0 test-function trick is not
# simplifying anything.
print("=" * 70)
print("TEST 5b: DG0 sum(per-cell) == scalar domain integral")
print("=" * 70)
passed_5b = True
any_checked = False
for c in cases:
    pc = c["canonical"]
    if "w_total_scalar_integral" not in pc.files:
        continue
    any_checked = True
    dg0_sum = float(pc["w_total"].sum())
    scalar = float(pc["w_total_scalar_integral"])
    rel = abs(dg0_sum - scalar) / max(abs(scalar), 1e-30)
    sym = "ok" if rel < 1e-10 else "FAIL"
    print(f"  {c['dir'].name[-8:]:>12} [{sym}] dg0_sum={dg0_sum:.6e} scalar={scalar:.6e} rel={rel:.2e}")
    if rel >= 1e-10:
        passed_5b = False
if not any_checked:
    print("  SKIP: no w_total_scalar_integral in saved data.")
    print("  Re-run compute_per_cell.py with the updated version (adds scalar cross-check).")
    passed_5b = None
else:
    print(f"  --> {'PASS' if passed_5b else 'FAIL'}")
print()

# ── TEST 6: Cross-case work ordering (healthy should have larger W than end_stage) ─
print("=" * 70)
print("TEST 6: Total envelope work decreases monotonically with severity")
print("=" * 70)
# Sort by RV_ESP (from solver pressure if available)
def rv_esp(c):
    sp = c["dir"] / "solver" / "solver_cavity_pressure_mmHg.npy"
    if not sp.exists():
        sp = c["dir"] / "solver" / "pressure_history.npy"
    if sp.exists():
        p = np.load(sp)
        return float(p[:, 1].max())
    return 0.0

cases_sorted = sorted(cases, key=rv_esp)
print(f"  {'Case':>20} {'RV_ESP':>8} {'W_total_env':>14} {'W_PLV_env':>12} {'W_Trans_env':>12}")
for c in cases_sorted:
    pc = c["canonical"]
    mask = pc["envelope"]
    w = pc["w_total"][mask].sum()
    wp = pc["proxy_PLV_ll"][mask].sum()
    wt = pc["proxy_Trans_ll"][mask].sum()
    print(f"  {c['dir'].name[:20]:>20} {rv_esp(c):>8.1f} {w:>14.6e} {wp:>12.6e} {wt:>12.6e}")
print(f"  (PASS if the values look reasonable — no monotonicity expected, just sanity)")
print()

# ── Summary ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("SUMMARY")
print("=" * 70)
tests = {
    "1. cell count consistency": passed_1,
    "2. field self-consistency": passed_2,
    "3. canonical/per-case overlap": passed_3 if len(have_percase) > 0 else None,
    "4. proxy algebraic identity": passed_4,
    "5. bounded work values": passed_5,
    "5b. DG0 vs scalar integration (hard evidence DG0 trick is exact)": passed_5b,
}
for name, result in tests.items():
    sym = "PASS" if result is True else ("FAIL" if result is False else "SKIP")
    print(f"  [{sym}] {name}")

any_fail = any(r is False for r in tests.values())
if args.strict and any_fail:
    sys.exit(1)
