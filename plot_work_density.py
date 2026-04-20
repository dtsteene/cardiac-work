#!/usr/bin/env python3
"""
plot_work_density.py — INTENSIVE work density comparisons.

CONCEPTUAL FIX (2026-04-15): the previous analysis compared TOTAL work
(extensive, scales with region volume) against TOTAL proxy (extensive,
sum of per-cell cavity-pressure × strain). Under wall-thickness
variation, the region's volume changes substantially and the total
proxy is dominated by volume scaling rather than by the mechanical
change we care about. The CLINICAL myocardial work index is an
INTENSIVE quantity — pressure × region-averaged strain, integrated
over one cardiac cycle — so it should be compared against the TRUE
work DENSITY, not the total work.

Units: W_density and Proxy_density are both in Pa (= J/m³). This is
the natural unit of the pressure-strain loop area.

The conversion is algebraically exact: since the per-cell proxy field
stored in per_cell_data.npz already has the form
    proxy_PLV_ll[c] = ∫_T P_LV(t) * dε_ll(c,t)/dt * V_c * dt
the region sum divided by V_region equals
    ∫_T P_LV(t) * d<ε_ll>_V(t)/dt * dt
where <ε_ll>_V is the volume-weighted regional mean strain — i.e. the
intensive clinical proxy. Same trick for W_true, PRV, and Trans.

Usage:
    # Spectrum (7 severities, 2026-04-12 canonical)
    python3 plot_work_density.py --mode spectrum \\
        results/sims/2026-04-12/UKB_6beats_run_1020849 \\
        results/sims/2026-04-12/UKB_6beats_run_1020851 \\
        results/sims/2026-04-12/UKB_6beats_run_1020852 \\
        results/sims/2026-04-12/UKB_6beats_run_1020853 \\
        results/sims/2026-04-12/UKB_6beats_run_1020854 \\
        results/sims/2026-04-12/UKB_6beats_run_1020855 \\
        results/sims/2026-04-12/UKB_6beats_run_1020856

    # Thickness (15 sims across 2026-04-14 + 2026-04-15)
    python3 plot_work_density.py --mode thickness \\
        results/sims/2026-04-14/UKB_1beats_run_1029754 ...
"""
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import pearsonr


def safe_r(x, y):
    if np.std(x) == 0 or np.std(y) == 0 or len(x) < 3:
        return float("nan")
    return pearsonr(x, y)[0]


def r_to_alpha(r):
    ar = abs(r) if not np.isnan(r) else 0
    return 0.25 + 0.75 * ar


# ── Region definitions ──────────────────────────────────────────────────────
def region_masks(pc, septum_key):
    return {
        "LV":     pc["region_tags"] == 1,      # LV free wall (LDRB)
        "RV":     pc["region_tags"] == 2,      # RV free wall (LDRB)
        "Septum": pc[septum_key].astype(bool), # Geometric or LDRB septum
    }


def region_densities(pc, septum_key):
    """Compute intensive (density) W_true and three proxy values for a
    single case, per LDRB-region. Units: Pa = J/m³.

    Conversion: per-cell per_cell_data.npz stores integrated values
    already multiplied by cell volumes, so a sum over a region divided
    by the region's total volume gives the volume-weighted average
    (= intensive) quantity directly.
    """
    cv = pc["cell_volumes"]
    out = {}
    for rname, mask in region_masks(pc, septum_key).items():
        V = float(cv[mask].sum())
        if V <= 0 or not mask.any():
            out[rname] = None
            continue
        out[rname] = {
            "V_m3":   V,
            "W_true": float(pc["w_total"][mask].sum())       / V,
            "PLV":    float(pc["proxy_PLV_ll"][mask].sum())  / V,
            "PRV":    float(pc["proxy_PRV_ll"][mask].sum())  / V,
            "Trans":  float(pc["proxy_Trans_ll"][mask].sum()) / V,
        }
    return out


# ── Plot (mirrors the benched plot_work_breakdown.py layout) ──────────────
def make_plot(ordered_cases, labels, septum_key, title_septum, sweep_label,
              out_path, n_sep_cells):
    """Work-breakdown style 2×2 plot using DENSITY (Pa) instead of total
    work (J). ordered_cases is a list of dicts {label, densities:{LV,RV,Septum:{W_true,PLV,PRV,Trans}}}."""
    n = len(ordered_cases)
    x = np.arange(n)
    ms, lw = 8, 2.2

    PA_TO_KPA = 1.0e-3

    # Collect time series: one array per (region, key) across cases
    work = {}
    table_r = {}
    for rname in ("LV", "RV", "Septum"):
        rt = {}
        W_true = np.array([c["densities"][rname]["W_true"] for c in ordered_cases]) * PA_TO_KPA
        work[(rname, "true")] = W_true
        for pname in ("PLV", "PRV", "Trans"):
            W_proxy = np.array([c["densities"][rname][pname] for c in ordered_cases]) * PA_TO_KPA
            work[(rname, pname)] = W_proxy
            rt[pname] = safe_r(W_true, W_proxy)
        table_r[rname] = rt

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1],
                            hspace=0.35, wspace=0.3)

    # ── LV panel (one proxy: P_LV) ────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, work[("LV", "true")], "-D", color="k", lw=lw + 0.5, ms=ms + 1, zorder=5)
    ax.set_ylabel(r"$W^{density}_{true}$ (kPa)", fontsize=10)
    ax.grid(alpha=0.2)
    ax_r = ax.twinx()
    ax_r.plot(x, work[("LV", "PLV")], "-o", color="C0", lw=lw, ms=ms - 1, alpha=0.85)
    ax_r.set_ylabel(r"$W^{density}_{proxy}$ (kPa)", fontsize=10, color="C0")
    ax_r.tick_params(axis="y", labelcolor="C0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=10)
    ax.set_title("LV free wall", fontsize=12, fontweight="bold")
    r_lv = table_r["LV"]["PLV"]
    ax.legend([Line2D([0], [0], color="k", marker="D", lw=lw + 0.5, ms=ms),
                Line2D([0], [0], color="C0", marker="o", lw=lw, ms=ms - 1)],
               [r"$W^{density}_{true}$ (left)",
                fr"$P_{{LV}}$ proxy density (right)  r={r_lv:+.3f}"],
               fontsize=8, loc="lower left")

    # ── RV panel (one proxy: P_RV) ────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, work[("RV", "true")], "-D", color="k", lw=lw + 0.5, ms=ms + 1, zorder=5)
    ax.set_ylabel(r"$W^{density}_{true}$ (kPa)", fontsize=10)
    ax.grid(alpha=0.2)
    ax_r = ax.twinx()
    ax_r.plot(x, work[("RV", "PRV")], "-s", color="C3", lw=lw, ms=ms - 1, alpha=0.85)
    ax_r.set_ylabel(r"$W^{density}_{proxy}$ (kPa)", fontsize=10, color="C3")
    ax_r.tick_params(axis="y", labelcolor="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=10)
    ax.set_title("RV free wall", fontsize=12, fontweight="bold")
    r_rv = table_r["RV"]["PRV"]
    ax.legend([Line2D([0], [0], color="k", marker="D", lw=lw + 0.5, ms=ms),
                Line2D([0], [0], color="C3", marker="s", lw=lw, ms=ms - 1)],
               [r"$W^{density}_{true}$ (left)",
                fr"$P_{{RV}}$ proxy density (right)  r={r_rv:+.3f}"],
               fontsize=8, loc="lower left")

    # ── Septum panel (all three proxies, opacity ∝ |r|) ─────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x, work[("Septum", "true")], "-D", color="k", lw=lw + 0.5, ms=ms + 1, zorder=5)
    ax.set_ylabel(r"$W^{density}_{true}$ (kPa)", fontsize=10)
    ax.grid(alpha=0.2)
    ax_r = ax.twinx()
    r_plv   = table_r["Septum"]["PLV"]
    r_prv   = table_r["Septum"]["PRV"]
    r_trans = table_r["Septum"]["Trans"]
    ax_r.plot(x, work[("Septum", "PLV")],  "-o", color="C0", lw=lw, ms=ms - 1, alpha=r_to_alpha(r_plv))
    ax_r.plot(x, work[("Septum", "PRV")],  "-s", color="C3", lw=lw, ms=ms - 1, alpha=r_to_alpha(r_prv))
    ax_r.plot(x, work[("Septum", "Trans")], "-^", color="C2", lw=lw, ms=ms - 1, alpha=r_to_alpha(r_trans))
    ax_r.set_ylabel(r"$W^{density}_{proxy}$ (kPa)", fontsize=10, color="gray")
    ax_r.tick_params(axis="y", labelcolor="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=10)
    ax.set_title(f"Septum ({title_septum}, {n_sep_cells} cells)", fontsize=12, fontweight="bold")
    ax.legend([Line2D([0], [0], color="k", marker="D", lw=lw + 0.5, ms=ms),
                Line2D([0], [0], color="C0", marker="o", lw=lw, ms=ms - 1),
                Line2D([0], [0], color="C3", marker="s", lw=lw, ms=ms - 1),
                Line2D([0], [0], color="C2", marker="^", lw=lw, ms=ms - 1)],
               [r"$W^{density}_{true}$ (left)",
                r"$P_{LV}$", r"$P_{RV}$", r"$P_{LV}-P_{RV}$"],
               fontsize=8, loc="lower left", title="right axis:", title_fontsize=7)

    # ── Correlation table (3×3) ───────────────────────────────────
    ax_tab = fig.add_subplot(gs[1, 1])
    ax_tab.axis("off")
    ax_tab.set_title(r"Pearson r  ($W^{density}_{true}$ vs $W^{density}_{proxy}$)",
                      fontsize=12, fontweight="bold", pad=15)
    col_labels = [r"$P_{LV}$", r"$P_{RV}$", r"$P_{Trans}$"]
    row_labels = ["LV", "RV", "Septum"]
    cell_text, cell_colors = [], []
    for rname in row_labels:
        rt, rc = [], []
        for pname in ["PLV", "PRV", "Trans"]:
            r_val = table_r[rname][pname]
            rt.append(f"{r_val:+.3f}" if not np.isnan(r_val) else "—")
            ar = abs(r_val) if not np.isnan(r_val) else 0
            if not np.isnan(r_val) and r_val > 0:
                g = min(1.0, 0.6 + 0.4 * ar)
                rc.append((1 - 0.35 * ar, g, 1 - 0.35 * ar))
            else:
                rc.append((1, 1 - 0.35 * ar, 1 - 0.35 * ar))
        cell_text.append(rt)
        cell_colors.append(rc)

    table = ax_tab.table(cellText=cell_text, rowLabels=row_labels,
                          colLabels=col_labels, cellColours=cell_colors,
                          cellLoc="center", bbox=[0.15, 0.25, 0.8, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    for i, rname in enumerate(row_labels):
        vals = [abs(table_r[rname][p]) if not np.isnan(table_r[rname][p]) else 0
                for p in ["PLV", "PRV", "Trans"]]
        best_col = int(np.argmax(vals))
        table[i + 1, best_col].set_text_props(fontweight="bold", fontsize=14)

    ax_tab.text(0.55, 0.07,
                 f"Septum: {title_septum}  |  "
                 f"Strain: volume-weighted longitudinal  |  n={n}\n"
                 f"Sweep: {sweep_label}  |  Units: kPa (= mJ / mL)",
                 transform=ax_tab.transAxes, fontsize=9, ha="center", va="bottom",
                 style="italic", color="gray")

    fig.suptitle(f"Work DENSITY across {sweep_label} — {title_septum} septum\n"
                  f"Intensive comparison: true density vs intensive clinical proxy",
                  fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def load_case(d):
    d = Path(d).resolve()
    pc_path = d / "per_cell_data.npz"
    if not pc_path.exists():
        return None
    pc = np.load(pc_path, allow_pickle=True)
    desc = (d / "run_description.txt").read_text().strip() if (d / "run_description.txt").exists() else ""
    return {"dir": d, "pc": {k: pc[k] for k in pc.files}, "desc": desc}


# ── Mode dispatchers ───────────────────────────────────────────────────────
SPECTRUM_KEYS = ["healthy", "mild", "moderate_severe", "moderate",
                  "severe", "very_severe", "end_stage", "borderline"]
SPECTRUM_DISPLAY = {
    "healthy": "Borderline PH",
    "mild": "Mild",
    "moderate": "Moderate",
    "moderate_severe": "Mod–severe",
    "severe": "Severe",
    "very_severe": "Very severe",
    "end_stage": "End-stage",
    "borderline": "Borderline",
}
SPECTRUM_ORDER = ["healthy", "borderline", "mild", "moderate", "moderate_severe",
                   "severe", "very_severe", "end_stage"]

RVFW_ORDER = ["rvfw_03mm", "rvfw_4p5mm", "rvfw_06mm", "rvfw_7p5mm",
               "rvfw_09mm", "rvfw_10p5mm", "rvfw_12mm"]
LVFW_ORDER = ["lvfw_03mm", "lvfw_06mm", "lvfw_09mm", "lvfw_12mm"]
THICKNESS_LABELS = {
    "rvfw_03mm": "+3", "rvfw_4p5mm": "+4.5", "rvfw_06mm": "+6",
    "rvfw_7p5mm": "+7.5", "rvfw_09mm": "+9", "rvfw_10p5mm": "+10.5",
    "rvfw_12mm": "+12",
    "lvfw_03mm": "+3", "lvfw_06mm": "+6", "lvfw_09mm": "+9", "lvfw_12mm": "+12",
}


def parse_severity(desc):
    for k in sorted(SPECTRUM_KEYS, key=len, reverse=True):
        if f" {k} " in f" {desc} ":
            return k
    return None


def parse_variant(desc):
    for k in RVFW_ORDER + LVFW_ORDER:
        if k in desc:
            return k
    return None


def mode_spectrum(cases, out_dir):
    """Spectrum sweep: 7 severities, one mesh, LDRB + geometric septum."""
    # Match each case to a severity key
    tagged = []
    for c in cases:
        sev = parse_severity(c["desc"])
        if sev is None:
            print(f"  SKIP {c['dir'].name}: cannot parse severity from: {c['desc']}")
            continue
        tagged.append({"case": c, "severity": sev})

    for sep_key, sep_name in [("is_ldrb_septum", "LDRB"),
                                ("is_geometric_septum", "Geometric")]:
        ordered = []
        labels = []
        for sev in SPECTRUM_ORDER:
            hit = next((t for t in tagged if t["severity"] == sev), None)
            if hit is None:
                continue
            densities = region_densities(hit["case"]["pc"], sep_key)
            if any(densities[r] is None for r in ("LV", "RV", "Septum")):
                print(f"  SKIP {sev} ({sep_name}): empty region")
                continue
            ordered.append({"label": SPECTRUM_DISPLAY[sev], "densities": densities})
            labels.append(SPECTRUM_DISPLAY[sev])
        if len(ordered) < 3:
            print(f"  SKIP spectrum {sep_name}: only {len(ordered)} cases")
            continue
        n_sep = int(np.mean([c["case"]["pc"][sep_key].sum() for c in tagged
                              if SPECTRUM_DISPLAY[c["severity"]] in labels]))
        out_path = out_dir / f"work_density_spectrum_{sep_name.lower()}.png"
        make_plot(ordered, labels, sep_key, sep_name,
                   "PAH severity spectrum (n=7 hemodynamic regimes)",
                   out_path, n_sep_cells=n_sep)


def mode_thickness(cases, out_dir):
    """Thickness sweep: 2 severities × 2 families (RVFW, LVFW) × 2 septum defs."""
    tagged = []
    for c in cases:
        sev = parse_severity(c["desc"])
        var = parse_variant(c["desc"])
        if sev is None or var is None:
            print(f"  SKIP {c['dir'].name}: cannot parse: {c['desc']}")
            continue
        tagged.append({"case": c, "severity": sev, "variant": var})

    for sev in ("severe", "healthy"):
        for family_name, family_order, family_title in [
            ("rvfw", RVFW_ORDER, "RV free wall thickness (+mm)"),
            ("lvfw", LVFW_ORDER, "LV free wall thickness (+mm)"),
        ]:
            subset = [t for t in tagged if t["severity"] == sev and t["variant"] in family_order]
            if len(subset) < 3:
                print(f"  SKIP {sev}/{family_name}: only {len(subset)} cases")
                continue
            subset.sort(key=lambda t: family_order.index(t["variant"]))
            for sep_key, sep_name in [("is_ldrb_septum", "LDRB"),
                                        ("is_geometric_septum", "Geometric")]:
                ordered = []
                labels = []
                for t in subset:
                    densities = region_densities(t["case"]["pc"], sep_key)
                    if any(densities[r] is None for r in ("LV", "RV", "Septum")):
                        continue
                    ordered.append({"label": THICKNESS_LABELS[t["variant"]],
                                     "densities": densities})
                    labels.append(THICKNESS_LABELS[t["variant"]])
                if len(ordered) < 3:
                    continue
                n_sep = int(np.mean([t["case"]["pc"][sep_key].sum() for t in subset]))
                sev_label = "severe PAH" if sev == "severe" else "healthy"
                out_path = (out_dir /
                            f"work_density_thickness_{sev}_{sep_name.lower()}_{family_name}.png")
                make_plot(ordered, labels, sep_key, sep_name,
                           f"{family_title}  ({sev_label} circulation)",
                           out_path, n_sep_cells=n_sep)


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["spectrum", "thickness"], required=True)
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    default_out = {
        "spectrum":  Path("results/analysis/work_density_spectrum"),
        "thickness": Path("results/analysis/work_density_thickness"),
    }
    out_dir = args.output_dir or default_out[args.mode]
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for d in args.result_dirs:
        c = load_case(d)
        if c is None:
            print(f"  SKIP {d}: no per_cell_data.npz")
            continue
        cases.append(c)
    print(f"Loaded {len(cases)} cases")

    if args.mode == "spectrum":
        mode_spectrum(cases, out_dir)
    else:
        mode_thickness(cases, out_dir)

    # Also print the correlation tables as console output for the data dump
    print()
    print("=" * 72)
    print(f"DATA TABLES — {args.mode.upper()}  (W density in kPa = mJ/mL)")
    print("=" * 72)
    print_tables(cases, args.mode)
    print("Done.")


def print_tables(cases, mode):
    """Print raw W_density tables and Pearson r tables to stdout."""
    PA_TO_KPA = 1e-3
    if mode == "spectrum":
        tagged = [(parse_severity(c["desc"]), c) for c in cases]
        tagged = [(s, c) for (s, c) in tagged if s is not None]
        for sep_key, sep_name in [("is_ldrb_septum", "LDRB"),
                                    ("is_geometric_septum", "Geometric")]:
            print(f"\n--- Spectrum / {sep_name} septum ---")
            rows = []
            for sev in SPECTRUM_ORDER:
                hit = next(((s, c) for (s, c) in tagged if s == sev), None)
                if hit is None:
                    continue
                _s, c = hit
                d = region_densities(c["pc"], sep_key)
                if any(d[r] is None for r in ("LV", "RV", "Septum")):
                    continue
                rows.append((SPECTRUM_DISPLAY[sev], d))
            if len(rows) < 3:
                continue
            print(f"{'severity':<16} {'LV W_true':>10} {'LV r_PLV':>10}  "
                  f"{'RV W_true':>10} {'RV r_PRV':>10}  "
                  f"{'Sep W_true':>11} {'Sep r_Tr':>10}")
            # Print raw densities
            for lbl, d in rows:
                print(f"{lbl:<16} "
                      f"{d['LV']['W_true']*PA_TO_KPA:>+9.3f}k "
                      f"{'':>10}  "
                      f"{d['RV']['W_true']*PA_TO_KPA:>+9.3f}k "
                      f"{'':>10}  "
                      f"{d['Septum']['W_true']*PA_TO_KPA:>+10.3f}k "
                      f"{'':>10}")
            # Print Pearson r
            for rname in ("LV", "RV", "Septum"):
                Wt = np.array([r[1][rname]["W_true"] for r in rows])
                for p in ("PLV", "PRV", "Trans"):
                    Wp = np.array([r[1][rname][p] for r in rows])
                    r = safe_r(Wt, Wp)
                    print(f"    r({rname}, {p:<5}) = {r:+.4f}")
    else:
        # thickness
        tagged = [(parse_severity(c["desc"]), parse_variant(c["desc"]), c) for c in cases]
        tagged = [t for t in tagged if t[0] is not None and t[1] is not None]
        for sev in ("severe", "healthy"):
            for family_name, family_order in [("RVFW", RVFW_ORDER), ("LVFW", LVFW_ORDER)]:
                subset = [t for t in tagged if t[0] == sev and t[1] in family_order]
                if len(subset) < 3:
                    continue
                subset.sort(key=lambda t: family_order.index(t[1]))
                for sep_key, sep_name in [("is_ldrb_septum", "LDRB"),
                                            ("is_geometric_septum", "Geometric")]:
                    rows = []
                    for _s, v, c in subset:
                        d = region_densities(c["pc"], sep_key)
                        if any(d[r] is None for r in ("LV", "RV", "Septum")):
                            continue
                        rows.append((THICKNESS_LABELS[v], d))
                    if len(rows) < 3:
                        continue
                    print(f"\n--- Thickness / {sev} / {family_name} / {sep_name} septum (n={len(rows)}) ---")
                    print(f"{'variant':<8} {'LV W_true':>12} {'RV W_true':>12} "
                          f"{'Sep W_true':>12}  (all in kPa)")
                    for lbl, d in rows:
                        print(f"{lbl:<8} "
                              f"{d['LV']['W_true']*PA_TO_KPA:>+11.3f} "
                              f"{d['RV']['W_true']*PA_TO_KPA:>+11.3f} "
                              f"{d['Septum']['W_true']*PA_TO_KPA:>+11.3f}")
                    for rname in ("LV", "RV", "Septum"):
                        Wt = np.array([r[1][rname]["W_true"] for r in rows])
                        line = f"    {rname:<8} "
                        for p in ("PLV", "PRV", "Trans"):
                            Wp = np.array([r[1][rname][p] for r in rows])
                            r = safe_r(Wt, Wp)
                            line += f"r({p:<5})={r:+7.4f}  "
                        print(line)


if __name__ == "__main__":
    main()
