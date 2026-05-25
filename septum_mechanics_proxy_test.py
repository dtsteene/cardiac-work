#!/usr/bin/env python3
"""Mechanistic septum proxy tests.

The septum is not a free wall.  This script treats it as a shared wall and
tests pressure choices that are more natural for a two-sided structure:

    P_LV, P_RV, P_LV - P_RV, mean(P_LV, P_RV)
    tau-weighted pressure through the septal wall
    nearest-side pressure through the septal wall
    lambda P_LV + (1 - lambda) P_RV

Two questions are kept separate:

1. Which pressure choice best orders cases in the pressure sweep?
2. Which pressure choice best preserves septum/free-wall magnitude ratios?

Those are not the same question, and for the septum that distinction matters.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


ROOT = Path("/home/dtsteene/D1/cardiac-work")
OUT = ROOT / "results" / "analysis" / "septum_mechanics"
KPA = 1e-3

CASES = [
    ("sPAP22", 1047450),
    ("sPAP25", 1048194),
    ("sPAP30", 1047451),
    ("sPAP35", 1048195),
    ("sPAP45", 1047452),
    ("sPAP50", 1048196),
    ("sPAP55", 1047453),
    ("sPAP60", 1048197),
    ("sPAP65", 1047454),
    ("sPAP70", 1048198),
    ("sPAP75", 1047455),
    ("sPAP80", 1048199),
    ("sPAP85", 1047456),
    ("sPAP87", 1048200),
    ("sPAP92", 1048201),
    ("sPAP95", 1047457),
]


def find_run(job_id: int) -> Path:
    for root in [
        ROOT / "results" / "sims" / "2026-04-23",
        ROOT / "results" / "sims" / "2026-04-24",
    ]:
        run = root / f"UKB_6beats_run_{job_id}"
        if run.exists():
            return run
    raise FileNotFoundError(job_id)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(pearsonr(x, y)[0])


def density(pc: np.lib.npyio.NpzFile, mask: np.ndarray, values: str | np.ndarray) -> float:
    arr = pc[values] if isinstance(values, str) else values
    volume = float(pc["cell_volumes"][mask].sum())
    return float(-arr[mask].sum() / volume * KPA)


def masks(pc: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    tags = pc["region_tags"]
    return {
        "LV": tags == 1,
        "RV": tags == 2,
        "Septum": pc["is_geometric_septum"].astype(bool),
    }


def septal_pressure_candidates(pc: np.lib.npyio.NpzFile, strain_suffix: str) -> dict[str, np.ndarray]:
    # Canonical convention for pressure choices: tau=0 on the LV side and
    # tau=1 on the RV side. The saved Laplace scalar has the opposite orientation.
    tau = 1.0 - pc["lv_rv_scalar"] if "lv_rv_scalar" in pc.files else pc["tau"]
    plv = pc[f"proxy_PLV_{strain_suffix}"]
    prv = pc[f"proxy_PRV_{strain_suffix}"]
    return {
        "PLV": plv,
        "PRV": prv,
        "Trans": plv - prv,
        "Mean": 0.5 * (plv + prv),
        "NearestSide": np.where(tau < 0.5, plv, prv),
        "TauWeighted": (1.0 - tau) * plv + tau * prv,
    }


def case_values() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case, job_id in CASES:
        pc = np.load(find_run(job_id) / "per_cell_data.npz", allow_pickle=True)
        region_masks = masks(pc)
        sept = region_masks["Septum"]
        tau = 1.0 - pc["lv_rv_scalar"] if "lv_rv_scalar" in pc.files else pc["tau"]

        base: dict[str, object] = {
            "case": case,
            "job_id": job_id,
            "W_total_LV": density(pc, region_masks["LV"], "w_total"),
            "W_total_RV": density(pc, region_masks["RV"], "w_total"),
            "W_total_Septum": density(pc, sept, "w_total"),
            "W_ff_Septum": density(pc, sept, "w_ff"),
            "W_ss_Septum": density(pc, sept, "w_ss"),
            "W_nn_Septum": density(pc, sept, "w_nn"),
            "W_cross_Septum": density(pc, sept, "w_cross"),
            "tau_min_Septum": float(tau[sept].min()),
            "tau_max_Septum": float(tau[sept].max()),
            "tau_mean_Septum": float(tau[sept].mean()),
        }
        for strain_suffix in ("ll", "ff"):
            for region_name, mask in region_masks.items():
                base[f"{strain_suffix}_adjacent_{region_name}"] = density(
                    pc,
                    mask,
                    f"proxy_PLV_{strain_suffix}"
                    if region_name == "LV"
                    else f"proxy_PRV_{strain_suffix}",
                )
            for proxy, values in septal_pressure_candidates(pc, strain_suffix).items():
                base[f"{strain_suffix}_{proxy}_Septum"] = density(pc, sept, values)
        rows.append(base)
    return rows


def correlation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    targets = [
        ("W_total_Septum", "septum total tensor work"),
        ("W_ff_Septum", "septum fibre work"),
    ]
    proxies = ["PLV", "PRV", "Trans", "Mean", "NearestSide", "TauWeighted"]
    for strain_suffix, strain_label in [("ll", "longitudinal strain"), ("ff", "fibre strain")]:
        for target_key, target_label in targets:
            y = np.array([float(r[target_key]) for r in rows])
            for proxy in proxies:
                x = np.array([float(r[f"{strain_suffix}_{proxy}_Septum"]) for r in rows])
                rv = corr(x, y)
                out.append(
                    {
                        "strain": strain_label,
                        "target": target_label,
                        "target_key": target_key,
                        "proxy": proxy,
                        "r": rv,
                        "r2": rv * rv,
                    }
                )
    return out


def ratio_error_for_values(
    rows: list[dict[str, object]],
    strain_suffix: str,
    septum_proxy_values: np.ndarray,
) -> tuple[float, float, float]:
    """Return mean abs log error plus mean absolute raw errors for S/LV and S/RV."""
    log_errors = []
    raw_errors = []
    for row, septum_proxy in zip(rows, septum_proxy_values):
        true_s_lv = float(row["W_total_Septum"]) / float(row["W_total_LV"])
        true_s_rv = float(row["W_total_Septum"]) / float(row["W_total_RV"])
        pred_s_lv = septum_proxy / float(row[f"{strain_suffix}_adjacent_LV"])
        pred_s_rv = septum_proxy / float(row[f"{strain_suffix}_adjacent_RV"])
        for true, pred in [(true_s_lv, pred_s_lv), (true_s_rv, pred_s_rv)]:
            raw_errors.append(abs(pred - true))
            if true > 0 and pred > 0:
                log_errors.append(abs(np.log(pred / true)))
    return float(np.mean(log_errors)), float(np.mean(raw_errors)), float(np.max(raw_errors))


def ratio_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    proxies = ["PLV", "PRV", "Trans", "Mean", "NearestSide", "TauWeighted"]
    for strain_suffix, strain_label in [("ll", "longitudinal strain"), ("ff", "fibre strain")]:
        for proxy in proxies:
            values = np.array([float(r[f"{strain_suffix}_{proxy}_Septum"]) for r in rows])
            mean_log, mean_raw, max_raw = ratio_error_for_values(rows, strain_suffix, values)
            out.append(
                {
                    "strain": strain_label,
                    "proxy": proxy,
                    "mean_abs_log_ratio_error": mean_log,
                    "mean_abs_raw_ratio_error": mean_raw,
                    "max_abs_raw_ratio_error": max_raw,
                }
            )
    return out


def lambda_scan(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    lambdas = np.linspace(-1.0, 2.0, 601)
    scan_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    for strain_suffix, strain_label in [("ll", "longitudinal strain"), ("ff", "fibre strain")]:
        plv = np.array([float(r[f"{strain_suffix}_PLV_Septum"]) for r in rows])
        prv = np.array([float(r[f"{strain_suffix}_PRV_Septum"]) for r in rows])
        y_total = np.array([float(r["W_total_Septum"]) for r in rows])
        y_ff = np.array([float(r["W_ff_Septum"]) for r in rows])

        for lam in lambdas:
            values = lam * plv + (1.0 - lam) * prv
            ratio_log, ratio_raw, _ = ratio_error_for_values(rows, strain_suffix, values)
            scan_rows.append(
                {
                    "strain": strain_label,
                    "lambda": float(lam),
                    "r_total": corr(values, y_total),
                    "r_fibre": corr(values, y_ff),
                    "mean_abs_log_ratio_error": ratio_log,
                    "mean_abs_raw_ratio_error": ratio_raw,
                }
            )

        strain_scan = [r for r in scan_rows if r["strain"] == strain_label]
        for objective, selector in [
            ("best correlation with total tensor work", lambda r: float(r["r_total"])),
            ("best correlation with fibre work", lambda r: float(r["r_fibre"])),
            ("best septum/free-wall ratio preservation", lambda r: -float(r["mean_abs_log_ratio_error"])),
        ]:
            best = max(strain_scan, key=selector)
            best_rows.append(
                {
                    "strain": strain_label,
                    "objective": objective,
                    "lambda": best["lambda"],
                    "r_total": best["r_total"],
                    "r_fibre": best["r_fibre"],
                    "mean_abs_log_ratio_error": best["mean_abs_log_ratio_error"],
                    "interpretation": lambda_interpretation(float(best["lambda"])),
                }
            )
    return scan_rows, best_rows


def lambda_interpretation(lam: float) -> str:
    if abs(lam) < 0.03:
        return "approximately RV pressure"
    if abs(lam - 0.5) < 0.03:
        return "approximately mean pressure"
    if abs(lam - 1.0) < 0.03:
        return "approximately LV pressure"
    if lam > 1.0:
        return "LV pressure with RV pressure subtracted"
    if 0.5 < lam < 1.0:
        return "between mean pressure and LV pressure"
    if 0.0 < lam < 0.5:
        return "between RV pressure and mean pressure"
    return "RV pressure plus an opposite LV term"


def layer_rows() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    selected = {"sPAP22", "sPAP45", "sPAP75", "sPAP95"}
    for case, job_id in CASES:
        if case not in selected:
            continue
        pc = np.load(find_run(job_id) / "per_cell_data.npz", allow_pickle=True)
        sept = pc["is_geometric_septum"].astype(bool)
        tau = 1.0 - pc["lv_rv_scalar"] if "lv_rv_scalar" in pc.files else pc["tau"]
        q = np.quantile(tau[sept], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        layers = [
            ("LV-side third", q[0], q[1]),
            ("middle third", q[1], q[2]),
            ("RV-side third", q[2], q[3] + 1e-12),
        ]
        for layer, lo, hi in layers:
            mask = sept & (tau >= lo) & (tau < hi)
            plv = density(pc, mask, "proxy_PLV_ll")
            prv = density(pc, mask, "proxy_PRV_ll")
            out.append(
                {
                    "case": case,
                    "layer": layer,
                    "tau_min": float(tau[mask].min()),
                    "tau_max": float(tau[mask].max()),
                    "W_total": density(pc, mask, "w_total"),
                    "W_ff": density(pc, mask, "w_ff"),
                    "W_nn": density(pc, mask, "w_nn"),
                    "PLV_ll": plv,
                    "PRV_ll": prv,
                    "Mean_ll": 0.5 * (plv + prv),
                    "Trans_ll": plv - prv,
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(corrs: list[dict[str, object]], ratios: list[dict[str, object]], bests: list[dict[str, object]]) -> None:
    print("=" * 86)
    print("Septum proxy correlations, corrected n=16")
    print("=" * 86)
    for strain in ["longitudinal strain", "fibre strain"]:
        print(f"\n{strain}")
        print(f"{'proxy':<14} {'r vs total':>12} {'r vs fibre':>12} {'ratio log err':>15}")
        for proxy in ["PLV", "PRV", "Trans", "Mean", "NearestSide", "TauWeighted"]:
            total = next(
                r
                for r in corrs
                if r["strain"] == strain and r["proxy"] == proxy and r["target_key"] == "W_total_Septum"
            )
            fibre = next(
                r
                for r in corrs
                if r["strain"] == strain and r["proxy"] == proxy and r["target_key"] == "W_ff_Septum"
            )
            ratio = next(r for r in ratios if r["strain"] == strain and r["proxy"] == proxy)
            print(
                f"{proxy:<14} {float(total['r']):>+12.3f} "
                f"{float(fibre['r']):>+12.3f} "
                f"{float(ratio['mean_abs_log_ratio_error']):>15.3f}"
            )

    print("\nBest lambda pressure mixtures: lambda P_LV + (1-lambda) P_RV")
    print(f"{'strain':<20} {'objective':<42} {'lambda':>8} {'interpretation'}")
    for row in bests:
        print(
            f"{row['strain']:<20} {row['objective']:<42} "
            f"{float(row['lambda']):>8.3f} {row['interpretation']}"
        )


def make_lambda_figure(scan_rows: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    colors = {"longitudinal strain": "#4C78A8", "fibre strain": "#E15759"}
    for strain, color in colors.items():
        subset = [r for r in scan_rows if r["strain"] == strain]
        lam = np.array([float(r["lambda"]) for r in subset])
        r_total = np.array([float(r["r_total"]) for r in subset])
        ratio_err = np.array([float(r["mean_abs_log_ratio_error"]) for r in subset])
        axes[0].plot(lam, r_total, color=color, linewidth=2.0, label=strain)
        axes[1].plot(lam, ratio_err, color=color, linewidth=2.0, label=strain)

    for ax in axes:
        for x, label in [(0.0, "RV"), (0.5, "mean"), (1.0, "LV")]:
            ax.axvline(x, color="0.35", linestyle="--", linewidth=1.0)
            ax.text(x, ax.get_ylim()[1], label, ha="center", va="top", fontsize=9)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(r"$\lambda$ in $\lambda P_{LV} + (1-\lambda)P_{RV}$")
    axes[0].set_ylabel("correlation with septum tensor work")
    axes[1].set_ylabel("mean abs log ratio error")
    axes[0].legend(framealpha=0.95)
    path = OUT / "fig_septum_lambda_scan.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "fig_septum_lambda_scan.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def make_layer_figure(rows: list[dict[str, object]]) -> Path:
    cases = ["sPAP22", "sPAP45", "sPAP75", "sPAP95"]
    layers = ["LV-side third", "middle third", "RV-side third"]
    fig, axes = plt.subplots(1, len(cases), figsize=(12, 3.8), sharey=True, constrained_layout=True)
    for ax, case in zip(axes, cases):
        vals = [
            float(next(r for r in rows if r["case"] == case and r["layer"] == layer)["W_total"])
            for layer in layers
        ]
        ax.bar(np.arange(3), vals, color=["#4C78A8", "#BAB0AC", "#F28E2B"], width=0.65)
        ax.set_title(case)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(["LV side", "mid", "RV side"], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("septal tensor work density (kPa)")
    path = OUT / "fig_septum_layer_work.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "fig_septum_layer_work.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = case_values()
    corrs = correlation_rows(rows)
    ratios = ratio_rows(rows)
    scan, bests = lambda_scan(rows)
    layers = layer_rows()

    print_summary(corrs, ratios, bests)

    write_csv(OUT / "septum_case_values.csv", rows)
    write_csv(OUT / "septum_proxy_correlations.csv", corrs)
    write_csv(OUT / "septum_ratio_preservation.csv", ratios)
    write_csv(OUT / "septum_lambda_scan.csv", scan)
    write_csv(OUT / "septum_best_lambda.csv", bests)
    write_csv(OUT / "septum_layer_summary.csv", layers)
    fig1 = make_lambda_figure(scan)
    fig2 = make_layer_figure(layers)

    print(f"\nSaved {OUT / 'septum_case_values.csv'}")
    print(f"Saved {OUT / 'septum_proxy_correlations.csv'}")
    print(f"Saved {OUT / 'septum_ratio_preservation.csv'}")
    print(f"Saved {OUT / 'septum_lambda_scan.csv'}")
    print(f"Saved {OUT / 'septum_best_lambda.csv'}")
    print(f"Saved {OUT / 'septum_layer_summary.csv'}")
    print(f"Saved {fig1}")
    print(f"Saved {fig2}")


if __name__ == "__main__":
    main()
