#!/usr/bin/env python3
"""Compare clinical-style strain directions in the h=5 pressure sweep.

The thesis primarily tests pressure-longitudinal-strain work.  Newer
per-cell postprocessing files also contain geometric radial and
circumferential pressure-strain integrals.  This script asks whether those
candidate strain directions improve the same model-side tests:

1. preservation of the free-wall LV/RV tensor-work ratio,
2. correlation with septal tensor work across the loading sweep, and
3. preservation of septum/free-wall work-density magnitude.

Radial and circumferential arrays are not available for the three endpoint
mesh-convergence cases in the h=5 manifest, so the main cross-direction
comparison uses the common subset where all four directions are present.
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
SIM_ROOT = ROOT / "results" / "sims"
MANIFEST = ROOT / "results" / "analysis" / "h5_sweep_submission" / "h5_corrected_sweep_cases.tsv"
OUT = ROOT / "results" / "analysis" / "h5_strain_directions"
KPA = 1e-3

STRAINS = {
    "ll": "longitudinal",
    "ff": "fibre",
    "radial": "geometric radial",
    "circ": "geometric circumferential",
}
PRESSURES = ["PLV", "PRV", "Trans", "Mean", "NearestSide", "TauWeighted"]


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def find_run(job_id: str) -> Path:
    matches = sorted(SIM_ROOT.glob(f"*/UKB_6beats_run_{job_id}"))
    if not matches:
        raise FileNotFoundError(job_id)
    return matches[-1]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def corr(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) < 3 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(pearsonr(x_arr, y_arr)[0])


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


def has_strain(pc: np.lib.npyio.NpzFile, suffix: str) -> bool:
    return f"proxy_PLV_{suffix}" in pc.files and f"proxy_PRV_{suffix}" in pc.files


def pressure_arrays(pc: np.lib.npyio.NpzFile, suffix: str) -> dict[str, np.ndarray]:
    # Canonical convention for pressure choices: tau=0 on the LV side and
    # tau=1 on the RV side. The saved Laplace scalar has the opposite orientation.
    tau = 1.0 - pc["lv_rv_scalar"] if "lv_rv_scalar" in pc.files else pc["tau"]
    plv = pc[f"proxy_PLV_{suffix}"]
    prv = pc[f"proxy_PRV_{suffix}"]
    trans_key = f"proxy_Trans_{suffix}"
    trans = pc[trans_key] if trans_key in pc.files else plv - prv
    return {
        "PLV": plv,
        "PRV": prv,
        "Trans": trans,
        "Mean": 0.5 * (plv + prv),
        "NearestSide": np.where(tau < 0.5, plv, prv),
        "TauWeighted": (1.0 - tau) * plv + tau * prv,
    }


def case_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in read_manifest():
        case = item["case"]
        run = find_run(item["job_id"])
        pc = np.load(run / "per_cell_data.npz", allow_pickle=True)
        region_masks = masks(pc)

        row: dict[str, object] = {
            "case": case,
            "job_id": item["job_id"],
            "run_dir": str(run),
        }
        for suffix in STRAINS:
            row[f"has_{suffix}"] = has_strain(pc, suffix)

        for region, mask in region_masks.items():
            row[f"{region}_volume_mL"] = float(pc["cell_volumes"][mask].sum() * 1e6)
            row[f"{region}_W_total_kPa"] = density(pc, mask, "w_total")
            row[f"{region}_W_ff_kPa"] = density(pc, mask, "w_ff")
            row[f"{region}_W_ss_kPa"] = density(pc, mask, "w_ss")
            row[f"{region}_W_nn_kPa"] = density(pc, mask, "w_nn")
            row[f"{region}_W_cross_kPa"] = density(pc, mask, "w_cross")
            for suffix in STRAINS:
                if not has_strain(pc, suffix):
                    continue
                arrays = pressure_arrays(pc, suffix)
                for pressure, values in arrays.items():
                    row[f"{region}_{pressure}_{suffix}_kPa"] = density(pc, mask, values)

        row["FW_tensor_LV_RV_ratio"] = float(row["LV_W_total_kPa"]) / float(row["RV_W_total_kPa"])
        row["Septum_to_FWmean_tensor_ratio"] = float(row["Septum_W_total_kPa"]) / (
            0.5 * (float(row["LV_W_total_kPa"]) + float(row["RV_W_total_kPa"]))
        )
        rows.append(row)
    return rows


def ratio_error(values: list[float], targets: list[float]) -> tuple[float, float, int]:
    raw = [abs(v - t) for v, t in zip(values, targets)]
    log = [
        abs(np.log(v / t))
        for v, t in zip(values, targets)
        if np.isfinite(v) and np.isfinite(t) and v > 0 and t > 0
    ]
    return float(np.mean(raw)), float(np.mean(log)) if log else float("nan"), len(log)


def summarize(rows: list[dict[str, object]], cohort: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    tensor_fw_ratio = [float(r["FW_tensor_LV_RV_ratio"]) for r in rows]
    tensor_sept_ratio = [float(r["Septum_to_FWmean_tensor_ratio"]) for r in rows]
    sept_tensor = [float(r["Septum_W_total_kPa"]) for r in rows]
    rv_tensor = [float(r["RV_W_total_kPa"]) for r in rows]

    for suffix, strain_label in STRAINS.items():
        available = [r for r in rows if r.get(f"has_{suffix}")]
        if not available:
            continue

        fw_pred = [
            float(r[f"LV_PLV_{suffix}_kPa"]) / float(r[f"RV_PRV_{suffix}_kPa"])
            for r in available
        ]
        fw_target = [float(r["FW_tensor_LV_RV_ratio"]) for r in available]
        fw_raw, fw_log, fw_log_n = ratio_error(fw_pred, fw_target)

        rv_proxy = [float(r[f"RV_PRV_{suffix}_kPa"]) for r in available]
        rv_target = [float(r["RV_W_total_kPa"]) for r in available]

        for pressure in PRESSURES:
            sept_key = f"Septum_{pressure}_{suffix}_kPa"
            if sept_key not in available[0]:
                continue
            sept_proxy = [float(r[sept_key]) for r in available]
            fw_mean_proxy = [
                0.5 * (float(r[f"LV_PLV_{suffix}_kPa"]) + float(r[f"RV_PRV_{suffix}_kPa"]))
                for r in available
            ]
            sept_ratio_pred = [s / f if abs(f) > 1e-30 else float("nan") for s, f in zip(sept_proxy, fw_mean_proxy)]
            sept_ratio_target = [float(r["Septum_to_FWmean_tensor_ratio"]) for r in available]
            sept_raw, sept_log, sept_log_n = ratio_error(sept_ratio_pred, sept_ratio_target)
            out.append(
                {
                    "cohort": cohort,
                    "n": len(available),
                    "strain_suffix": suffix,
                    "strain": strain_label,
                    "pressure": pressure,
                    "freewall_adjacent_mean_abs_ratio_error": fw_raw,
                    "freewall_adjacent_mean_abs_log_ratio_error": fw_log,
                    "freewall_adjacent_log_n": fw_log_n,
                    "freewall_ratio_r": corr(fw_pred, fw_target),
                    "rv_freewall_adjacent_r": corr(rv_proxy, rv_target),
                    "septum_r_vs_tensor": corr(sept_proxy, sept_tensor if len(available) == len(rows) else [float(r["Septum_W_total_kPa"]) for r in available]),
                    "septum_mean_abs_ratio_error": sept_raw,
                    "septum_mean_abs_log_ratio_error": sept_log,
                    "septum_log_n": sept_log_n,
                    "septum_proxy_mean_kPa": float(np.mean(sept_proxy)),
                    "septum_proxy_min_kPa": float(np.min(sept_proxy)),
                    "septum_proxy_max_kPa": float(np.max(sept_proxy)),
                    "septum_ratio_pred_mean": float(np.mean(sept_ratio_pred)),
                    "septum_ratio_pred_min": float(np.min(sept_ratio_pred)),
                    "septum_ratio_pred_max": float(np.max(sept_ratio_pred)),
                }
            )
    return out


def best_by_strain(summary: list[dict[str, object]], cohort: str) -> list[dict[str, object]]:
    rows = [r for r in summary if r["cohort"] == cohort]
    out: list[dict[str, object]] = []
    for suffix, strain_label in STRAINS.items():
        matches = [r for r in rows if r["strain_suffix"] == suffix]
        if not matches:
            continue
        best_corr = max(matches, key=lambda r: -np.inf if not np.isfinite(float(r["septum_r_vs_tensor"])) else float(r["septum_r_vs_tensor"]))
        valid_mag = [r for r in matches if np.isfinite(float(r["septum_mean_abs_log_ratio_error"]))]
        best_mag = min(valid_mag, key=lambda r: float(r["septum_mean_abs_log_ratio_error"])) if valid_mag else None
        best_raw_mag = min(matches, key=lambda r: float(r["septum_mean_abs_ratio_error"]))
        out.append(
            {
                "cohort": cohort,
                "n": matches[0]["n"],
                "strain_suffix": suffix,
                "strain": strain_label,
                "freewall_adjacent_mean_abs_ratio_error": matches[0]["freewall_adjacent_mean_abs_ratio_error"],
                "freewall_ratio_r": matches[0]["freewall_ratio_r"],
                "rv_freewall_adjacent_r": matches[0]["rv_freewall_adjacent_r"],
                "best_septum_correlation_pressure": best_corr["pressure"],
                "best_septum_correlation_r": best_corr["septum_r_vs_tensor"],
                "best_septum_magnitude_pressure": best_mag["pressure"] if best_mag else "",
                "best_septum_magnitude_log_error": best_mag["septum_mean_abs_log_ratio_error"] if best_mag else float("nan"),
                "best_septum_magnitude_log_n": best_mag["septum_log_n"] if best_mag else 0,
                "best_septum_raw_magnitude_pressure": best_raw_mag["pressure"],
                "best_septum_raw_magnitude_error": best_raw_mag["septum_mean_abs_ratio_error"],
            }
        )
    return out


def write_markdown(summary: list[dict[str, object]], best: list[dict[str, object]]) -> None:
    md = OUT / "h5_strain_direction_summary.md"
    common = [r for r in best if r["cohort"] == "common13_all_directions"]
    with md.open("w") as handle:
        handle.write("# H5 Strain Direction Diagnostic\n\n")
        handle.write(
            "This is a model-side diagnostic of pressure-strain proxies using the h=5 "
            "corrected sweep. Longitudinal and fibre proxies are available for all 16 "
            "cases. Geometric radial and circumferential proxies are available for 13 "
            "cases, so cross-direction comparisons should use the common13 cohort.\n\n"
        )
        handle.write("## Best Summary On The Common 13-Case Cohort\n\n")
        handle.write(
            "| strain | free-wall LV/RV error | RV free-wall r | best septal r | best septal magnitude error | notes |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---|\n")
        for row in common:
            note = ""
            if row["strain_suffix"] == "circ":
                note = "septal signed magnitudes often have opposite sign to free-wall proxy"
            elif row["strain_suffix"] == "radial":
                note = "good free-wall ratio but poor RV trend and only modest septal ranking"
            elif row["strain_suffix"] == "ff":
                note = "model-side material direction, not routine clinical strain"
            else:
                note = "clinical pressure-strain baseline"
            if np.isfinite(float(row["best_septum_magnitude_log_error"])):
                mag_text = (
                    f"{row['best_septum_magnitude_pressure']} "
                    f"{float(row['best_septum_magnitude_log_error']):.3f}"
                )
            else:
                mag_text = (
                    "no positive signed ratios; "
                    f"raw {row['best_septum_raw_magnitude_pressure']} "
                    f"{float(row['best_septum_raw_magnitude_error']):.3f}"
                )
            handle.write(
                f"| {row['strain']} | "
                f"{float(row['freewall_adjacent_mean_abs_ratio_error']):.3f} | "
                f"{float(row['rv_freewall_adjacent_r']):+.3f} | "
                f"{row['best_septum_correlation_pressure']} {float(row['best_septum_correlation_r']):+.3f} | "
                f"{mag_text} | "
                f"{note} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "The alternative clinical-style directions do not cleanly solve the proxy "
            "problem in this diagnostic. Fibre strain improves several septal tests, "
            "which supports the claim that strain direction matters, but fibre strain "
            "is model-side rather than a routine clinical direction. Geometric radial "
            "strain preserves the free-wall LV/RV magnitude ratio well on this subset, "
            "but its RV free-wall sweep correlation is negative and its septal ranking "
            "remains modest. Geometric circumferential strain is especially problematic "
            "in the septum because the signed septal pressure-strain density often has "
            "the opposite sign from the free-wall circumferential proxy. The cautious "
            "thesis reading is therefore that richer strain measurements are worth "
            "future study, but this quick model-side test does not provide a ready "
            "replacement for the longitudinal pressure-strain proxy.\n"
        )


def make_figure(best: list[dict[str, object]]) -> None:
    common = [r for r in best if r["cohort"] == "common13_all_directions"]
    labels = [r["strain"].replace("geometric ", "") for r in common]
    fw = [float(r["freewall_adjacent_mean_abs_ratio_error"]) for r in common]
    sept = [float(r["best_septum_magnitude_log_error"]) for r in common]
    corr_vals = [float(r["best_septum_correlation_r"]) for r in common]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    x = np.arange(len(labels))
    colors = ["#4C78A8", "#E15759", "#F28E2B", "#59A14F"]
    axes[0].bar(x - 0.18, fw, width=0.36, color=colors, alpha=0.9, label="free-wall LV/RV")
    axes[0].bar(x + 0.18, sept, width=0.36, color=colors, alpha=0.45, label="best septum magnitude")
    axes[0].set_ylabel("error")
    axes[0].set_title("Magnitude preservation")
    axes[0].legend(framealpha=0.95)
    axes[1].bar(x, corr_vals, width=0.55, color=colors)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("best septal correlation")
    axes[1].set_title("Sweep ranking")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(OUT / "fig_h5_strain_direction_diagnostic.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "fig_h5_strain_direction_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = case_rows()
    common = [r for r in rows if all(r.get(f"has_{suffix}") for suffix in STRAINS)]
    if not common:
        raise SystemExit("No common rows with all strain directions")

    summary = summarize(rows, "available_per_direction")
    summary += summarize(common, "common13_all_directions")
    best = best_by_strain(summary, "available_per_direction")
    best += best_by_strain(summary, "common13_all_directions")

    write_csv(OUT / "h5_strain_direction_case_values.csv", rows)
    write_csv(OUT / "h5_strain_direction_summary.csv", summary)
    write_csv(OUT / "h5_strain_direction_best.csv", best)
    write_markdown(summary, best)
    make_figure(best)

    print(f"Loaded {len(rows)} h=5 manifest cases; {len(common)} have ll/ff/radial/circ.")
    print("\nCommon 13-case best summary:")
    for row in [r for r in best if r["cohort"] == "common13_all_directions"]:
        print(
            f"{row['strain']:<24} "
            f"FW err={float(row['freewall_adjacent_mean_abs_ratio_error']):.3f} "
            f"RV r={float(row['rv_freewall_adjacent_r']):+.3f} "
            f"best sept r={row['best_septum_correlation_pressure']} "
            f"{float(row['best_septum_correlation_r']):+.3f} "
            f"best sept mag={row['best_septum_magnitude_pressure']} "
            f"{float(row['best_septum_magnitude_log_error']):.3f} "
            f"(raw {row['best_septum_raw_magnitude_pressure']} "
            f"{float(row['best_septum_raw_magnitude_error']):.3f})"
        )
    print(f"\nWrote {OUT / 'h5_strain_direction_summary.md'}")


if __name__ == "__main__":
    main()
