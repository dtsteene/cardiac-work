#!/usr/bin/env python3
"""Work-component breakdown for the corrected n=16 pressure spectrum.

The goal is descriptive rather than inferential: show how much of the full
tensor work density comes from the fibre, sheet, sheet-normal, and cross terms,
and how these terms redistribute as RV pressure is increased.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/dtsteene/D1/cardiac-work")
OUT = ROOT / "results" / "analysis" / "work_components_n16"
KPA = 1e-3

CASES = [
    ("sPAP22", 1047450), ("sPAP25", 1048194),
    ("sPAP30", 1047451), ("sPAP35", 1048195),
    ("sPAP45", 1047452), ("sPAP50", 1048196),
    ("sPAP55", 1047453), ("sPAP60", 1048197),
    ("sPAP65", 1047454), ("sPAP70", 1048198),
    ("sPAP75", 1047455), ("sPAP80", 1048199),
    ("sPAP85", 1047456), ("sPAP87", 1048200),
    ("sPAP92", 1048201), ("sPAP95", 1047457),
]

COMPONENTS = [
    ("w_ff", "fibre"),
    ("w_ss", "sheet"),
    ("w_nn", "normal"),
    ("w_cross", "cross"),
]

REGIONS = [
    ("LV", "LV free wall"),
    ("RV", "RV free wall"),
    ("Septum", "septum"),
]


def find_run(job_id: int) -> Path:
    roots = [
        ROOT / "results" / "sims" / "2026-04-23",
        ROOT / "results" / "sims" / "2026-04-24",
    ]
    for root in roots:
        run = root / f"UKB_6beats_run_{job_id}"
        if run.exists():
            return run
    raise FileNotFoundError(f"Could not find run for job {job_id}")


def region_masks(pc: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    return {
        "LV": pc["region_tags"] == 1,
        "RV": pc["region_tags"] == 2,
        "Septum": pc["is_geometric_septum"].astype(bool),
    }


def density(pc: np.lib.npyio.NpzFile, mask: np.ndarray, key: str) -> float:
    volume = float(pc["cell_volumes"][mask].sum())
    return float(-pc[key][mask].sum() / volume * KPA)


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case, job_id in CASES:
        run = find_run(job_id)
        pc = np.load(run / "per_cell_data.npz", allow_pickle=True)
        pressure = np.load(run / "solver" / "solver_cavity_pressure_mmHg.npy")
        beat_len = pressure.shape[0] // 6
        last = pressure[5 * beat_len:]

        masks = region_masks(pc)
        for region, _ in REGIONS:
            mask = masks[region]
            row: dict[str, object] = {
                "case": case,
                "job_id": job_id,
                "region": region,
                "rvsp_mmHg": float(last[:, 1].max()),
                "lvsp_mmHg": float(last[:, 0].max()),
                "volume_mL": float(pc["cell_volumes"][mask].sum() * 1e6),
                "W_total_kPa": density(pc, mask, "w_total"),
            }
            for key, label in COMPONENTS:
                row[f"W_{label}_kPa"] = density(pc, mask, key)
            row["W_component_sum_kPa"] = sum(float(row[f"W_{label}_kPa"]) for _, label in COMPONENTS)
            row["closure_error_kPa"] = float(row["W_component_sum_kPa"]) - float(row["W_total_kPa"])
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for region, _ in REGIONS:
        rr = [row for row in rows if row["region"] == region]
        total = np.array([float(row["W_total_kPa"]) for row in rr])
        abs_den = np.array([
            [abs(float(row[f"W_{label}_kPa"])) for _, label in COMPONENTS]
            for row in rr
        ])
        abs_share = abs_den / abs_den.sum(axis=1, keepdims=True)
        for idx, (_, label) in enumerate(COMPONENTS):
            vals = np.array([float(row[f"W_{label}_kPa"]) for row in rr])
            signed_fraction = vals / total
            out.append({
                "region": region,
                "component": label,
                "mean_kPa": float(vals.mean()),
                "min_kPa": float(vals.min()),
                "max_kPa": float(vals.max()),
                "mean_signed_fraction_of_total": float(signed_fraction.mean()),
                "min_signed_fraction_of_total": float(signed_fraction.min()),
                "max_signed_fraction_of_total": float(signed_fraction.max()),
                "mean_absolute_share": float(abs_share[:, idx].mean()),
                "min_absolute_share": float(abs_share[:, idx].min()),
                "max_absolute_share": float(abs_share[:, idx].max()),
            })
    return out


def make_component_figure(rows: list[dict[str, object]]) -> None:
    colors = {
        "total": "#111111",
        "fibre": "#238b45",
        "sheet": "#b2182b",
        "normal": "#2166ac",
        "cross": "#7a7a7a",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6), sharex=True)
    for ax, (region, title) in zip(axes, REGIONS):
        rr = sorted([row for row in rows if row["region"] == region], key=lambda row: float(row["rvsp_mmHg"]))
        x = np.array([float(row["rvsp_mmHg"]) for row in rr])
        ax.axhline(0, color="#c7c7c7", linewidth=0.8)
        ax.plot(x, [float(row["W_total_kPa"]) for row in rr],
                color=colors["total"], linewidth=2.5, marker="o", label="total")
        for _, label in COMPONENTS:
            ax.plot(x, [float(row[f"W_{label}_kPa"]) for row in rr],
                    color=colors[label], linewidth=1.8, marker="o", markersize=3.4, label=label)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("achieved RV systolic pressure (mmHg)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("net work density per cycle (kPa)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.suptitle("Tensor-work component breakdown across the pressure-overload spectrum",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_work_components_vs_rvsp.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "fig_work_components_vs_rvsp.pdf", bbox_inches="tight")
    plt.close(fig)


def make_share_figure(summaries: list[dict[str, object]]) -> None:
    colors = {
        "fibre": "#238b45",
        "sheet": "#b2182b",
        "normal": "#2166ac",
        "cross": "#7a7a7a",
    }
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    x = np.arange(len(REGIONS))
    bottom = np.zeros(len(REGIONS))
    for _, label in COMPONENTS:
        vals = []
        for region, _ in REGIONS:
            row = next(r for r in summaries if r["region"] == region and r["component"] == label)
            vals.append(float(row["mean_absolute_share"]))
        ax.bar(x, vals, bottom=bottom, color=colors[label], label=label)
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([title for _, title in REGIONS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mean share of |component work|")
    ax.set_title("Relative size of tensor-work components", fontsize=12, fontweight="bold")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_work_component_absolute_shares.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "fig_work_component_absolute_shares.pdf", bbox_inches="tight")
    plt.close(fig)


def write_notes(rows: list[dict[str, object]], summaries: list[dict[str, object]]) -> None:
    lines = [
        "# Work Component Breakdown, Corrected n=16",
        "",
        "Work is reported as positive net work density, in kPa (= mJ/mL).",
        "Components are the orthonormal fibre-sheet-normal decomposition of the full tensor contraction.",
        "",
        "## Mean Signed Fractions",
        "",
        "| region | fibre | sheet | normal | cross |",
        "|---|---:|---:|---:|---:|",
    ]
    for region, _ in REGIONS:
        vals = []
        for _, label in COMPONENTS:
            row = next(r for r in summaries if r["region"] == region and r["component"] == label)
            vals.append(f"{100 * float(row['mean_signed_fraction_of_total']):.1f}%")
        lines.append(f"| {region} | {' | '.join(vals)} |")

    lines += [
        "",
        "## Low-To-High Pressure Change",
        "",
        "| region | total low | total high | change | dominant positive change |",
        "|---|---:|---:|---:|---|",
    ]
    for region, _ in REGIONS:
        rr = [row for row in rows if row["region"] == region]
        low = rr[0]
        high = rr[-1]
        changes = {
            label: float(high[f"W_{label}_kPa"]) - float(low[f"W_{label}_kPa"])
            for _, label in COMPONENTS
        }
        dominant = max(changes.items(), key=lambda item: abs(item[1]))
        change = float(high["W_total_kPa"]) - float(low["W_total_kPa"])
        lines.append(
            f"| {region} | {float(low['W_total_kPa']):.2f} | "
            f"{float(high['W_total_kPa']):.2f} | {change:+.2f} | "
            f"{dominant[0]} ({dominant[1]:+.2f}) |"
        )

    lines += [
        "",
        "## Suggested Thesis Use",
        "",
        "- Use `fig_work_components_vs_rvsp` in the results chapter before the proxy-correlation plots.",
        "- Use `fig_work_component_absolute_shares` only if the reader needs an intuitive size comparison; otherwise keep it as backup.",
        "- Main text point: fibre work dominates the free-wall trend, but the septum has a much larger non-fibre contribution, especially normal and cross terms.",
    ]
    (OUT / "work_component_notes.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    summaries = summary_rows(rows)
    write_csv(OUT / "work_component_case_values.csv", rows)
    write_csv(OUT / "work_component_summary.csv", summaries)
    make_component_figure(rows)
    make_share_figure(summaries)
    write_notes(rows, summaries)
    print(f"Wrote {OUT / 'work_component_case_values.csv'}")
    print(f"Wrote {OUT / 'work_component_summary.csv'}")
    print(f"Wrote {OUT / 'fig_work_components_vs_rvsp.png'}")
    print(f"Wrote {OUT / 'fig_work_component_absolute_shares.png'}")
    print(f"Wrote {OUT / 'work_component_notes.md'}")


if __name__ == "__main__":
    main()
