"""Regenerate the chapter-5 headline figures from the canonical capped-sweep analysis.

Companion to plot_h5_thesis_updates.py - this script was added 2026-05-14 to give
the two headline figures (fig_5_0_freewall_headline.png and fig_5_0_septum_headline.png)
a discoverable, reproducible source. Earlier versions of these two PNGs had no
discoverable source script.

Data flow:
  Canonical CSV
    /home/dtsteene/D1/cardiac-work/results/analysis/
      capped_shared_l5_sweep_20260510_141015/h5_sweep_case_values.csv
  -> compute four ratios + their MAE / log-MAE (eta) across the 16 cases
  -> render two figures (freewall single panel, septum two panel)
  -> save to OUTPUT_DIR (default /tmp/headline_regen) as PNG + PDF.

Run on the login node - pandas + matplotlib only, no FEniCSx, no Slurm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CANONICAL_CSV = Path(
    "/home/dtsteene/D1/cardiac-work/results/analysis/"
    "capped_shared_l5_sweep_20260510_141015/h5_sweep_case_values.csv"
)

# 300 dpi keeps PNG dimensions >=1500px wide so they match the original
# May-11 thesis renders (freewall ~1460x1447, septum ~1548x2972).
SAVE_DPI = 300


def compute_ratios(df: pd.DataFrame) -> dict:
    """Compute the four headline ratios and their error metrics.

    The W_total_kPa and proxy *_kPa columns are already work *densities*
    (kJ/m^3 == kPa), so ratios are taken directly without dividing by volume
    again. This matches analyze_h5_sweep_core.py:200-207 / 253-269.
    """
    out: dict = {}

    # --- Freewall: LV / RV (canonical column FW_tensor_LV_RV_ratio == LV/RV) ---
    fe_fw_col = df["FW_tensor_LV_RV_ratio"].to_numpy()
    proxy_fw_col = df["FW_adjacent_ll_LV_RV_ratio"].to_numpy()
    out["fw_fe"] = fe_fw_col
    out["fw_proxy"] = proxy_fw_col
    out["fw_mae"] = float(np.mean(np.abs(proxy_fw_col - fe_fw_col)))
    # Sanity vs raw kPa columns.
    fw_recomputed = (df["LV_W_total_kPa"] / df["RV_W_total_kPa"]).to_numpy()
    out["fw_fe_recomputed_check"] = float(np.max(np.abs(fw_recomputed - fe_fw_col)))

    proxy_choices = {
        "PLV": "Septum_PLV_ll_kPa",
        "Mean": "Septum_Mean_ll_kPa",
        "Trans": "Septum_Trans_ll_kPa",
    }

    # --- Septum vs LV free-wall (per-panel ratios) ---
    sep_W = df["Septum_W_total_kPa"].to_numpy()
    lv_W = df["LV_W_total_kPa"].to_numpy()
    rv_W = df["RV_W_total_kPa"].to_numpy()
    fe_sep_lv = sep_W / lv_W
    fe_sep_rv = sep_W / rv_W

    lv_adj_proxy = df["LV_PLV_ll_kPa"].to_numpy()
    rv_adj_proxy = df["RV_PRV_ll_kPa"].to_numpy()
    sep_lv_proxy = {k: df[v].to_numpy() / lv_adj_proxy for k, v in proxy_choices.items()}
    sep_rv_proxy = {k: df[v].to_numpy() / rv_adj_proxy for k, v in proxy_choices.items()}

    out["sep_lv_fe"] = fe_sep_lv
    out["sep_rv_fe"] = fe_sep_rv
    out["sep_lv_proxy"] = sep_lv_proxy
    out["sep_rv_proxy"] = sep_rv_proxy

    # --- Septum vs free-wall mean: the canonical "ratio preservation" metric.
    # Matches analyze_h5_sweep_core.py:253-269 -> mean_abs_log_error.
    fw_mean = 0.5 * (lv_W + rv_W)
    fw_adj_mean = 0.5 * (lv_adj_proxy + rv_adj_proxy)
    fe_sep_fwmean = sep_W / fw_mean
    sep_fwmean_proxy = {
        k: df[v].to_numpy() / fw_adj_mean for k, v in proxy_choices.items()
    }
    out["sep_fwmean_fe"] = fe_sep_fwmean
    out["sep_fwmean_proxy"] = sep_fwmean_proxy

    def log_mae(proxy: np.ndarray, fe: np.ndarray) -> float:
        return float(np.mean(np.abs(np.log(np.abs(proxy / fe)))))

    out["eta_sep_lv"] = {k: log_mae(sep_lv_proxy[k], fe_sep_lv) for k in proxy_choices}
    out["eta_sep_rv"] = {k: log_mae(sep_rv_proxy[k], fe_sep_rv) for k in proxy_choices}
    out["eta_sep_fwmean"] = {
        k: log_mae(sep_fwmean_proxy[k], fe_sep_fwmean) for k in proxy_choices
    }
    out["eta_combined"] = {
        k: log_mae(
            np.concatenate([sep_lv_proxy[k], sep_rv_proxy[k]]),
            np.concatenate([fe_sep_lv, fe_sep_rv]),
        )
        for k in proxy_choices
    }
    return out


def plot_freewall(metrics: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    x = metrics["fw_fe"]
    y = metrics["fw_proxy"]

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)

    ax.plot(lim, lim, ls="--", color="lightgrey", lw=1.2, zorder=1)
    ax.scatter(
        x,
        y,
        s=80,
        color="tab:blue",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label=r"adjacent ($p_\mathrm{LV}$ / $p_\mathrm{RV}$)",
    )

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("LV / RV free-wall ratio")
    ax.set_xlabel(r"FE work-density ratio $W_\mathrm{LV}/W_\mathrm{RV}$")
    ax.set_ylabel(r"Proxy ratio $W_\mathrm{PS,LV}/W_\mathrm{PS,RV}$")

    ax.text(
        0.03,
        0.97,
        f"MAE = {metrics['fw_mae']:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
    )
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_5_0_freewall_headline.png", dpi=SAVE_DPI)
    fig.savefig(out_dir / "fig_5_0_freewall_headline.pdf")
    plt.close(fig)


def plot_septum(metrics: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))

    styles = {
        "PLV": dict(color="tab:blue", marker="o", label=r"$p_\mathrm{LV}$"),
        "Mean": dict(color="tab:green", marker="^", label="mean"),
        "Trans": dict(color="tab:red", marker="D", label="transmural"),
    }

    # Canonical "ratio preservation" metric from analyze_h5_sweep_core.py:
    # eta = mean |log(proxy_ratio / FE_ratio)| with ratios taken against
    # the LV/RV free-wall MEAN (not per side). The existing thesis figure
    # displays these combined-across-cases eta values on BOTH panels.
    eta_used = metrics["eta_sep_fwmean"]
    eta_text = (
        r"log MAE: $p_\mathrm{LV}$ = "
        + f"{eta_used['PLV']:.2f}, "
        + f"mean = {eta_used['Mean']:.2f}, "
        + f"trans = {eta_used['Trans']:.2f}"
    )

    panels = [
        ("sep_lv_fe", "sep_lv_proxy", "Septum / LV free-wall ratio",
         r"FE work-density ratio $W_\mathrm{Sep}/W_\mathrm{LV}$",
         r"Proxy ratio $W_\mathrm{PS,Sep}/W_\mathrm{PS,LV}$"),
        ("sep_rv_fe", "sep_rv_proxy", "Septum / RV free-wall ratio",
         r"FE work-density ratio $W_\mathrm{Sep}/W_\mathrm{RV}$",
         r"Proxy ratio $W_\mathrm{PS,Sep}/W_\mathrm{PS,RV}$"),
    ]

    for ax, (fe_key, proxy_key, title, xlabel, ylabel) in zip(axes, panels):
        fe = metrics[fe_key]
        proxies = metrics[proxy_key]

        all_x = [fe]
        all_y = [proxies[k] for k in styles]
        lo = min(np.min(fe), *[p.min() for p in all_y])
        hi = max(np.max(fe), *[p.max() for p in all_y])
        pad = 0.05 * (hi - lo)
        lim = (max(0.0, lo - pad), hi + pad)

        ax.plot(lim, lim, ls="--", color="lightgrey", lw=1.0, zorder=1,
                label="identity y = x")
        for key, style in styles.items():
            ax.scatter(
                fe,
                proxies[key],
                s=55,
                color=style["color"],
                marker=style["marker"],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
                label=style["label"],
            )

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(
            0.03,
            0.97,
            eta_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    # Reorder: PLV, mean, trans, identity
    order = []
    for want in (r"$p_\mathrm{LV}$", "mean", "transmural", "identity y = x"):
        for i, lab in enumerate(labels):
            if lab == want and i not in order:
                order.append(i)
                break
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "fig_5_0_septum_headline.png", dpi=SAVE_DPI,
                bbox_inches="tight")
    fig.savefig(out_dir / "fig_5_0_septum_headline.pdf",
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=CANONICAL_CSV,
        help="Canonical h5_sweep_case_values.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/headline_regen"),
        help="Output directory for regenerated figures",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv).sort_values("case").reset_index(drop=True)
    print(f"Loaded {len(df)} cases from {args.csv}")

    metrics = compute_ratios(df)

    print("\n--- Freewall (LV / RV) ---")
    print(f"  MAE                  = {metrics['fw_mae']:.4f}")
    print(f"  max |recomputed FE - canonical FE| = "
          f"{metrics['fw_fe_recomputed_check']:.2e}")

    print("\n--- Septum / LV free-wall (eta = mean |log proxy - log FE|) ---")
    for k, v in metrics["eta_sep_lv"].items():
        print(f"  {k:5s} eta = {v:.4f}")

    print("\n--- Septum / RV free-wall ---")
    for k, v in metrics["eta_sep_rv"].items():
        print(f"  {k:5s} eta = {v:.4f}")

    print("\n--- Septum vs free-wall MEAN (canonical h5_septum_ratio_preservation.csv) ---")
    for k, v in metrics["eta_sep_fwmean"].items():
        print(f"  {k:5s} eta = {v:.4f}")

    print("\n--- Combined per-panel pooled (Sep/LV + Sep/RV concatenated) ---")
    for k, v in metrics["eta_combined"].items():
        print(f"  {k:5s} eta = {v:.4f}")

    plot_freewall(metrics, args.out)
    plot_septum(metrics, args.out)
    print(f"\nWrote figures to {args.out}")


if __name__ == "__main__":
    main()
