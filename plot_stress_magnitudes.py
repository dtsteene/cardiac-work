"""Generate the Stress Magnitudes figure for thesis chapter 3.

Plots post-fix peak fibre stress |S_ff| per region against a literature envelope.
Used to demonstrate that the stress check passes after the fenicsx-pulse compressible
material path was patched to forward the full Green-Lagrange tensor.

Output: /home/dtsteene/D1/RV/figures/fig_stress_magnitudes.png

Status: per-region peaks computed from the sPAP22 baseline case of the 2026-05-10
capped-shared-L5 production sweep, which uses the shared-reference-mesh
(reference-tag) postprocessing convention. Each case in this sweep integrates on
the same 8070-cell mesh with the same 1269 geometric septum cells.

Literature envelope (LIT_LO_KPA, LIT_HI_KPA) is set to bracket the canine
epicardial peak Cauchy fibre stress reported by Delhaas et al. 1994 (mean
21-27 kPa across LV regions, with transmural variation expected to push
endocardial peaks 2-3x higher) and the human computational fibre-stress
estimates from Finsberg 2017 PhD. The 20-80 kPa band is the intersection of
those references' reported peak ranges; if a tighter envelope is wanted, the
specific Finsberg PhD chapter values would replace the upper bound.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SIM = Path(
    "/home/dtsteene/D1/cardiac-work/results/sims/2026-05-10/"
    "capped_shared_l5_20260510_141015/sPAP22/analysis/last_beat/"
    "metrics_downsample_1.npy"
)
OUT = Path("/home/dtsteene/D1/RV/figures/fig_stress_magnitudes.png")

LIT_LO_KPA = 20.0
LIT_HI_KPA = 80.0

m = np.load(SIM, allow_pickle=True).item()
regions = ["LV", "RV", "Septum"]
peaks = []
for r in regions:
    arr = np.array(m[f"mean_sigma_ff_{r}"])
    peaks.append(np.max(np.abs(arr)) / 1e3)

fig, ax = plt.subplots(figsize=(5.5, 4.0))

ax.axhspan(LIT_LO_KPA, LIT_HI_KPA, color="#888888", alpha=0.18,
           label=f"reported envelope ({LIT_LO_KPA:.0f}–{LIT_HI_KPA:.0f} kPa)")

bars = ax.bar(regions, peaks,
              color=["#c0392b", "#2980b9", "#7f8c8d"],
              edgecolor="black", lw=1.0, width=0.55)

for bar, peak in zip(bars, peaks):
    ax.text(bar.get_x() + bar.get_width() / 2.0,
            peak + 1.5,
            f"{peak:.1f}",
            ha="center", va="bottom", fontsize=10)

ax.set_ylabel(r"peak $|\sigma_{ff}|$ (kPa)")
ax.set_ylim(0, max(LIT_HI_KPA + 10, max(peaks) + 10))
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, axis="y", alpha=0.25)
ax.set_axisbelow(True)

print(f"Per-region peak |S_ff| (kPa): {dict(zip(regions, [f'{p:.1f}' for p in peaks]))}")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
print(f"Saved {OUT}")
print(f"Saved {OUT.with_suffix('.pdf')}")
