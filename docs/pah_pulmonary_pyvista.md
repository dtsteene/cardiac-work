# PAH pulmonary sweep — PyVista / ParaView exports

Three products, all on the one shared mesh (8070 cells; `region_tag`: 1=LV, 2=RV, 3=Septum).
Two density fields are carried everywhere and kept as **separate cell arrays** so each can be
colour-scaled independently (pressure-strain is ~7-10× smaller than stress-strain, so a shared
scale would mute it):

| field | meaning |
|-------|---------|
| `w_total_density_Pa` (ED export) / `cum_work_density_Pa` (beat) | true **stress-strain** work density ∮S:dE |
| `proxy_combined_ll_density_Pa` (ED/interp) / `cum_ps_density_Pa` (beat) | region-appropriate **pressure-strain** proxy: RV→P_RV, LV+septum→P_LV |

## 1. Through-beat animation (work building up over one heartbeat)
`pah_pulmonary_beat/<bundle>/<case>/beat.pvd` — case0_rv25 (low) and case7_rv95 (high), per bundle.
PVD timestep = beat phase ∈ [0,1]. Watch where work concentrates as the beat proceeds.

## 2. ED-static sweep (one frame per case, at end-diastole of the last beat)
`pah_pulmonary_ed/<bundle>/sweep.pvd` — 8 cases, PVD timestep = RV systolic mmHg.
`global_ranges.json` has per-field min/max across the bundle.

## 3. Interpolated severity sweep (smooth D-sign animation)
`pah_pulmonary_sweep_interp/<bundle>/sweep_interp.pvd` — 85 frames lerped between the 8 ED cases
(geometry **and** fields). Shows the septum bulging toward the LV and the work distribution
shifting toward the RV. `clim.json` gives **separate** global ranges for ss and ps.

---

## Scaling in a PyVista notebook (separate ss / ps scales)

```python
import json, pyvista as pv
bundle = "no_frank_starling"
root = f"paraview_exports/pah_pulmonary_sweep_interp/{bundle}"
reader = pv.get_reader(f"{root}/sweep_interp.pvd")
clim = json.load(open(f"{root}/clim.json"))

# robust GLOBAL scale across the whole sweep, one per field (use ["min"],["max"] for full range)
clim_ss = (0, clim["ss"]["p98"])   # stress-strain
clim_ps = (0, clim["ps"]["p98"])   # pressure-strain — its OWN, smaller scale

reader.set_active_time_value(reader.time_values[-1])   # most severe
g = reader.read()[0]
import numpy as np
g.cell_data["ss_abs"] = np.abs(g.cell_data["w_total_density_Pa"])
g.cell_data["ps_abs"] = np.abs(g.cell_data["proxy_combined_ll_density_Pa"])

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0); pl.add_mesh(g.copy(), scalars="ss_abs", clim=clim_ss, cmap="inferno")
pl.subplot(0, 1); pl.add_mesh(g.copy(), scalars="ps_abs", clim=clim_ps, cmap="inferno")
pl.link_views(); pl.show()
```

- **Per-case (individual) scaling** instead of global: drop `clim=...` and PyVista autoscales each
  frame — good for seeing within-case distribution, bad for comparing magnitudes across severity.
- **Global scaling** (above): fixed `clim` from `clim.json` — comparable across the sweep.
- Always use a **separate** `clim` for ss and ps; never put pressure-strain on the stress-strain scale.
- Clip to one wall with `g.threshold(value=0.5, scalars="region_tag")` after setting
  `region_tag` ranges (LV=1, RV=2, Septum=3), e.g. septum only: `g.threshold([2.5, 3.5], scalars="region_tag")`.
