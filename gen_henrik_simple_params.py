#!/usr/bin/env python3
"""
gen_henrik_simple_params.py — produce circulation parameter JSONs for the
"Henrik simple" approach: take Regazzoni 2020 defaults and scale only the
pulmonary arterial Windkessel (C_AR / R_AR) by a single factor.

No Optuna tuning. No chamber elastance scaling. This is the test the
e-mail thread agreed on (Espen rejected the EA-scaling variant as
"fudging"; this version is just the C_AR/R_AR change).

For each factor, runs the standalone 0D for 50 beats to settle on a
steady state, then saves the parameter set + final state in the same
JSON layout the FEM driver expects, into data/henrik_simple_pul/.
"""
import json
from pathlib import Path
import numpy as np

from circulation.regazzoni2020 import Regazzoni2020

OUT = Path("/global/D1/homes/dtsteene/cardiac-work/data/henrik_simple_pul")
OUT.mkdir(parents=True, exist_ok=True)

FACTORS = [1.00, 0.75, 0.50, 0.25, 0.10]   # Henrik's set
NBEATS_SETTLE = 50
DT = 1e-3

def to_plain(v):
    """Strip pint Quantity, leave bare floats / dicts intact."""
    if hasattr(v, "magnitude"):
        return float(v.magnitude)
    if isinstance(v, dict):
        return {k: to_plain(x) for k, x in v.items()}
    return v

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Defaults, with units stripped so the FEM driver can json.load() and pass
# straight back into Regazzoni2020.
defaults = to_plain(Regazzoni2020.default_parameters())
C_AR_default = defaults["circulation"]["PUL"]["C_AR"]
R_AR_default = defaults["circulation"]["PUL"]["R_AR"]

print(f"Default PUL C_AR = {C_AR_default}, R_AR = {R_AR_default}")
print(f"Generating {len(FACTORS)} cases × scale (C_AR × f, R_AR / f)")
print()

for f in FACTORS:
    case = f"pul_f{int(round(f*100)):03d}"   # pul_f100, pul_f075, pul_f050, pul_f025, pul_f010
    params = json.loads(json.dumps(defaults))   # deep copy via JSON round-trip
    params["circulation"]["PUL"]["C_AR"] = C_AR_default * f
    params["circulation"]["PUL"]["R_AR"] = R_AR_default / f

    # Standalone 0D run to converge
    model = Regazzoni2020(parameters=params, add_units=False, verbose=False)
    history = model.solve(num_beats=NBEATS_SETTLE, dt=DT)
    state = dict(zip(model.state_names(), model.state))

    # Last-beat hemodynamic readings for sanity print
    samples = int((1.0 / params["HR"]) / DT)
    sl = slice(-samples, None)
    p_LV_max = float(np.max(history["p_LV"][sl]))
    p_LV_min = float(np.min(history["p_LV"][sl]))
    p_RV_max = float(np.max(history["p_RV"][sl]))
    p_RV_min = float(np.min(history["p_RV"][sl]))
    V_LV_max = float(np.max(history["V_LV"][sl]))
    V_LV_min = float(np.min(history["V_LV"][sl]))
    V_RV_max = float(np.max(history["V_RV"][sl]))
    V_RV_min = float(np.min(history["V_RV"][sl]))

    out_json = OUT / f"henrik_simple_pul_{case}.json"
    save = {
        "description": ("Henrik simple — Regazzoni defaults with "
                        f"PUL C_AR scaled ×{f}, R_AR scaled ×{1/f:.3f}. "
                        "No Optuna, no chamber EA scaling."),
        "factor_pul": f,
        "case": case,
        "parameters": params,
        "initial_state": state,
        "metrics_achieved_0D": {
            "LV_ESP_mmHg": round(p_LV_max, 1),
            "LV_EDP_mmHg": round(p_LV_min, 1),
            "RV_ESP_mmHg": round(p_RV_max, 1),
            "RV_EDP_mmHg": round(p_RV_min, 1),
            "LV_EDV_mL":   round(V_LV_max, 1),
            "LV_ESV_mL":   round(V_LV_min, 1),
            "RV_EDV_mL":   round(V_RV_max, 1),
            "RV_ESV_mL":   round(V_RV_min, 1),
            "LV_SV_mL":    round(V_LV_max - V_LV_min, 1),
            "LV_EF_pct":   round((V_LV_max - V_LV_min) / V_LV_max * 100, 1),
            "RV_EF_pct":   round((V_RV_max - V_RV_min) / V_RV_max * 100, 1),
        },
    }
    with open(out_json, "w") as f_out:
        json.dump(save, f_out, cls=_NpEncoder, indent=2)
    m = save["metrics_achieved_0D"]
    print(f"{case}: C_AR={params['circulation']['PUL']['C_AR']:.3f} "
          f"R_AR={params['circulation']['PUL']['R_AR']:.3f} | "
          f"0D LV {m['LV_ESP_mmHg']:>5.1f}/{m['LV_EDP_mmHg']:>4.1f} mmHg, "
          f"RV {m['RV_ESP_mmHg']:>5.1f}/{m['RV_EDP_mmHg']:>4.1f} mmHg | "
          f"LV_EDV {m['LV_EDV_mL']:.1f}, RV_EDV {m['RV_EDV_mL']:.1f} | "
          f"EF {m['LV_EF_pct']:.0f}%/{m['RV_EF_pct']:.0f}% | "
          f"saved {out_json.name}")

print()
print(f"All JSONs in {OUT}/")
