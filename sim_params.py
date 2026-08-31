"""Reading `simulation_params.json` back into the objects a replay needs.

`complete_cycle.py` writes the forward simulation's settings to
`simulation_params.json`; `postprocess_metrics.py` and `compute_per_cell.py`
both rebuild the model from it. This module holds the parts of that
reconstruction that are pure deserialisation, so the two replay paths cannot
drift apart.

Kept deliberately small: it depends only on `pulse`, not on a mesh or a
checkpoint, so it is importable and testable without FEniCSx state.
"""

from __future__ import annotations

import logging

import pulse

logger = logging.getLogger(__name__)

# Relative width below which a serialised field counts as constant.
_UNIFORM_RTOL = 1e-9


def material_params_from_sim_params(sim_params: dict) -> dict[str, pulse.Variable]:
    """Rebuild the Holzapfel-Ogden parameters as `pulse.Variable` objects.

    Each entry is normally ``{"value": float, "unit": str}``. Regional material
    scaling (``LV_/RV_/SEPTUM_MATERIAL_SCALE != 1``) instead serialises the
    parameter as a spatially-varying Function, which has no scalar ``value`` —
    only summary statistics. For whole-heart uniform scaling that field is
    constant, so ``local_mean`` is the exact scalar; for genuinely non-uniform
    scaling it is an approximation and we say so.

    Units matter: ``str(v.unit)`` on a `pulse.Variable` returns the SI
    decomposition ("kg/m/s²"), not the original unit ("kPa"), so
    `complete_cycle.py` records ``original_unit``. Reconstructing from the
    decomposed unit would make the material 1000x too soft.

    Raises KeyError if an entry has neither a scalar value nor field statistics.
    """
    material_params = {}
    for name, entry in sim_params["material_params"].items():
        material_params[name] = pulse.Variable(_scalar_from_entry(name, entry),
                                               entry["unit"])
    return material_params


def _scalar_from_entry(name: str, entry: dict) -> float:
    if "value" in entry:
        return entry["value"]

    if entry.get("kind") == "Function":
        low, high = entry.get("local_min"), entry.get("local_max")
        if low is not None and high is not None and _is_non_uniform(low, high):
            logger.warning(
                "material_params['%s'] is a non-uniform field; replay uses "
                "local_mean=%s (approximate).", name, entry.get("local_mean"))
        return entry.get("local_mean", entry.get("local_min"))

    raise KeyError(
        f"material_params['{name}'] has neither 'value' nor field statistics: {entry}"
    )


def _is_non_uniform(low: float, high: float) -> bool:
    return abs(high - low) > _UNIFORM_RTOL * (abs(high) + 1e-12)
