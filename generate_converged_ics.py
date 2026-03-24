"""
Generate converged 0D initial conditions for circulation parameter files.

Runs the standalone Regazzoni2020 0D model for many beats from default ICs
until the state converges, then saves the converged state as `initial_state`
in the JSON files. This dramatically reduces the number of coupled 3D-0D
beats needed to reach steady state.

Usage:
    python generate_converged_ics.py data/ukb_circ/optimized_regazzoni_ukb_*.json
    python generate_converged_ics.py data/ukb_circ/  # all JSON files in dir
"""

import numpy as np
import json
import sys
from pathlib import Path

import circulation
from circulation.regazzoni2020 import Regazzoni2020

# Convergence criteria
MAX_BEATS = 200
RTOL = 1e-5  # relative tolerance for state convergence
VERBOSE = True


def get_updated_parameters_from_json(json_path, bpm=75):
    """Load and update parameters consistently with complete_cycle.py logic."""
    with open(json_path) as f:
        data = json.load(f)

    json_params = data.get("parameters", data)
    params = Regazzoni2020.default_parameters()

    # Deep update from JSON
    def deep_update(base, override):
        for k, v in override.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                deep_update(base[k], v)
            else:
                base[k] = v

    deep_update(params, json_params)

    # Match the timing logic from complete_cycle.py
    HR_HZ = bpm / 60.0
    RR_INTERVAL = 60.0 / bpm
    factor = RR_INTERVAL / 0.8

    TC_ACTIVATION = 0.25
    TR_ACTIVATION = 0.40

    ref_LV_tC = 0.1 * factor
    time_shift = -ref_LV_tC

    for chamber in ["LA", "RA", "LV", "RV"]:
        original_tC = params["chambers"][chamber]["tC"]
        if hasattr(original_tC, "magnitude"):
            original_tC = original_tC.magnitude
        scaled_tC = original_tC * factor
        new_tC = scaled_tC + time_shift
        params["chambers"][chamber]["tC"] = new_tC * circulation.units.ureg("s")
        params["chambers"][chamber]["TC"] *= factor
        params["chambers"][chamber]["TR"] *= factor

    for chamber in ["LV", "RV"]:
        params["chambers"][chamber]["tC"] = 0.0 * circulation.units.ureg("s")
        params["chambers"][chamber]["TC"] = TC_ACTIVATION * circulation.units.ureg("s")
        params["chambers"][chamber]["TR"] = TR_ACTIVATION * circulation.units.ureg("s")

    params["HR"] = circulation.units.ureg(f"{HR_HZ} Hz")
    return params


def run_to_convergence(params, max_beats=MAX_BEATS, rtol=RTOL):
    """Run 0D model until state converges. Returns converged state dict."""
    model = Regazzoni2020(parameters=params)

    state_names = model.state_names()
    prev_state = np.copy(model.state)

    # Run in chunks of beats and check convergence
    chunk_size = 5
    total_beats = 0

    while total_beats < max_beats:
        model.solve(num_beats=chunk_size, initial_state=None)
        total_beats += chunk_size

        curr_state = np.copy(model.state)

        # Check relative convergence
        rel_change = np.abs(curr_state - prev_state) / (np.abs(prev_state) + 1e-15)
        max_change = np.max(rel_change)

        if VERBOSE:
            worst_idx = np.argmax(rel_change)
            print(f"  Beat {total_beats}: max_rel_change = {max_change:.2e} "
                  f"(worst: {state_names[worst_idx]} = {curr_state[worst_idx]:.4f})")

        if max_change < rtol:
            print(f"  Converged after {total_beats} beats (rtol={rtol})")
            break

        prev_state = curr_state

    if total_beats >= max_beats:
        print(f"  WARNING: Did not converge after {max_beats} beats (max_change={max_change:.2e})")

    # Return converged state as dict
    converged = dict(zip(state_names, model.state))
    return converged, total_beats


def update_json_file(json_path, converged_state):
    """Update JSON file with converged initial state."""
    with open(json_path) as f:
        data = json.load(f)

    # Store old IC for comparison
    old_ic = data.get("initial_state", {})

    # Update initial_state
    data["initial_state"] = converged_state

    # Add metadata
    data["initial_state_converged"] = True
    data["initial_state_method"] = "standalone_0D_convergence"

    # Write back
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return old_ic


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("optimized_regazzoni_*.json")))
        elif p.suffix == ".json":
            paths.append(p)

    if not paths:
        print("No JSON files found")
        sys.exit(1)

    print(f"Processing {len(paths)} circulation parameter files...\n")

    for json_path in paths:
        name = json_path.stem.replace("optimized_regazzoni_ukb_", "")
        print(f"=== {name} ({json_path.name}) ===")

        # Load and configure parameters
        params = get_updated_parameters_from_json(json_path)

        # Run to convergence
        converged_state, n_beats = run_to_convergence(params)

        # Print converged state
        print(f"  Converged state:")
        for k, v in converged_state.items():
            print(f"    {k}: {v:.4f}")

        # Update JSON
        old_ic = update_json_file(json_path, converged_state)

        # Show changes
        print(f"  IC changes:")
        for k in converged_state:
            old_v = old_ic.get(k, "N/A")
            new_v = converged_state[k]
            if isinstance(old_v, (int, float)):
                delta = abs(new_v - old_v) / abs(old_v) * 100 if old_v != 0 else 0
                print(f"    {k}: {old_v:.2f} -> {new_v:.2f} ({delta:.1f}% change)")
        print()


if __name__ == "__main__":
    main()
