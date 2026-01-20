import numpy as np


def compute_proxy_sequence_prefers_current_state(volumes, pressures, current_states=None):
    """Mimic fixed proxy calculation that ALWAYS prefers current_state over history.

    volumes, pressures: list of tuples (V_LV, V_RV) and (p_LV, p_RV)
    current_states: optional list of dicts; if provided, these override volumes/pressures
    Returns list of proxy work (J) for LV, RV.
    """
    V_LV_prev = None
    V_RV_prev = None
    mmHg_mL_to_J = 1.33322e-4
    out_lv = []
    out_rv = []
    
    for i, ((V_LV, V_RV), (p_LV, p_RV)) in enumerate(zip(volumes, pressures)):
        # If current_state provided, prefer it (ground truth)
        if current_states and i < len(current_states):
            cs = current_states[i]
            V_LV = cs.get("V_LV", V_LV)
            V_RV = cs.get("V_RV", V_RV)
            p_LV = cs.get("p_LV", p_LV)
            p_RV = cs.get("p_RV", p_RV)
        
        if V_LV_prev is None:
            # First timestep
            V_LV_prev, V_RV_prev = V_LV, V_RV
            out_lv.append(0.0)
            out_rv.append(0.0)
            continue
        
        dV_LV = V_LV - V_LV_prev
        dV_RV = V_RV - V_RV_prev
        out_lv.append(p_LV * dV_LV * mmHg_mL_to_J)
        out_rv.append(p_RV * dV_RV * mmHg_mL_to_J)
        V_LV_prev, V_RV_prev = V_LV, V_RV
    
    return np.array(out_lv), np.array(out_rv)


def test_proxy_always_prefers_current_state():
    """Test that proxy calc uses current_state even when history is long."""
    # History (0D circulation, wrong volumes)
    history_volumes = [(0.0, 0.0)] * 5  # All zeros from 0D history
    
    # Pressures
    pressures = [
        (10.0, 5.0),
        (50.0, 15.0),
        (100.0, 20.0),
        (80.0, 18.0),
        (40.0, 10.0),
    ]
    
    # Current states (FEM cavity volumes - the truth!)
    current_states = [
        {"V_LV": 100.0, "V_RV": 150.0, "p_LV": 10.0, "p_RV": 5.0},
        {"V_LV": 105.0, "V_RV": 155.0, "p_LV": 50.0, "p_RV": 15.0},
        {"V_LV": 110.0, "V_RV": 160.0, "p_LV": 100.0, "p_RV": 20.0},
        {"V_LV": 105.0, "V_RV": 155.0, "p_LV": 80.0, "p_RV": 18.0},
        {"V_LV": 100.0, "V_RV": 150.0, "p_LV": 40.0, "p_RV": 10.0},
    ]
    
    lv, rv = compute_proxy_sequence_prefers_current_state(history_volumes, pressures, current_states)
    
    # First step should be zero
    assert lv[0] == 0.0 and rv[0] == 0.0
    
    # All subsequent steps should be nonzero (because current_state has nonzero dV)
    assert np.count_nonzero(lv[1:]) == len(lv) - 1
    assert np.count_nonzero(rv[1:]) == len(rv) - 1
    
    # Verify representative values
    # Step 1: dV_LV = 105-100=5, p_LV=50
    expected_1 = 50.0 * 5.0 * 1.33322e-4
    assert np.isclose(lv[1], expected_1)


def test_proxy_nonzero_with_stateful_volumes():
    volumes = [
        (100.0, 150.0),  # init
        (105.0, 155.0),
        (110.0, 160.0),
        (105.0, 155.0),
    ]
    pressures = [
        (10.0, 5.0),
        (50.0, 15.0),
        (100.0, 20.0),
        (80.0, 18.0),
    ]
    lv, rv = compute_proxy_sequence_prefers_current_state(volumes, pressures)
    assert lv[0] == 0.0 and rv[0] == 0.0
    assert np.count_nonzero(lv[1:]) == 3
    assert np.count_nonzero(rv[1:]) == 3
    # check a representative value
    assert np.isclose(lv[1], 50.0 * 5.0 * 1.33322e-4)

