"""Tests for sim_params.material_params_from_sim_params.

Needs `pulse`, so unlike the other unit tests it is skipped rather than failed
when the FEniCSx environment is not active.

    python3 tests/test_sim_params.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pulse  # noqa: F401
except ImportError:
    print("SKIP test_sim_params.py — pulse not importable (activate the RV env)")
    raise SystemExit(0)

from sim_params import material_params_from_sim_params, _scalar_from_entry

passed = failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 — a test runner reports, it does not raise
        failed += 1
        print(f"FAIL {name}: {exc}")
    else:
        passed += 1
        print(f"PASS {name}")


def test_scalar_entry():
    sp = {"material_params": {"a": {"value": 2.28, "unit": "kPa"}}}
    out = material_params_from_sim_params(sp)
    assert float(out["a"].value) == 2.28
    # The unit must survive as kPa. pulse.Variable.unit reports the SI
    # decomposition ("kilogram / meter / second ** 2"); only original_unit
    # preserves what was written. Rebuilding from .unit would make the
    # material 1000x too soft — the serialisation bug of 2026-03-08.
    assert str(out["a"].original_unit) == "kilopascal"
    assert str(out["a"].unit) != str(out["a"].original_unit)


def test_uniform_function_uses_local_mean():
    entry = {"kind": "Function", "local_min": 5.0, "local_max": 5.0,
             "local_mean": 5.0, "unit": "kPa"}
    assert _scalar_from_entry("a", entry) == 5.0


def test_non_uniform_function_still_returns_mean():
    # Approximate but deliberate: it warns and proceeds rather than failing,
    # because strain metrics are kinematic and unaffected by the moduli.
    entry = {"kind": "Function", "local_min": 1.0, "local_max": 9.0,
             "local_mean": 5.0, "unit": "kPa"}
    assert _scalar_from_entry("a", entry) == 5.0


def test_function_without_mean_falls_back_to_min():
    entry = {"kind": "Function", "local_min": 3.0, "local_max": 3.0, "unit": "kPa"}
    assert _scalar_from_entry("a", entry) == 3.0


def test_unusable_entry_raises():
    try:
        _scalar_from_entry("a", {"unit": "kPa"})
    except KeyError:
        return
    raise AssertionError("expected KeyError for an entry with no value or field stats")


for _name, _fn in list(globals().items()):
    if _name.startswith("test_"):
        check(_name, _fn)

print(f"\n{passed}/{passed + failed} passed")
sys.exit(1 if failed else 0)
