import sys, importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pah_pulmonary_batch"))
import numpy as np
import make_pah_handover as mph

def test_unloaded_frame_is_raw():
    E = np.array([0.10, 0.06, 0.02, 0.07])
    np.testing.assert_allclose(mph.frame_strain(E, "unloaded"), E)

def test_clinical_frame_zeros_at_ed_max():
    E = np.array([0.10, 0.06, 0.02, 0.07])   # ED = most stretched = 0.10
    out = mph.frame_strain(E, "clinical")
    assert out.max() == 0.0
    np.testing.assert_allclose(out, E - 0.10)

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
