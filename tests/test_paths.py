"""Unit tests for paths — pure stdlib, safe to run on the login node.

    python3 tests/test_paths.py
"""

import importlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths


def _clear_env():
    os.environ.pop("CARDIAC_RESULTS_ROOT", None)


def test_results_root_defaults_to_repo_results():
    saved = os.environ.get("CARDIAC_RESULTS_ROOT")
    try:
        _clear_env()
        importlib.reload(paths)
        assert paths.results_root() == paths.REPO_ROOT / "results", paths.results_root()
    finally:
        if saved is None:
            _clear_env()
        else:
            os.environ["CARDIAC_RESULTS_ROOT"] = saved
        importlib.reload(paths)


def test_results_root_honours_env():
    saved = os.environ.get("CARDIAC_RESULTS_ROOT")
    tmp = Path(tempfile.mkdtemp()).resolve()
    try:
        os.environ["CARDIAC_RESULTS_ROOT"] = str(tmp)
        importlib.reload(paths)
        assert paths.results_root() == tmp, paths.results_root()
    finally:
        if saved is None:
            _clear_env()
        else:
            os.environ["CARDIAC_RESULTS_ROOT"] = saved
        importlib.reload(paths)


def test_repo_root_is_this_repo():
    importlib.reload(paths)
    assert (paths.REPO_ROOT / "complete_cycle.py").exists(), paths.REPO_ROOT


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
