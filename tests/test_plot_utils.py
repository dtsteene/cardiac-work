"""Tests for plot_utils.load_metrics directory-layout handling.

run_postprocessing.py hands plot_loops.py and eval_proxies.py a PER-BEAT
directory (analysis/last_beat), where the metrics file sits flat. The run root
uses a metrics/ subdirectory instead. load_metrics must handle both, or the
figure and proxy steps fail on every run.

    python3 tests/test_plot_utils.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_utils import load_metrics  # noqa: E402

SENTINEL = {"work_ff_LV": np.array([1.0, 2.0, 3.0])}


def _write(directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / name, SENTINEL, allow_pickle=True)


def test_reads_metrics_subdir():
    """The run root layout: <folder>/metrics/metrics_downsample_1.npy."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write(root / "metrics", "metrics_downsample_1.npy")
        assert "work_ff_LV" in load_metrics(root)


def test_reads_flat_per_beat_dir():
    """The per-beat layout run_postprocessing actually passes: file sits flat.

    Regression guard for abc5a75, which dropped this fallback and made
    plot_loops.py and eval_proxies.py fail on every run.
    """
    with tempfile.TemporaryDirectory() as td:
        beat_dir = Path(td) / "analysis" / "last_beat"
        _write(beat_dir, "metrics_downsample_1.npy")
        assert "work_ff_LV" in load_metrics(beat_dir)


def test_prefers_finest_downsampling():
    """_1 (finest) must win over _10, in either layout."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write(root / "metrics", "metrics_downsample_10.npy")
        np.save(root / "metrics" / "metrics_downsample_1.npy",
                {"marker": np.array([1.0])}, allow_pickle=True)
        assert "marker" in load_metrics(root)


def test_metrics_subdir_wins_over_flat():
    """When both exist, the canonical metrics/ layout takes precedence."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write(root / "metrics", "metrics_downsample_1.npy")
        np.save(root / "metrics_downsample_1.npy",
                {"stale": np.array([0.0])}, allow_pickle=True)
        assert "work_ff_LV" in load_metrics(root)


def test_raises_when_no_metrics_anywhere():
    """An unpostprocessed run must stop the caller, not yield an empty figure."""
    with tempfile.TemporaryDirectory() as td:
        try:
            load_metrics(Path(td))
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError")


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
