import importlib, os
from pathlib import Path

def test_results_root_defaults_to_repo_results(monkeypatch):
    monkeypatch.delenv("CARDIAC_RESULTS_ROOT", raising=False)
    import paths; importlib.reload(paths)
    assert paths.results_root() == paths.REPO_ROOT / "results"

def test_results_root_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CARDIAC_RESULTS_ROOT", str(tmp_path))
    import paths; importlib.reload(paths)
    assert paths.results_root() == tmp_path

def test_repo_root_is_this_repo():
    import paths; importlib.reload(paths)
    assert (paths.REPO_ROOT / "complete_cycle.py").exists()
