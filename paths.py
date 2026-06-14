"""Single source of truth for the results root, so the repo is user-agnostic.

Resolution order:
  1. $CARDIAC_RESULTS_ROOT if set (lets a collaborator point at the shared dir)
  2. <repo>/results  (a symlink to the shared dir on the production system)
"""
from __future__ import annotations
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def results_root() -> Path:
    env = os.environ.get("CARDIAC_RESULTS_ROOT")
    return Path(env).resolve() if env else (REPO_ROOT / "results")


RESULTS_ROOT = results_root()
