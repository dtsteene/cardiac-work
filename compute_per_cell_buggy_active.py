"""Thin wrapper around compute_per_cell.py that pre-applies the pre-99e78f0
active-stress monkey patch. Same CLI as compute_per_cell.py."""
from __future__ import annotations

import runpy

import pulse_legacy_active_patch  # noqa: F401  side-effect: monkey patch

runpy.run_path(
    "/global/D1/homes/dtsteene/cardiac-work/compute_per_cell.py",
    run_name="__main__",
)
