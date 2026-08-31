"""Smoke test: every Python module in the repo parses.

Catches syntax errors across the whole tree in under a second, without
importing anything — so it needs no FEniCSx and is safe on the login node.

    python3 tests/test_syntax.py
"""
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "data", "results",
             "paraview_exports", "logs"}


def python_files():
    for path in sorted(REPO.rglob("*.py")):
        if SKIP_DIRS.isdisjoint(path.relative_to(REPO).parts):
            yield path


def main():
    failures = []
    checked = 0
    for path in python_files():
        checked += 1
        try:
            ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            failures.append((path.relative_to(REPO), exc))

    for rel, exc in failures:
        print(f"FAIL {rel}:{exc.lineno}: {exc.msg}")

    if failures:
        print(f"\n{len(failures)}/{checked} files have syntax errors")
        return 1
    print(f"{checked}/{checked} files parse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
