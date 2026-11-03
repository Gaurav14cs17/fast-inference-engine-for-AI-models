#!/usr/bin/env python3
"""Verify the numbered-module DAG.

Run from the project root:

    python tools/check_dag.py

Every numbered file may only import modules with a lower number.  Package
``__init__.py`` files use the loader and are not part of the DAG check.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "study_book"
NUMBERED = re.compile(r"^(\d\d)_([a-zA-Z0-9_]+)\.py$")

FROM_PACKAGE = re.compile(r"from\s+study_book\.([a-z0-9_]+)\s+import\s+([A-Za-z0-9_,\s]+)")
FROM_MODULE = re.compile(r"from\s+study_book\.([a-z0-9_]+)\.([A-Za-z0-9_]+)\s+import")
IMPORT_MODULE = re.compile(r"import\s+study_book\.([a-z0-9_]+)\.([A-Za-z0-9_]+)")


def main() -> int:
    files = sorted(
        (p for p in PKG.rglob("[0-9][0-9]_*.py") if NUMBERED.match(p.name)),
        key=lambda p: int(p.name[:2]),
    )
    alias_to_number: dict[str, int] = {}
    for path in files:
        match = NUMBERED.match(path.name)
        assert match
        alias_to_number[match.group(2)] = int(match.group(1))

    errors: list[str] = []
    edges: list[tuple[int, int, str]] = []

    for path in files:
        number = int(path.name[:2])
        src = path.read_text(encoding="utf-8")

        deps: set[str] = set()
        for match in FROM_PACKAGE.finditer(src):
            deps.update(part.strip() for part in match.group(2).split(","))
        for match in FROM_MODULE.finditer(src):
            deps.add(match.group(2))
        for match in IMPORT_MODULE.finditer(src):
            deps.add(match.group(2))

        for alias in sorted(deps):
            alias = alias.split(" as ")[0].strip()
            if alias not in alias_to_number:
                # Likely a package name or a standard-library symbol.
                continue
            dep = alias_to_number[alias]
            edges.append((dep, number, path.name))
            if dep >= number:
                errors.append(
                    f"{path.relative_to(ROOT)}: imports {alias} (u{dep:02d}) but is u{number:02d}"
                )

    print(f"{'from':>4} -> {'to':<4}  file")
    for dep, number, name in sorted(edges):
        print(f"u{dep:02d} -> u{number:02d}   {name}")

    if errors:
        print("\nDAG ERRORS:", file=sys.stderr)
        for error in errors:
            print(" -", error, file=sys.stderr)
        return 1

    print(f"\nOK: {len(files)} numbered files, {len(edges)} import edges, DAG only goes forward.")
    print(
        "Chapters: 01 foundations(01-04) 02 configuration(05-07) 03 model_blocks(08-19) "
        "04 generation(20-25) 05 runtime(26-29)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
