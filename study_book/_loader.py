"""Dynamic loader for the numbered study modules.

The filenames are intentionally numbered (``01_dtypes.py``, ``02_buffer.py``,
...) to show the DAG order, but Python identifiers cannot start with a digit.
Each subpackage's ``__init__.py`` therefore registers its numbered files under
clean aliases:

    01_dtypes.py       -> study_book.chapter_01_foundations.dtypes
    02_buffer.py       -> study_book.chapter_01_foundations.buffer
    03_kernels.py      -> study_book.chapter_01_foundations.kernels
    04_backend.py      -> study_book.chapter_01_foundations.backend
    ...

The loader executes files in numeric order, so a file may import any alias from
a lower number.  ``tools/check_dag.py`` verifies that rule statically.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

NUMBERED = re.compile(r"^(\d\d)_([a-zA-Z0-9_]+)\.py$")


def alias_for(path: Path) -> str:
    match = NUMBERED.match(path.name)
    if not match:
        raise ValueError(f"not a numbered module: {path.name}")
    return match.group(2)


def load_numbered(
    package_name: str,
    package_path: str | Path,
    verbose: bool = False,
) -> dict[str, ModuleType]:
    """Load every ``NN_name.py`` in ``package_path`` as ``package_name.name``."""
    path = Path(package_path)
    files = sorted(
        (p for p in path.glob("[0-9][0-9]_*.py") if NUMBERED.match(p.name)),
        key=lambda p: int(p.name[:2]),
    )
    loaded: dict[str, ModuleType] = {}
    for file in files:
        alias = alias_for(file)
        full_name = f"{package_name}.{alias}"
        if full_name in sys.modules:
            module = sys.modules[full_name]
        else:
            spec = importlib.util.spec_from_file_location(full_name, file)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load {file}")
            module = importlib.util.module_from_spec(spec)
            # Register before exec so lower-numbered imports resolve.
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
        setattr(sys.modules[package_name], alias, module)
        loaded[alias] = module
        if verbose:
            print(f"{file.name:24s} -> {full_name}")
    return loaded
