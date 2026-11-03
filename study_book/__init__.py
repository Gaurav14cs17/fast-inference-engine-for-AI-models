"""uzu study book.

Public chapters, loaded in DAG order:

    chapter_01_foundations   01-04
    chapter_02_configuration 05-07
    chapter_03_model_blocks  08-19
    chapter_04_generation    20-25
    chapter_05_runtime       26-29

Importing ``study_book`` loads every numbered module and registers clean aliases
(``study_book.chapter_01_foundations.dtypes``,
``study_book.chapter_03_model_blocks.attention``, ...).  The numbered filenames
remain on disk so the study order is obvious.
"""

from __future__ import annotations

from ._loader import load_numbered

# The order of these imports is the DAG order.
from . import (  # noqa: F401,E402
    chapter_01_foundations,
    chapter_02_configuration,
    chapter_03_model_blocks,
    chapter_04_generation,
    chapter_05_runtime,
)

__all__ = [
    "load_numbered",
    "chapter_01_foundations",
    "chapter_02_configuration",
    "chapter_03_model_blocks",
    "chapter_04_generation",
    "chapter_05_runtime",
]
