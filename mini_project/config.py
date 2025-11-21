"""
Configuration values for the mini spreadsheet project.

Keeping configuration isolated makes the rest of the code concise and easy
to tweak without editing logic files.
"""

from pathlib import Path
from typing import List

# Files to ingest. Extend this list whenever new spreadsheets arrive.
FILES_TO_READ: List[Path] = [
    Path("OnBench.xlsx"),
    Path("AvailablePositions.xlsb"),
]

# Where converted CSV files will be stored.
OUTPUT_DIR: Path = Path("mini_output")

# Location for the persistent Chroma database.
VECTOR_DB_DIR: Path = Path("mini_chroma")

