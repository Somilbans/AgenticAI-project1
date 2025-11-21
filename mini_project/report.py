"""
User-friendly reporting helpers.
"""

from pathlib import Path
from typing import Iterable, Tuple


def describe(result: Tuple[Path, int, int]) -> str:
    """
    Turn a (destination, rows, columns) tuple into a readable string.
    """
    destination, rows, cols = result
    return f"{destination.name}: {rows} rows x {cols} columns"


def print_summary(results: Iterable[Tuple[Path, int, int]]) -> None:
    """
    Display a line-by-line summary for quick verification.
    """
    print("\nConverted files:")
    for result in results:
        print(f" - {describe(result)}")

