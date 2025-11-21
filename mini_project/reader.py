"""
Functions dedicated to reading spreadsheet files into pandas DataFrames.

Splitting reader logic keeps main.py small and highlights the only Excel-specific
details (engine selection for .xlsb files).
"""

import re
from pathlib import Path

import pandas as pd


def load_workbook(file_path: Path) -> pd.DataFrame:
    """
    Load a workbook regardless of whether it is .xlsx or .xlsb.

    pandas needs the pyxlsb engine for binary Excel files (.xlsb). For .xlsx
    we can rely on the default engine (openpyxl), so we only set the engine
    when needed.
    """
    extension = file_path.suffix.lower()
    engine = "pyxlsb" if extension == ".xlsb" else None
    df = pd.read_excel(file_path, engine=engine)
    df = _normalize_headers(df)
    return df


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make headers machine-friendly by:
    1. Stripping whitespace.
    2. Replacing non-alphanumeric characters with underscores.
    3. Collapsing multiple underscores.
    """
    df = df.copy()
    cleaned = []
    for column in df.columns:
        name = re.sub(r"\s+", " ", str(column).strip())  # normalize spaces
        name = re.sub(r"[\/,\\]+", "", name)  # drop slashes and commas entirely
        name = re.sub(r"[^0-9a-zA-Z]+", "_", name)  # replace other special chars
        name = re.sub(r"_+", "_", name).strip("_").lower()
        cleaned.append(name)
    df.columns = cleaned
    return df

