"""
Output helpers.

We isolate persistence so swapping CSV for Parquet (or anything else) is a
single-line edit.
"""

from pathlib import Path

import pandas as pd


def save_as_csv(df: pd.DataFrame, source_file: Path, output_dir: Path) -> Path:
    """
    Persist the DataFrame to CSV and return the destination path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{source_file.stem}.csv"
    df.to_csv(destination, index=False)
    return destination

