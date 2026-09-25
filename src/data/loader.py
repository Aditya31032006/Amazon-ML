"""
Data Access Layer (Repository Pattern).
Handles reading TSV datasets and persisting/loading optimized Parquet files.
"""
from pathlib import Path
from typing import Optional
import pandas as pd

class DataLoader:
    """Repository handling file I/O for TSV and Parquet datasets."""

    @staticmethod
    def read_tsv(filepath: Path, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Reads a tab-separated TSV file with UTF-8 encoding.
        Ensures all columns are strings to prevent accidental type inference.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        return pd.read_csv(
            filepath,
            sep="\t",
            dtype=str,
            nrows=nrows,
            encoding="utf-8",
            keep_default_na=False,
            na_values=[""]
        )

    @staticmethod
    def save_tsv(df: pd.DataFrame, filepath: Path) -> None:
        """Saves a DataFrame to TSV format with UTF-8 encoding and tab delimiter."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, sep="\t", index=False, encoding="utf-8")

    @staticmethod
    def save_parquet(df: pd.DataFrame, filepath: Path) -> None:
        """Saves a DataFrame to Parquet format, creating parent directories if needed."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(filepath, index=False, engine="pyarrow", compression="snappy")

    @staticmethod
    def read_parquet(filepath: Path) -> pd.DataFrame:
        """Reads a Parquet file rapidly into memory."""
        if not filepath.exists():
            raise FileNotFoundError(f"Parquet file not found: {filepath}")
        return pd.read_parquet(filepath, engine="pyarrow")

