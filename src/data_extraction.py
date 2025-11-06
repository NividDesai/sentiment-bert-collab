"""
data_extraction.py
Functions to load the dataset safely.
"""
import pandas as pd
from typing import Tuple

EXPECTED_COLUMNS = ["text", "label"]  # adapt to your dataset columns

def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV into a DataFrame. Raises ValueError if columns missing.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {path}") from e
    except Exception as e:
        raise RuntimeError(f"Error reading CSV: {e}") from e

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Basic cleaning: drop NA in essential cols
    df = df.dropna(subset=EXPECTED_COLUMNS).reset_index(drop=True)
    return df
