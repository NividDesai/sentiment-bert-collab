"""
data_extraction.py
Functions to load the dataset safely and derive sentiment labels.
"""
import pandas as pd
from typing import List

# Expected columns from your dataset
EXPECTED_COLUMNS = ["content", "score"]

def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV into a DataFrame, verify required columns, 
    and create a 'label' column derived from 'score'.
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

    # Keep only relevant columns
    df = df[["content", "score"]].dropna().reset_index(drop=True)

    # Create sentiment label from score
    df["label"] = df["score"].apply(lambda x: 1 if x >= 3 else 0)

    # Rename content → text for compatibility with rest of pipeline
    df = df.rename(columns={"content": "text"})

    return df[["text", "label", "score"]]
