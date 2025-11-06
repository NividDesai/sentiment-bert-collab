"""
data_processing.py
Text cleaning and preprocessing utilities.
"""
import re
from typing import List
import pandas as pd

def basic_clean(text: str) -> str:
    """
    Apply basic text cleaning: lowercase, remove urls, strip extra spaces, remove non-ascii.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s\.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def apply_clean(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].astype(str).apply(basic_clean)
    return df
