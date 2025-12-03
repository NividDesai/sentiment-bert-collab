from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_split(
    df: pd.DataFrame, label_col: str = "label", test_size=0.2, random_state=42
):
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=random_state
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df
