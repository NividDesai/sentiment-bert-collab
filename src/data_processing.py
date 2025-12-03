"""
data_processing.py
Functions for data preprocessing, tokenization, and train/validation splitting.
"""

from transformers import AutoTokenizer
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split


def get_tokenizer(model_name: str = "bert-base-uncased"):
    """
    Load and return a pre-trained tokenizer.

    Args:
        model_name: Name of the pre-trained model

    Returns:
        Tokenizer instance
    """
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_dataframe(
    df: pd.DataFrame, tokenizer, text_col: str = "text", max_length: int = 128
) -> Dict[str, Any]:
    """
    Tokenize text data in a DataFrame.

    Args:
        df: DataFrame containing text data
        tokenizer: Tokenizer instance
        text_col: Name of the column containing text
        max_length: Maximum sequence length

    Returns:
        Dictionary with tokenized inputs (input_ids, attention_mask)
    """
    texts = df[text_col].tolist()
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def train_val_split(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets with stratification.

    Args:
        df: DataFrame to split
        label_col: Name of the label column for stratification
        test_size: Proportion of data for validation (0.0 to 1.0)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df)
    """
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=random_state
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df
