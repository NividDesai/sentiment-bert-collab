from transformers import AutoTokenizer
from typing import Dict, Any
import pandas as pd


def get_tokenizer(model_name: str = "bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_dataframe(
    df: pd.DataFrame, tokenizer, text_col: str = "text", max_length: int = 128
) -> Dict[str, Any]:
    texts = df[text_col].tolist()
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
