import pandas as pd
from src.tokenize_helper import get_tokenizer, tokenize_dataframe

def test_tokenizer_shapes():
    tokenizer = get_tokenizer()
    df = pd.DataFrame({"text":["a b c", "d e f"], "label":[1,0]})
    out = tokenize_dataframe(df, tokenizer)
    assert out["input_ids"].shape[0] == 2
    assert out["attention_mask"].shape == out["input_ids"].shape