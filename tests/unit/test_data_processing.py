import pandas as pd
from src.data_processing import basic_clean, apply_clean
from src.utils import train_val_split

def test_basic_clean():
    text = "Hello WORLD!!! Visit https://x.com"
    assert "http" not in basic_clean(text)
    assert basic_clean(text).islower()

def test_apply_clean_and_split():
    df = pd.DataFrame({"text":["A b", "C d", "E f", "G h"], "label":[0,1,0,1]})
    df2 = apply_clean(df)
    train, val = train_val_split(df2, test_size=0.5, random_state=1)
    assert len(train) == 2
    assert len(val) == 2
