import os
import pandas as pd
import pytest
from src.data_extraction import load_csv

TEST_CSV = "tests/fixtures/test_dataset.csv"

def setup_module(module):
    os.makedirs("tests/fixtures", exist_ok=True)
    df = pd.DataFrame({"content": ["good", "bad"], "score": [5, 1]})
    df.to_csv(TEST_CSV, index=False)

def teardown_module(module):
    try:
        os.remove(TEST_CSV)
    except OSError:
        pass

def test_load_csv_success():
    df = load_csv(TEST_CSV)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["text", "label", "score"]

    assert len(df) == 2

def test_load_csv_missing_file():
    with pytest.raises(FileNotFoundError):
        load_csv("tests/fixtures/no_file.csv")

def test_load_csv_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("only_column\nvalue")
    with pytest.raises(ValueError):
        load_csv(str(p))
