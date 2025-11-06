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

def test_load_csv_generic_error(tmp_path):
    """Test generic exception handling for corrupted files"""
    p = tmp_path / "corrupted.csv"
    # Create a file that will cause a parsing error
    p.write_text("content,score\n\"unclosed quote,5\nmore,data")
    with pytest.raises(RuntimeError):
        load_csv(str(p))

def test_load_csv_label_creation():
    """Test that labels are created correctly from scores"""
    df = load_csv(TEST_CSV)
    # score=5 should be label=1 (positive)
    positive_row = df[df['score'] == 5].iloc[0]
    assert positive_row['label'] == 1
    # score=1 should be label=0 (negative)
    negative_row = df[df['score'] == 1].iloc[0]
    assert negative_row['label'] == 0

def test_load_csv_with_na_values(tmp_path):
    """Test that NA values are dropped"""
    p = tmp_path / "with_na.csv"
    df = pd.DataFrame({
        "content": ["good", None, "bad", "okay"],
        "score": [5, 3, None, 4]
    })
    df.to_csv(p, index=False)
    result_df = load_csv(str(p))
    # Should only have 2 rows (good and okay), NA rows dropped
    assert len(result_df) == 2
    assert all(result_df['text'].notna())
    assert all(result_df['score'].notna())
