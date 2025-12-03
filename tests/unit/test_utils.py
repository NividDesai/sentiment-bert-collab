"""
Tests for utils.py
"""

import pandas as pd
from src.utils import train_val_split


def test_train_val_split_basic():
    """Test basic train/val split functionality"""
    df = pd.DataFrame(
        {"text": [f"sample {i}" for i in range(100)], "label": [0, 1] * 50}
    )

    train_df, val_df = train_val_split(df, test_size=0.2, random_state=42)

    assert len(train_df) == 80
    assert len(val_df) == 20
    assert list(train_df.columns) == list(df.columns)
    assert list(val_df.columns) == list(df.columns)


def test_train_val_split_stratification():
    """Test that split maintains label distribution"""
    df = pd.DataFrame(
        {
            "text": [f"sample {i}" for i in range(100)],
            "label": [0] * 30 + [1] * 70,  # 30% class 0, 70% class 1
        }
    )

    train_df, val_df = train_val_split(df, test_size=0.2, random_state=42)

    # Check proportions are maintained (approximately)
    train_class_1_ratio = (train_df["label"] == 1).sum() / len(train_df)
    val_class_1_ratio = (val_df["label"] == 1).sum() / len(val_df)

    assert 0.65 < train_class_1_ratio < 0.75  # Should be around 0.70
    assert 0.65 < val_class_1_ratio < 0.75  # Should be around 0.70


def test_train_val_split_no_overlap():
    """Test that train and val sets don't overlap"""
    df = pd.DataFrame(
        {"text": [f"unique_sample_{i}" for i in range(50)], "label": [0, 1] * 25}
    )

    train_df, val_df = train_val_split(df, test_size=0.3, random_state=42)

    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])

    assert len(train_texts.intersection(val_texts)) == 0


def test_train_val_split_reset_index():
    """Test that indices are properly reset"""
    df = pd.DataFrame(
        {"text": [f"sample {i}" for i in range(30)], "label": [0, 1] * 15}
    )

    train_df, val_df = train_val_split(df, test_size=0.2, random_state=42)

    # Check indices start from 0 and are consecutive
    assert train_df.index[0] == 0
    assert val_df.index[0] == 0
    assert list(train_df.index) == list(range(len(train_df)))
    assert list(val_df.index) == list(range(len(val_df)))


def test_train_val_split_reproducibility():
    """Test that split is reproducible with same random state"""
    df = pd.DataFrame(
        {"text": [f"sample {i}" for i in range(50)], "label": [0, 1] * 25}
    )

    train_df1, val_df1 = train_val_split(df, test_size=0.2, random_state=42)
    train_df2, val_df2 = train_val_split(df, test_size=0.2, random_state=42)

    pd.testing.assert_frame_equal(train_df1, train_df2)
    pd.testing.assert_frame_equal(val_df1, val_df2)


def test_train_val_split_different_sizes():
    """Test different test_size values"""
    df = pd.DataFrame(
        {"text": [f"sample {i}" for i in range(100)], "label": [0, 1] * 50}
    )

    for test_size in [0.1, 0.2, 0.3, 0.4]:
        train_df, val_df = train_val_split(df, test_size=test_size, random_state=42)
        expected_val_size = int(100 * test_size)
        assert abs(len(val_df) - expected_val_size) <= 1  # Allow for rounding
        assert len(train_df) + len(val_df) == 100
