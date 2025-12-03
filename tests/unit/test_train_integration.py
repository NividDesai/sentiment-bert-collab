"""
Integration tests for the training pipeline
"""

import os
import pandas as pd
import pytest
import torch
from src.data_extraction import load_csv
from src.utils import train_val_split
from src.tokenize_helper import get_tokenizer, tokenize_dataframe
from src.model import build_model, train_model


TEST_CSV = "tests/fixtures/integration_test.csv"


def _check_accelerate_installed():
    """Helper to check if accelerate is installed"""
    try:
        import accelerate

        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def test_data():
    """Create test dataset"""
    os.makedirs("tests/fixtures", exist_ok=True)
    df = pd.DataFrame(
        {
            "content": [
                "This is great",
                "This is terrible",
                "Amazing product",
                "Worst experience",
                "Love it",
                "Hate it",
                "Excellent service",
                "Poor quality",
                "Highly recommend",
                "Never again",
            ]
            * 2,  # 20 samples
            "score": [5, 1, 5, 1, 5, 1, 5, 1, 5, 1] * 2,
        }
    )
    df.to_csv(TEST_CSV, index=False)
    yield TEST_CSV
    try:
        os.remove(TEST_CSV)
    except OSError:
        pass


@pytest.mark.skipif(
    not _check_accelerate_installed(),
    reason="Requires accelerate>=0.26.0 package for training. Install with: pip install accelerate>=0.26.0",
)
def test_full_pipeline(test_data):
    """Test the complete training pipeline end-to-end (requires accelerate package)"""
    # Load data
    df = load_csv(test_data)
    assert len(df) == 20
    assert "label" in df.columns

    # Split data
    train_df, val_df = train_val_split(df, test_size=0.2, random_state=42)
    assert len(train_df) > 0
    assert len(val_df) > 0

    # Tokenize
    tokenizer = get_tokenizer("bert-base-uncased")
    tokenized_train = tokenize_dataframe(train_df, tokenizer, max_length=32)
    tokenized_val = tokenize_dataframe(val_df, tokenizer, max_length=32)

    assert tokenized_train["input_ids"].shape[0] == len(train_df)
    assert tokenized_val["input_ids"].shape[0] == len(val_df)

    # Build model
    model = build_model(num_labels=2)
    assert model is not None

    # Train (1 epoch for speed)
    labels_train = torch.tensor(train_df["label"].values)
    labels_val = torch.tensor(val_df["label"].values)

    trainer, eval_results = train_model(
        model=model,
        tokenized_train=tokenized_train,
        labels_train=labels_train,
        tokenized_val=tokenized_val,
        labels_val=labels_val,
        output_dir="./test_outputs",
        epochs=1,
    )

    assert trainer is not None
    assert "eval_accuracy" in eval_results
    assert 0 <= eval_results["eval_accuracy"] <= 1

    # Cleanup
    import shutil

    if os.path.exists("./test_outputs"):
        shutil.rmtree("./test_outputs")
