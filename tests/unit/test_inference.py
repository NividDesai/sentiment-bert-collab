"""
Tests for inference.py
"""

import torch
import pytest
import os
import shutil
from src.inference import load_model, predict_sentiment, predict_batch
from src.model import build_model
from transformers import AutoTokenizer


def test_load_model_cpu():
    """Test loading model on CPU"""
    # Create a temporary model to load
    model = build_model(num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Save to temporary directory
    temp_dir = "./test_model_temp"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)

        # Load the model
        loaded_model, loaded_tokenizer, device = load_model(temp_dir, device="cpu")

        # Verify the model and tokenizer are loaded
        assert loaded_model is not None
        assert loaded_tokenizer is not None
        assert device == "cpu"

        # Verify model is in eval mode
        assert not loaded_model.training

    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_load_model_auto_device():
    """Test loading model with auto device detection"""
    model = build_model(num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    temp_dir = "./test_model_auto_device"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)

        # Load with auto device detection (device=None)
        loaded_model, loaded_tokenizer, device = load_model(temp_dir, device=None)

        # Device should be either 'cpu' or 'cuda'
        assert device in ["cpu", "cuda"]

        # If CUDA is available, device should be 'cuda', otherwise 'cpu'
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device == expected_device

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_model_cuda():
    """Test loading model on CUDA"""
    model = build_model(num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    temp_dir = "./test_model_cuda"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)

        # Load on CUDA
        loaded_model, loaded_tokenizer, device = load_model(temp_dir, device="cuda")

        assert device == "cuda"
        # Verify model is on CUDA
        assert next(loaded_model.parameters()).is_cuda

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_predict_sentiment_structure():
    """Test predict_sentiment output structure"""
    # Create a simple model and tokenizer for testing
    model = build_model(num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    result = predict_sentiment("This is a test", model, tokenizer, device="cpu")

    # Check output structure
    assert "text" in result
    assert "sentiment" in result
    assert "confidence" in result
    assert "probabilities" in result

    assert result["text"] == "This is a test"
    assert result["sentiment"] in ["Positive", "Negative"]
    assert 0 <= result["confidence"] <= 1
    assert "negative" in result["probabilities"]
    assert "positive" in result["probabilities"]


def test_predict_sentiment_probabilities_sum():
    """Test that probabilities sum to 1"""
    model = build_model(num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    result = predict_sentiment("Test text", model, tokenizer, device="cpu")

    prob_sum = result["probabilities"]["negative"] + result["probabilities"]["positive"]
    assert abs(prob_sum - 1.0) < 0.001  # Allow for floating point errors


def test_predict_batch_length():
    """Test batch prediction returns correct number of results"""
    model = build_model(num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
    results = predict_batch(texts, model, tokenizer, device="cpu", batch_size=2)

    assert len(results) == 4
    assert all("sentiment" in r for r in results)
    assert all("confidence" in r for r in results)


def test_predict_batch_structure():
    """Test batch prediction output structure"""
    model = build_model(num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    texts = ["Good product", "Bad service"]
    results = predict_batch(texts, model, tokenizer, device="cpu")

    for i, result in enumerate(results):
        assert result["text"] == texts[i]
        assert result["sentiment"] in ["Positive", "Negative"]
        assert 0 <= result["confidence"] <= 1
        assert "probabilities" in result


def test_predict_batch_different_sizes():
    """Test batch prediction with different batch sizes"""
    model = build_model(num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    texts = ["Text " + str(i) for i in range(10)]

    # Test with different batch sizes
    for batch_size in [1, 3, 5, 10]:
        results = predict_batch(
            texts, model, tokenizer, device="cpu", batch_size=batch_size
        )
        assert len(results) == 10


def test_predict_sentiment_consistency():
    """Test that predictions are consistent for same input"""
    model = build_model(num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model.eval()

    text = "Consistent test"
    result1 = predict_sentiment(text, model, tokenizer, device="cpu")
    result2 = predict_sentiment(text, model, tokenizer, device="cpu")

    # Should get same results
    assert result1["sentiment"] == result2["sentiment"]
    assert abs(result1["confidence"] - result2["confidence"]) < 0.001


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_predict_sentiment_gpu():
    """Test prediction on GPU if available"""
    model = build_model(num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    result = predict_sentiment("GPU test", model, tokenizer, device="cuda")

    assert "sentiment" in result
    assert result["sentiment"] in ["Positive", "Negative"]
