"""
inference.py
Functions for running inference with trained models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List


def load_model(model_path: str, device: str = None):
    """
    Load a trained model and tokenizer from disk.

    Args:
        model_path: Path to the saved model directory
        device: Device to load model on ('cpu', 'cuda', or None for auto-detect)

    Returns:
        Tuple of (model, tokenizer, device)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def predict_sentiment(
    text: str, model, tokenizer, device: str = "cpu"
) -> Dict[str, any]:
    """
    Predict sentiment for a single text.

    Args:
        text: Input text to classify
        model: Trained model
        tokenizer: Tokenizer instance
        device: Device to run inference on

    Returns:
        Dictionary with sentiment, confidence, and probabilities
    """
    # Tokenize
    inputs = tokenizer(
        text, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

    sentiment = "Positive" if predicted_class == 1 else "Negative"

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": {
            "negative": probabilities[0][0].item(),
            "positive": probabilities[0][1].item(),
        },
    }


def predict_batch(
    texts: List[str], model, tokenizer, device: str = "cpu", batch_size: int = 8
) -> List[Dict[str, any]]:
    """
    Predict sentiment for a batch of texts.

    Args:
        texts: List of input texts
        model: Trained model
        tokenizer: Tokenizer instance
        device: Device to run inference on
        batch_size: Batch size for processing

    Returns:
        List of prediction dictionaries
    """
    results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)

        # Process results
        for j, text in enumerate(batch_texts):
            pred_class = predicted_classes[j].item()
            confidence = probabilities[j][pred_class].item()
            sentiment = "Positive" if pred_class == 1 else "Negative"

            results.append(
                {
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "probabilities": {
                        "negative": probabilities[j][0].item(),
                        "positive": probabilities[j][1].item(),
                    },
                }
            )

    return results
