import torch
from src.model import build_model
from transformers import AutoTokenizer

def test_model_instantiation_and_forward():
    model = build_model(num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(["I love it", "I hate it"], padding=True, truncation=True, return_tensors="pt")
    # forward pass
    outputs = model(**inputs)
    logits = outputs.logits
    assert logits.shape[0] == 2
    assert logits.shape[1] == 2