"""
inference.py
Simple inference utility. Example usage:
python -m src.inference "I loved the movie"
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

MODEL_DIR = "./outputs"  # where the trainer saved weights; update if different

def load_resources(model_dir: str = MODEL_DIR, model_name: str = "bert-base-uncased"):
    # If a fine-tuned model exists in model_dir, load from there; otherwise load base model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_dir if torch.cuda.is_available() else model_name)
    return tokenizer, model

def predict(texts: List[str], tokenizer, model, max_length: int = 128):
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    preds = torch.argmax(probs, dim=-1).tolist()
    return [{"text": t, "pred": p, "scores": probs[i].tolist()} for i, (t, p) in enumerate(zip(texts, preds))]

if _name_ == "_main_":
    import sys
    tokenizer, model = load_resources()
    texts = sys.argv[1:] or ["This is great!"]
    res = predict(texts, tokenizer, model)
    for r in res:
        print(f"Text: {r['text']}\nPrediction: {r['pred']}\nScores: {r['scores']}\n")