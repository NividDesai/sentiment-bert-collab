"""
model.py
Defines model creation and a train function using Hugging Face Trainer
"""
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from typing import Dict
import numpy as np
from datasets import Dataset

def build_model(model_name: str = "bert-base-uncased", num_labels: int = 2):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model

def hf_dataset_from_tensors(tokenized, labels):
    # tokenized: dict with input_ids, attention_mask (tensors)
    # datasets expects lists or numpy arrays
    data = {k: v.numpy().tolist() for k, v in tokenized.items()}
    data["labels"] = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    return Dataset.from_dict(data)

def train_model(model, tokenized_train, labels_train, tokenized_val, labels_val, output_dir="./outputs", epochs=1):
    train_ds = hf_dataset_from_tensors(tokenized_train, labels_train)
    val_ds = hf_dataset_from_tensors(tokenized_val, labels_val)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="no",
        seed=42,
        disable_tqdm=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer.train()
    eval_res = trainer.evaluate()
    return trainer, eval_res