"""
train.py
Main training script for sentiment analysis with BERT
"""
import os
import argparse
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_extraction import load_csv
from src.data_processing import train_val_split, get_tokenizer, tokenize_dataframe
from src.model import build_model, train_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train BERT for sentiment analysis")
    parser.add_argument("--data", type=str, default="dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory for models and results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(trainer, tokenized_test, labels_test):
    """Evaluate model and return metrics"""
    from src.model import hf_dataset_from_tensors
    
    test_ds = hf_dataset_from_tensors(tokenized_test, labels_test)
    predictions = trainer.predict(test_ds)
    
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = labels_test.numpy() if hasattr(labels_test, 'numpy') else np.array(labels_test)
    
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'f1': f1_score(true_labels, pred_labels, average='weighted'),
        'precision': precision_score(true_labels, pred_labels, average='weighted'),
        'recall': recall_score(true_labels, pred_labels, average='weighted')
    }
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    return metrics, cm, pred_labels, true_labels


def plot_confusion_matrix(cm, output_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")


def save_metrics(metrics, output_dir):
    """Save metrics to file"""
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Model Evaluation Metrics\n")
        f.write("=" * 40 + "\n")
        for key, value in metrics.items():
            f.write(f"{key.capitalize()}: {value:.4f}\n")
    print(f"Metrics saved to {metrics_file}")


def main():
    """Main training pipeline"""
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("BERT Sentiment Analysis Training")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Model: {args.model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Load and split data
    print("\n[1/6] Loading dataset...")
    df = load_csv(args.data)
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    print("\n[2/6] Splitting dataset...")
    train_df, val_df = train_val_split(df, test_size=0.2, random_state=args.seed)
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
    
    # Tokenize data
    print("\n[3/6] Tokenizing data...")
    tokenizer = get_tokenizer(args.model_name)
    
    tokenized_train = tokenize_dataframe(train_df, tokenizer, max_length=args.max_length)
    tokenized_val = tokenize_dataframe(val_df, tokenizer, max_length=args.max_length)
    
    labels_train = torch.tensor(train_df['label'].values)
    labels_val = torch.tensor(val_df['label'].values)
    
    print(f"Tokenized shapes - Train: {tokenized_train['input_ids'].shape}, Val: {tokenized_val['input_ids'].shape}")
    
    # Build model
    print("\n[4/6] Building model...")
    model = build_model(model_name=args.model_name, num_labels=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n[5/6] Training model...")
    trainer, eval_results = train_model(
        model=model,
        tokenized_train=tokenized_train,
        labels_train=labels_train,
        tokenized_val=tokenized_val,
        labels_val=labels_val,
        output_dir=args.output_dir,
        epochs=args.epochs
    )
    
    print(f"\nTraining completed!")
    print(f"Validation accuracy: {eval_results['eval_accuracy']:.4f}")
    
    # Evaluate on validation set
    print("\n[6/6] Final evaluation...")
    metrics, cm, pred_labels, true_labels = evaluate_model(trainer, tokenized_val, labels_val)
    
    print("\nFinal Metrics:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")
    print("-" * 40)
    
    # Save results
    save_metrics(metrics, args.output_dir)
    plot_confusion_matrix(cm, args.output_dir)
    
    # Save model
    model_path = os.path.join(args.output_dir, 'final_model')
    trainer.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

