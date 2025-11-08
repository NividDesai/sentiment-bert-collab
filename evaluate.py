"""
evaluate.py
Comprehensive model evaluation script with detailed metrics and visualizations
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.data_extraction import load_csv
from src.data_processing import train_val_split, tokenize_dataframe


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate trained sentiment model")
    parser.add_argument("--model-path", type=str, default="./outputs/final_model",
                       help="Path to saved model")
    parser.add_argument("--data", type=str, default="dataset.csv",
                       help="Path to dataset CSV")
    parser.add_argument("--output-dir", type=str, default="./evaluation",
                       help="Output directory for evaluation results")
    parser.add_argument("--max-length", type=int, default=128,
                       help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    return parser.parse_args()


def predict_probabilities(model, tokenizer, texts, device='cpu', batch_size=8):
    """Get prediction probabilities for a list of texts"""
    model.eval()
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                          max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_probs)


def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {filepath}")


def plot_roc_curve(y_true, y_probs, output_dir):
    """Plot and save ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve',
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve saved to {filepath}")
    
    return roc_auc


def plot_precision_recall_curve(y_true, y_probs, output_dir):
    """Plot and save precision-recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs[:, 1])
    avg_precision = average_precision_score(y_true, y_probs[:, 1])
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Precision-Recall curve saved to {filepath}")
    
    return avg_precision


def plot_class_distribution(y_true, y_pred, class_names, output_dir):
    """Plot class distribution comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # True labels
    true_counts = pd.Series(y_true).value_counts().sort_index()
    axes[0].bar(range(len(true_counts)), true_counts.values,
                color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('True Label Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Predicted labels
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    axes[1].bar(range(len(pred_counts)), pred_counts.values,
                color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Class distribution plot saved to {filepath}")


def plot_confidence_distribution(y_probs, y_true, output_dir):
    """Plot confidence score distribution"""
    y_pred = np.argmax(y_probs, axis=1)
    confidence = np.max(y_probs, axis=1)
    correct = (y_pred == y_true)
    
    plt.figure(figsize=(12, 6))
    plt.hist(confidence[correct], bins=50, alpha=0.7, label='Correct Predictions',
             color='green', edgecolor='black')
    plt.hist(confidence[~correct], bins=50, alpha=0.7, label='Incorrect Predictions',
             color='red', edgecolor='black')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Confidence Distribution', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'confidence_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confidence distribution plot saved to {filepath}")


def save_metrics_report(metrics, output_dir):
    """Save comprehensive metrics to file"""
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    
    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:           {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (Macro):  {metrics['precision']:.4f}\n")
        f.write(f"Recall (Macro):     {metrics['recall']:.4f}\n")
        f.write(f"F1 Score (Macro):   {metrics['f1']:.4f}\n")
        f.write(f"ROC AUC:            {metrics['roc_auc']:.4f}\n")
        f.write(f"Avg Precision:      {metrics['avg_precision']:.4f}\n")
        f.write("\n")
        
        f.write("Classification Report:\n")
        f.write("-" * 70 + "\n")
        f.write(metrics['classification_report'])
        f.write("\n")
        
        f.write("Confusion Matrix:\n")
        f.write("-" * 70 + "\n")
        cm = metrics['confusion_matrix']
        f.write(f"                 Predicted Negative  Predicted Positive\n")
        f.write(f"Actual Negative  {cm[0][0]:>18}  {cm[0][1]:>18}\n")
        f.write(f"Actual Positive  {cm[1][0]:>18}  {cm[1][1]:>18}\n")
        f.write("\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"✓ Metrics report saved to {filepath}")


def save_predictions(texts, y_true, y_pred, y_probs, output_dir, n_samples=100):
    """Save sample predictions to CSV"""
    df = pd.DataFrame({
        'text': texts[:n_samples],
        'true_label': y_true[:n_samples],
        'predicted_label': y_pred[:n_samples],
        'confidence': np.max(y_probs[:n_samples], axis=1),
        'prob_negative': y_probs[:n_samples, 0],
        'prob_positive': y_probs[:n_samples, 1],
        'correct': (y_true[:n_samples] == y_pred[:n_samples])
    })
    
    filepath = os.path.join(output_dir, 'sample_predictions.csv')
    df.to_csv(filepath, index=False)
    print(f"✓ Sample predictions saved to {filepath}")


def main():
    """Main evaluation pipeline"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.data}")
    print(f"Output: {args.output_dir}")
    print("=" * 70 + "\n")
    
    # Load model
    print("[1/8] Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f"✓ Model loaded successfully on {device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Make sure you have trained a model first using train.py")
        return
    
    # Load data
    print("\n[2/8] Loading dataset...")
    df = load_csv(args.data)
    _, val_df = train_val_split(df, test_size=0.2, random_state=args.seed)
    print(f"✓ Loaded {len(val_df)} validation samples")
    
    # Get texts and labels
    texts = val_df['text'].tolist()
    y_true = val_df['label'].values
    
    # Get predictions
    print("\n[3/8] Generating predictions...")
    y_probs = predict_probabilities(model, tokenizer, texts, device, args.batch_size)
    y_pred = np.argmax(y_probs, axis=1)
    print(f"✓ Predictions generated for {len(texts)} samples")
    
    # Calculate metrics
    print("\n[4/8] Calculating metrics...")
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(
            y_true, y_pred, target_names=['Negative', 'Positive']
        )
    }
    
    # Print metrics
    print(f"✓ Accuracy:  {metrics['accuracy']:.4f}")
    print(f"✓ Precision: {metrics['precision']:.4f}")
    print(f"✓ Recall:    {metrics['recall']:.4f}")
    print(f"✓ F1 Score:  {metrics['f1']:.4f}")
    
    # Generate visualizations
    class_names = ['Negative', 'Positive']
    
    print("\n[5/8] Generating confusion matrix...")
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, args.output_dir)
    
    print("\n[6/8] Generating ROC curve...")
    roc_auc = plot_roc_curve(y_true, y_probs, args.output_dir)
    metrics['roc_auc'] = roc_auc
    
    print("\n[7/8] Generating precision-recall curve...")
    avg_precision = plot_precision_recall_curve(y_true, y_probs, args.output_dir)
    metrics['avg_precision'] = avg_precision
    
    print("\n[8/8] Generating additional plots...")
    plot_class_distribution(y_true, y_pred, class_names, args.output_dir)
    plot_confidence_distribution(y_probs, y_true, args.output_dir)
    
    # Save results
    print("\nSaving results...")
    save_metrics_report(metrics, args.output_dir)
    save_predictions(texts, y_true, y_pred, y_probs, args.output_dir)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - evaluation_metrics.txt (detailed metrics)")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curve.png")
    print(f"  - precision_recall_curve.png")
    print(f"  - class_distribution.png")
    print(f"  - confidence_distribution.png")
    print(f"  - sample_predictions.csv")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

