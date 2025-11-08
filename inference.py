"""
inference.py
Script for running inference on new text samples
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run inference with trained BERT model")
    parser.add_argument("--model-path", type=str, default="./outputs/final_model", help="Path to saved model")
    parser.add_argument("--text", type=str, help="Text to classify (single prediction)")
    parser.add_argument("--file", type=str, help="File with texts to classify (one per line)")
    return parser.parse_args()


def predict_sentiment(text, model, tokenizer, device='cpu'):
    """Predict sentiment for a single text"""
    # Tokenize
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': confidence,
        'probabilities': {
            'negative': probabilities[0][0].item(),
            'positive': probabilities[0][1].item()
        }
    }


def main():
    """Main inference pipeline"""
    args = parse_args()
    
    # Check if model exists
    try:
        print(f"Loading model from {args.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        
        # Use GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f"Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have trained a model first using train.py")
        return
    
    # Single text prediction
    if args.text:
        print("\n" + "=" * 60)
        result = predict_sentiment(args.text, model, tokenizer, device)
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities:")
        print(f"  Negative: {result['probabilities']['negative']:.4f}")
        print(f"  Positive: {result['probabilities']['positive']:.4f}")
        print("=" * 60)
    
    # Batch prediction from file
    elif args.file:
        print(f"\nProcessing file: {args.file}")
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"Found {len(texts)} texts to process\n")
            print("=" * 60)
            
            for i, text in enumerate(texts, 1):
                result = predict_sentiment(text, model, tokenizer, device)
                print(f"\n[{i}/{len(texts)}]")
                print(f"Text: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
                print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.4f})")
            
            print("\n" + "=" * 60)
            print(f"Processed {len(texts)} texts successfully")
            
        except FileNotFoundError:
            print(f"Error: File {args.file} not found")
        except Exception as e:
            print(f"Error processing file: {e}")
    
    else:
        print("Please provide either --text for single prediction or --file for batch prediction")
        print("Example: python inference.py --text 'This app is amazing!'")
        print("Example: python inference.py --file test_samples.txt")


if __name__ == "__main__":
    main()

