# Sentiment Analysis with BERT



A production-ready sentiment analysis system using BERT (Bidirectional Encoder Representations from Transformers) for binary sentiment classification of app reviews.

## 📋 Project Overview

This project implements a complete MLOps pipeline for sentiment analysis, including:
- Data extraction and preprocessing
- BERT-based model training
- Comprehensive testing and CI/CD
- Model evaluation and visualization
- Inference capabilities

**Team Members:** Nivid DESAI & Shreya PALLISSERY

## 🌟 Features

- ✅ **BERT-based Classification**: Uses pre-trained `bert-base-uncased` for transfer learning
- ✅ **Automated Testing**: Comprehensive unit and integration tests with **100% coverage** 🎯
- ✅ **CI/CD Pipeline**: GitHub Actions for automated testing and quality checks
- ✅ **Code Quality**: Pre-commit hooks, Black formatting, and Ruff linting
- ✅ **Reproducibility**: Fixed random seeds for consistent results
- ✅ **Visualization**: Confusion matrix and metrics reporting
- ✅ **Production Ready**: Inference script for deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/NividDESAI/sentiment-bert-collab.git
cd sentiment-bert-collab
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up pre-commit hooks** (optional but recommended)
```bash
pre-commit install
```

## 📊 Dataset

The project uses app review data with the following columns:
- `content`: Review text
- `score`: Rating (1-5)
- Derived `label`: Binary sentiment (0=negative for score<3, 1=positive for score≥3)

Dataset statistics:
- Total samples: 14,078 reviews
- Binary classification: Positive/Negative sentiment

## 🎯 Usage Examples

### Example 1: Complete Training Workflow

**Step 1: Prepare your data**
```bash
# Ensure dataset.csv is in the project root
# Format: columns 'content' and 'score'
ls dataset.csv
```

**Step 2: Train the model**
```bash
# Basic training with defaults
python train.py

# Advanced training with custom parameters
python train.py \
    --data dataset.csv \
    --model-name bert-base-uncased \
    --epochs 3 \
    --batch-size 8 \
    --max-length 128 \
    --output-dir ./outputs \
    --seed 42
```

**Step 3: Check results**
```bash
# View metrics
cat outputs/metrics.txt

# View confusion matrix
open outputs/confusion_matrix.png  # macOS
# or
start outputs/confusion_matrix.png  # Windows
```

**Training arguments:**
- `--data`: Path to CSV dataset (default: `dataset.csv`)
- `--model-name`: HuggingFace model identifier (default: `bert-base-uncased`)
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 8)
- `--max-length`: Maximum sequence length (default: 128)
- `--output-dir`: Output directory for model and results (default: `./outputs`)
- `--seed`: Random seed for reproducibility (default: 42)

**Output files:**
- `outputs/final_model/`: Saved model checkpoint (can be used for inference)
- `outputs/metrics.txt`: Performance metrics (accuracy, precision, recall, F1)
- `outputs/confusion_matrix.png`: Confusion matrix visualization

### Example 2: Single Text Inference

**Predict sentiment for one review:**
```bash
python inference.py \
    --model-path outputs/final_model/ \
    --text "This app is amazing! I love it!"
```

**Output:**
```
Text: This app is amazing! I love it!
Sentiment: Positive
Confidence: 0.95
Probabilities:
  - Negative: 0.05
  - Positive: 0.95
```

### Example 3: Batch Inference

**Step 1: Create input file**
```bash
# Create a file with one review per line
cat > reviews.txt << EOF
Great app, highly recommend to everyone!
Terrible experience, waste of money
The app is okay, nothing special
Amazing features and great user interface
Worst app I've ever used
EOF
```

**Step 2: Run batch prediction**
```bash
python inference.py \
    --model-path outputs/final_model/ \
    --file reviews.txt
```

**Output:**
```
Processing 5 reviews...

Review 1: Great app, highly recommend to everyone!
  Sentiment: Positive (Confidence: 0.92)

Review 2: Terrible experience, waste of money
  Sentiment: Negative (Confidence: 0.89)

...
```

### Example 4: Model Evaluation

**Run comprehensive evaluation:**
```bash
python evaluate.py \
    --model-path outputs/final_model/ \
    --data dataset.csv \
    --output-dir evaluation/
```

**Generated files:**
- `evaluation/confusion_matrix.png`: Confusion matrix heatmap
- `evaluation/roc_curve.png`: ROC curve with AUC
- `evaluation/pr_curve.png`: Precision-recall curve
- `evaluation/metrics_report.txt`: Detailed metrics
- `evaluation/sample_predictions.csv`: Sample predictions

### Example 5: Using Python API

**Load and use model programmatically:**
```python
from src.inference import load_model, predict_sentiment

# Load model
model, tokenizer, device = load_model("outputs/final_model/", device="cpu")

# Predict sentiment
result = predict_sentiment(
    "This is the best app ever!",
    model, tokenizer, device
)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Example 6: Testing

**Run all tests:**
```bash
pytest tests/unit/ -v
```

**Run with coverage report (100% coverage achieved! 🎉):**
```bash
pytest tests/unit/ --cov=src --cov-report=term-missing --cov-report=html
# Coverage: 100% (111/111 statements)
# 40 passed, 4 skipped

# View HTML report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

**Run specific test categories:**
```bash
# Test data extraction
pytest tests/unit/test_data_extraction.py -v

# Test model
pytest tests/unit/test_model.py -v

# Test inference
pytest tests/unit/test_inference.py -v

# Integration test (requires accelerate)
pytest tests/unit/test_train_integration.py -v
```

### Example 7: Development Workflow

**1. Make code changes**
```bash
# Edit source files
vim src/model.py
```

**2. Run tests**
```bash
pytest tests/unit/ --cov=src
```

**3. Format code**
```bash
black src/ tests/
ruff check src/ tests/
```

**4. Commit changes**
```bash
git add .
git commit -m "feat(model): add new feature"
git push
```

## 🏗️ Project Structure

```
sentiment-bert-collab/
├── src/
│   ├── data_extraction.py      # Data loading and preprocessing
│   ├── data_processing.py       # Tokenization and train/val splitting
│   ├── model.py                # Model definition and training
│   ├── inference.py             # Model loading and prediction functions
│   ├── tokenize_helper.py      # Tokenization utilities
│   └── utils.py                # Helper functions
├── tests/
│   ├── fixtures/               # Test data
│   └── unit/                   # Unit tests
│       ├── test_data_extraction.py
│       ├── test_data_processing.py
│       ├── test_model.py
│       ├── test_inference.py
│       ├── test_tokenize.py
│       ├── test_utils.py
│       └── test_train_integration.py
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── report/
│   └── project_report.md       # Detailed project documentation
├── train.py                    # Main training script
├── inference.py                # Inference CLI script
├── evaluate.py                 # Model evaluation script
├── dataset.csv                 # Training data
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 📚 Component Descriptions

### Core Modules (`src/`)

#### `data_extraction.py`
**Purpose:** Load and preprocess CSV data for sentiment analysis

**Key Functions:**
- `load_csv(path)`: Loads CSV file, validates columns, creates binary labels
- Handles missing files, corrupted data, and missing columns
- Creates sentiment labels from scores (score ≥ 3 = positive, < 3 = negative)

**Usage:**
```python
from src.data_extraction import load_csv
df = load_csv("dataset.csv")
# Returns: DataFrame with 'text', 'label', 'score' columns
```

#### `data_processing.py`
**Purpose:** Tokenization and data splitting for BERT model

**Key Functions:**
- `get_tokenizer(model_name)`: Loads BERT tokenizer
- `tokenize_dataframe(df, tokenizer)`: Converts text to BERT tokens
- `train_val_split(df)`: Creates stratified train/validation split (80/20)

**Usage:**
```python
from src.data_processing import get_tokenizer, tokenize_dataframe, train_val_split
tokenizer = get_tokenizer("bert-base-uncased")
tokenized = tokenize_dataframe(df, tokenizer, max_length=128)
train_df, val_df = train_val_split(df, test_size=0.2)
```

#### `model.py`
**Purpose:** BERT model architecture and training

**Key Functions:**
- `build_model(model_name, num_labels)`: Creates BERT classification model
- `train_model(...)`: Trains model using Hugging Face Trainer
- `hf_dataset_from_tensors(...)`: Converts tensors to HuggingFace Dataset

**Usage:**
```python
from src.model import build_model, train_model
model = build_model(num_labels=2)
trainer, eval_results = train_model(model, train_data, val_data, epochs=3)
```

#### `inference.py`
**Purpose:** Model loading and sentiment prediction

**Key Functions:**
- `load_model(model_path, device)`: Loads trained model and tokenizer
- `predict_sentiment(text, model, tokenizer)`: Predicts sentiment for single text
- `predict_batch(texts, model, tokenizer)`: Batch prediction for multiple texts

**Usage:**
```python
from src.inference import load_model, predict_sentiment
model, tokenizer, device = load_model("final_model/", device="cpu")
result = predict_sentiment("This app is great!", model, tokenizer, device)
# Returns: {'text': '...', 'sentiment': 'Positive', 'confidence': 0.95, ...}
```

#### `tokenize_helper.py`
**Purpose:** Tokenization utilities (legacy, functionality moved to `data_processing.py`)

#### `utils.py`
**Purpose:** Utility functions for data splitting

**Key Functions:**
- `train_val_split(df, test_size, random_state)`: Stratified train/validation split

### Scripts

#### `train.py`
**Purpose:** Main training script with CLI interface

**Features:**
- Command-line arguments for hyperparameters
- Full training pipeline (load → split → tokenize → train → evaluate)
- Generates confusion matrix visualization
- Saves model and metrics to disk
- Progress logging throughout

**Example:**
```bash
python train.py --data dataset.csv --epochs 3 --batch-size 8 --output-dir ./outputs
```

#### `inference.py`
**Purpose:** Command-line interface for sentiment prediction

**Features:**
- Single text prediction (`--text`)
- Batch prediction from file (`--file`)
- Displays sentiment, confidence, and probabilities
- Supports both CPU and GPU

**Example:**
```bash
python inference.py --model-path final_model/ --text "This app is amazing!"
python inference.py --model-path final_model/ --file reviews.txt
```

#### `evaluate.py`
**Purpose:** Comprehensive model evaluation and visualization

**Features:**
- Confusion matrix heatmap
- ROC curve with AUC
- Precision-recall curve
- Class distribution comparison
- Confidence score distribution
- Detailed metrics report (accuracy, precision, recall, F1)
- Sample predictions export to CSV

**Example:**
```bash
python evaluate.py --model-path final_model/ --data dataset.csv --output-dir evaluation/
```

### Testing (`tests/`)

#### Unit Tests
- **`test_data_extraction.py`**: 7 tests for data loading (100% coverage)
- **`test_data_processing.py`**: 9 tests for tokenization and splitting (100% coverage)
- **`test_model.py`**: 11 tests for model architecture and training (100% coverage)
- **`test_inference.py`**: 10 tests for inference functions (100% coverage)
- **`test_tokenize.py`**: Tokenizer tests
- **`test_utils.py`**: 6 tests for utility functions (100% coverage)

#### Integration Tests
- **`test_train_integration.py`**: End-to-end pipeline test

**Coverage:** 100% (111/111 statements) across all modules

### CI/CD (`.github/workflows/`)

#### `ci.yml`
**Purpose:** Automated testing and quality checks

**Features:**
- Runs on every push and pull request
- Executes all unit tests
- Generates coverage reports
- Checks code quality (Black, Ruff)
- Fails if tests don't pass or coverage drops

### Documentation (`report/`)

#### `project_report.md`
**Purpose:** Comprehensive project documentation

**Contents:**
- Executive summary
- Methodology and approach
- Implementation details
- Testing strategy
- Results and metrics
- Challenges faced
- Future improvements
- Division of labor
- GitHub and Trello screenshots

## 📈 Model Performance

### Metrics

After training for 3 epochs on the full dataset:

| Metric      | Value  |
|-------------|--------|
| Accuracy    | ~0.92  |
| Precision   | ~0.91  |
| Recall      | ~0.92  |
| F1 Score    | ~0.91  |

### Training Details

- **Model**: BERT base uncased (110M parameters)
- **Optimizer**: AdamW
- **Learning rate**: 5e-5 (default from Trainer)
- **Batch size**: 8
- **Max sequence length**: 128 tokens
- **Training time**: ~30 minutes on GPU (varies by hardware)

## 🔧 Development

### Code Quality

The project enforces code quality through:

1. **Black**: Code formatting (line length: 100)
```bash
black src/ tests/
```

2. **Ruff**: Fast Python linter
```bash
ruff check src/ tests/
```

3. **Pre-commit hooks**: Automatic checks before commits
```bash
pre-commit run --all-files
```

### Testing Strategy

- **Unit tests**: Test individual functions and components
- **Integration tests**: Test end-to-end pipeline
- **Coverage achieved**: **100% code coverage** 🎯
  - All 6 modules: 100% coverage
  - 44 total tests (40 passed, 4 skipped)
  - See `COVERAGE_ACHIEVEMENT.md` for details
- **CI/CD**: Automated testing on push/PR





