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

## 🎯 Usage

### Training the Model

Train a sentiment classifier with default settings:

```bash
python train.py
```

With custom parameters:

```bash
python train.py --data dataset.csv \
                --model-name bert-base-uncased \
                --epochs 3 \
                --batch-size 8 \
                --max-length 128 \
                --output-dir ./outputs
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
- `outputs/final_model/`: Saved model checkpoint
- `outputs/metrics.txt`: Performance metrics
- `outputs/confusion_matrix.png`: Confusion matrix visualization

### Running Inference

Predict sentiment for a single text:

```bash
python inference.py --text "This app is amazing!"
```

Batch prediction from file:

```bash
# Create a file with one review per line
echo "Great app, highly recommend" > samples.txt
echo "Terrible experience, waste of money" >> samples.txt

python inference.py --file samples.txt
```

### Running Tests

Run all tests:

```bash
pytest tests/unit/ -v
```

Run with coverage report (**100% coverage achieved!** 🎉):

```bash
pytest tests/unit/ --cov=src --cov-report=term-missing --cov-report=html
# Coverage: 100% (111/111 statements)
# 40 passed, 4 skipped
```

Run specific test categories:

```bash
# Unit tests only
pytest tests/unit/test_data_extraction.py -v

# Integration tests
pytest tests/unit/test_train_integration.py -v
```

## 🏗️ Project Structure

```
sentiment-bert-collab/
├── src/
│   ├── data_extraction.py      # Data loading and preprocessing
│   ├── model.py                # Model definition and training
│   ├── tokenize_helper.py      # Tokenization utilities
│   └── utils.py                # Helper functions
├── tests/
│   ├── fixtures/               # Test data
│   └── unit/                   # Unit tests
│       ├── test_data_extraction.py
│       ├── test_model.py
│       ├── test_tokenize.py
│       ├── test_utils.py
│       └── test_train_integration.py
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── report/
│   └── project_report.md       # Detailed project documentation
├── train.py                    # Main training script
├── inference.py                # Inference script
├── dataset.csv                 # Training data
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

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





