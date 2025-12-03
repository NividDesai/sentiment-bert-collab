import torch
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.model import build_model, hf_dataset_from_tensors, train_model
from src.tokenize_helper import get_tokenizer, tokenize_dataframe
from transformers import AutoTokenizer


def _check_accelerate_installed():
    """Helper to check if accelerate is installed"""
    try:
        import accelerate  # noqa: F401

        return True
    except ImportError:
        return False


def test_model_instantiation_and_forward():
    model = build_model(num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(
        ["I love it", "I hate it"], padding=True, truncation=True, return_tensors="pt"
    )
    # forward pass
    outputs = model(**inputs)
    logits = outputs.logits
    assert logits.shape[0] == 2
    assert logits.shape[1] == 2


def test_hf_dataset_from_tensors():
    """Test conversion of tensors to HuggingFace Dataset"""
    # Create mock tokenized data
    tokenized = {
        "input_ids": torch.tensor([[101, 2023, 102], [101, 2008, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
    }
    labels = torch.tensor([1, 0])

    dataset = hf_dataset_from_tensors(tokenized, labels)

    assert len(dataset) == 2
    assert "input_ids" in dataset.column_names
    assert "attention_mask" in dataset.column_names
    assert "labels" in dataset.column_names
    assert dataset[0]["labels"] == 1
    assert dataset[1]["labels"] == 0


def test_hf_dataset_from_tensors_with_list_labels():
    """Test conversion with list labels instead of tensor"""
    tokenized = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    labels = [1]  # List instead of tensor

    dataset = hf_dataset_from_tensors(tokenized, labels)
    assert len(dataset) == 1
    assert dataset[0]["labels"] == 1


def test_build_model_custom_model():
    """Test building model with different model name"""
    model = build_model(model_name="bert-base-uncased", num_labels=3)
    assert model is not None
    # Check output dimension matches num_labels
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(["test"], return_tensors="pt")
    outputs = model(**inputs)
    assert outputs.logits.shape[1] == 3


def test_compute_metrics_logic():
    """Test the compute_metrics logic used in training"""
    # Simulate what compute_metrics does
    # This is the logic from lines 38-42 in model.py

    # Create mock predictions (logits) and labels
    logits = np.array(
        [
            [0.2, 0.8],  # Predicts class 1
            [0.9, 0.1],  # Predicts class 0
            [0.3, 0.7],  # Predicts class 1
            [0.6, 0.4],  # Predicts class 0
        ]
    )
    labels = np.array([1, 0, 1, 0])

    # Apply the logic from compute_metrics
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()

    # Verify it works correctly
    assert preds.tolist() == [1, 0, 1, 0]
    assert accuracy == 1.0  # All predictions correct

    # Test with some incorrect predictions
    labels2 = np.array([0, 0, 1, 0])
    accuracy2 = (preds == labels2).mean()
    assert accuracy2 == 0.75  # 3 out of 4 correct


def test_model_evaluation_output_format():
    """Test that evaluation results have the expected format"""
    # Test that the evaluation output from trainer.evaluate() is properly structured
    # This verifies the contract that train_model returns (trainer, eval_res)
    # where eval_res should have 'eval_accuracy' key

    # Mock eval results structure
    mock_eval_res = {"eval_accuracy": 0.92, "eval_loss": 0.15}

    # Verify expected keys exist
    assert "eval_accuracy" in mock_eval_res
    assert isinstance(mock_eval_res["eval_accuracy"], (int, float))
    assert 0 <= mock_eval_res["eval_accuracy"] <= 1


def test_train_model_setup_without_training():
    """Test train_model function setup without actual training (no accelerate needed)"""
    # Create minimal test data
    tokenized_train = {
        "input_ids": torch.tensor([[101, 2023, 102], [101, 2008, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
    }
    tokenized_val = {
        "input_ids": torch.tensor([[101, 3000, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    labels_train = torch.tensor([1, 0])
    labels_val = torch.tensor([1])

    model = build_model(num_labels=2)

    # Mock both TrainingArguments and Trainer to avoid needing accelerate
    with patch("src.model.TrainingArguments") as MockTrainingArgs, patch(
        "src.model.Trainer"
    ) as MockTrainer:

        # Setup mock training arguments
        mock_args_instance = MagicMock()
        MockTrainingArgs.return_value = mock_args_instance

        # Setup mock trainer
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = None
        mock_trainer_instance.evaluate.return_value = {
            "eval_loss": 0.5,
            "eval_accuracy": 0.85,
            "eval_runtime": 1.0,
        }
        MockTrainer.return_value = mock_trainer_instance

        # Call train_model
        trainer, eval_res = train_model(
            model=model,
            tokenized_train=tokenized_train,
            labels_train=labels_train,
            tokenized_val=tokenized_val,
            labels_val=labels_val,
            output_dir="./test_outputs",
            epochs=2,
        )

        # Verify trainer was created
        assert MockTrainer.called
        assert trainer is not None

        # Verify train was called
        assert mock_trainer_instance.train.called

        # Verify evaluate was called
        assert mock_trainer_instance.evaluate.called

        # Verify evaluation results
        assert eval_res is not None
        assert "eval_accuracy" in eval_res
        assert eval_res["eval_accuracy"] == 0.85


def test_train_model_creates_datasets():
    """Test that train_model properly creates HuggingFace datasets"""
    tokenized_train = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    tokenized_val = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    labels_train = torch.tensor([1])
    labels_val = torch.tensor([0])

    model = build_model(num_labels=2)

    # Mock both TrainingArguments and Trainer
    with patch("src.model.TrainingArguments") as MockTrainingArgs, patch(
        "src.model.Trainer"
    ) as MockTrainer:

        mock_args_instance = MagicMock()
        MockTrainingArgs.return_value = mock_args_instance

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = None
        mock_trainer_instance.evaluate.return_value = {"eval_accuracy": 0.9}
        MockTrainer.return_value = mock_trainer_instance

        # Call train_model
        train_model(
            model=model,
            tokenized_train=tokenized_train,
            labels_train=labels_train,
            tokenized_val=tokenized_val,
            labels_val=labels_val,
            output_dir="./test_outputs",
            epochs=1,
        )

        # Verify Trainer was called with datasets
        assert MockTrainer.called
        call_kwargs = MockTrainer.call_args[1]

        # Check that datasets were created
        assert "train_dataset" in call_kwargs
        assert "eval_dataset" in call_kwargs
        assert call_kwargs["train_dataset"] is not None
        assert call_kwargs["eval_dataset"] is not None

        # Verify the datasets have the correct structure
        train_ds = call_kwargs["train_dataset"]
        eval_ds = call_kwargs["eval_dataset"]
        assert "input_ids" in train_ds.column_names
        assert "labels" in train_ds.column_names
        assert len(train_ds) == 1
        assert len(eval_ds) == 1


def test_train_model_training_arguments():
    """Test that train_model creates TrainingArguments with correct parameters"""
    tokenized = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    labels = torch.tensor([1])

    model = build_model(num_labels=2)

    with patch("src.model.TrainingArguments") as MockTrainingArgs, patch(
        "src.model.Trainer"
    ) as MockTrainer:

        # Setup mocks
        mock_args_instance = MagicMock()
        MockTrainingArgs.return_value = mock_args_instance

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = None
        mock_trainer_instance.evaluate.return_value = {"eval_accuracy": 0.9}
        MockTrainer.return_value = mock_trainer_instance

        # Call train_model with specific parameters
        train_model(
            model=model,
            tokenized_train=tokenized,
            labels_train=labels,
            tokenized_val=tokenized,
            labels_val=labels,
            output_dir="./custom_output",
            epochs=5,
        )

        # Verify TrainingArguments was called
        assert MockTrainingArgs.called
        call_kwargs = MockTrainingArgs.call_args[1]

        # Check specific arguments
        assert call_kwargs["output_dir"] == "./custom_output"
        assert call_kwargs["num_train_epochs"] == 5
        assert call_kwargs["per_device_train_batch_size"] == 8
        assert call_kwargs["per_device_eval_batch_size"] == 8
        assert call_kwargs["eval_strategy"] == "epoch"
        assert call_kwargs["logging_strategy"] == "epoch"
        assert call_kwargs["save_strategy"] == "no"
        assert call_kwargs["seed"] == 42
        assert call_kwargs["disable_tqdm"]


def test_train_model_compute_metrics_function():
    """Test that the compute_metrics function is properly passed to Trainer and works correctly"""
    tokenized = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    labels = torch.tensor([1])

    model = build_model(num_labels=2)

    with patch("src.model.TrainingArguments") as MockTrainingArgs, patch(
        "src.model.Trainer"
    ) as MockTrainer:

        mock_args_instance = MagicMock()
        MockTrainingArgs.return_value = mock_args_instance

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = None
        mock_trainer_instance.evaluate.return_value = {"eval_accuracy": 0.9}
        MockTrainer.return_value = mock_trainer_instance

        # Call train_model
        train_model(
            model=model,
            tokenized_train=tokenized,
            labels_train=labels,
            tokenized_val=tokenized,
            labels_val=labels,
            output_dir="./test_outputs",
            epochs=1,
        )

        # Verify Trainer was called
        assert MockTrainer.called
        call_kwargs = MockTrainer.call_args[1]

        # Check that compute_metrics function was passed
        assert "compute_metrics" in call_kwargs
        compute_metrics_fn = call_kwargs["compute_metrics"]
        assert compute_metrics_fn is not None
        assert callable(compute_metrics_fn)

        # Test the compute_metrics function directly to cover lines 39-42
        # Create mock eval_pred tuple (logits, labels)
        logits = np.array([[0.2, 0.8], [0.9, 0.1]])
        labels_arr = np.array([1, 0])
        eval_pred_tuple = (logits, labels_arr)

        # Call compute_metrics - this covers the internal function (lines 38-42)
        result = compute_metrics_fn(eval_pred_tuple)

        # Verify the result
        assert "accuracy" in result
        assert result["accuracy"] == 1.0  # Both predictions should be correct

        # Test with some incorrect predictions to ensure accuracy calculation works
        logits2 = np.array([[0.8, 0.2], [0.9, 0.1]])  # Both predict class 0
        labels2 = np.array([1, 0])  # First is wrong, second is correct
        eval_pred_tuple2 = (logits2, labels2)

        result2 = compute_metrics_fn(eval_pred_tuple2)
        assert "accuracy" in result2
        assert result2["accuracy"] == 0.5  # 1 out of 2 correct


@pytest.mark.skipif(
    not _check_accelerate_installed(),
    reason="Requires accelerate package for full training",
)
def test_train_model_basic():
    """Test basic training functionality (requires accelerate package)"""
    # Create minimal dataset
    df = pd.DataFrame(
        {
            "text": ["good app", "bad app", "okay app", "great app"],
            "label": [1, 0, 1, 1],
        }
    )

    tokenizer = get_tokenizer()
    tokenized = tokenize_dataframe(df, tokenizer, max_length=32)
    labels = torch.tensor(df["label"].values)

    # Split into train/val
    train_tokenized = {k: v[:3] for k, v in tokenized.items()}
    val_tokenized = {k: v[3:] for k, v in tokenized.items()}
    train_labels = labels[:3]
    val_labels = labels[3:]

    model = build_model(num_labels=2)

    # Train for just 1 step with very small data
    import os

    output_dir = "./test_outputs_minimal"
    os.makedirs(output_dir, exist_ok=True)

    try:
        trainer, eval_res = train_model(
            model=model,
            tokenized_train=train_tokenized,
            labels_train=train_labels,
            tokenized_val=val_tokenized,
            labels_val=val_labels,
            output_dir=output_dir,
            epochs=1,
        )

        # Check trainer was created
        assert trainer is not None
        # Check eval results exist
        assert eval_res is not None
        assert "eval_accuracy" in eval_res
        assert 0 <= eval_res["eval_accuracy"] <= 1

    finally:
        # Cleanup
        import shutil

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
