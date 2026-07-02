import numpy as np
import pytest

from ml.src.metrics import compute_multiclass_metrics, expected_calibration_error


def test_perfect_predictions_have_perfect_metrics() -> None:
    targets = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[targets] * 0.9 + 0.1 / 3
    metrics = compute_multiclass_metrics(targets, probabilities, ["a", "b", "c"])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["top_3_accuracy"] == 1.0
    assert metrics["macro_roc_auc"] == 1.0
    assert all(values["specificity"] == 1.0 for values in metrics["per_class"].values())


def test_missing_class_has_null_auc() -> None:
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
        ]
    )
    metrics = compute_multiclass_metrics(targets, probabilities, ["a", "b", "c"])
    assert metrics["per_class"]["c"]["roc_auc"] is None
    assert metrics["per_class"]["c"]["support"] == 0


def test_calibration_error_is_bounded() -> None:
    targets = np.array([0, 1])
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])
    error = expected_calibration_error(probabilities, targets)
    assert 0.0 <= error <= 1.0


def test_rejects_probability_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="probabilities shape"):
        compute_multiclass_metrics(np.array([0, 1]), np.array([[0.5, 0.5]]), ["a", "b"])
