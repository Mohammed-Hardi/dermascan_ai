from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 10,
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = predictions == targets
    boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not mask.any():
            continue
        bin_accuracy = correctness[mask].mean()
        bin_confidence = confidences[mask].mean()
        error += float(mask.mean() * abs(bin_accuracy - bin_confidence))
    return error


def compute_multiclass_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    if targets.ndim != 1:
        raise ValueError("targets must be a one-dimensional array")
    if probabilities.shape != (len(targets), len(class_names)):
        raise ValueError("probabilities shape must be [samples, classes]")
    if len(targets) == 0:
        raise ValueError("at least one sample is required")

    labels = np.arange(len(class_names))
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=labels)

    per_class: dict[str, dict[str, float | int | None]] = {}
    binary_targets = label_binarize(targets, classes=labels)
    valid_auc_values: list[float] = []
    for class_index, class_name in enumerate(class_names):
        true_positive = int(matrix[class_index, class_index])
        false_positive = int(matrix[:, class_index].sum() - true_positive)
        false_negative = int(matrix[class_index, :].sum() - true_positive)
        true_negative = int(matrix.sum() - true_positive - false_positive - false_negative)
        specificity_denominator = true_negative + false_positive
        specificity = (
            true_negative / specificity_denominator if specificity_denominator else 0.0
        )
        class_targets = binary_targets[:, class_index]
        auc_value: float | None = None
        if np.unique(class_targets).size == 2:
            auc_value = float(roc_auc_score(class_targets, probabilities[:, class_index]))
            valid_auc_values.append(auc_value)
        per_class[class_name] = {
            "precision": float(precision[class_index]),
            "recall_sensitivity": float(recall[class_index]),
            "specificity": float(specificity),
            "f1_score": float(f1[class_index]),
            "support": int(support[class_index]),
            "roc_auc": auc_value,
        }

    top_k = min(3, len(class_names))
    top_predictions = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    top_k_accuracy = float(
        np.mean([target in candidates for target, candidates in zip(targets, top_predictions)])
    )
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "sample_count": int(len(targets)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "top_3_accuracy": top_k_accuracy,
        "macro_roc_auc": float(np.mean(valid_auc_values)) if valid_auc_values else None,
        "expected_calibration_error": expected_calibration_error(probabilities, targets),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
