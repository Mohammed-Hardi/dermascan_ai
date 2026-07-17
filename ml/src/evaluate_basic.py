from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.src.augmentations import build_eval_transform
from ml.src.datasets import PROJECT_ROOT, SkinConditionDataset
from ml.src.models import create_model
from ml.src.train import load_config


def compute_basic_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    class_count = len(class_names)
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    supports: list[int] = []
    for index, class_name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        true_negative = int(matrix.sum() - true_positive - false_positive - false_negative)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = int(matrix[index, :].sum())
        per_class[class_name] = {
            "precision": precision,
            "recall_sensitivity": recall,
            "specificity": specificity,
            "f1_score": f1_score,
            "support": support,
        }
        f1_values.append(f1_score)
        supports.append(support)

    sample_count = int(len(targets))
    weighted_f1 = float(np.average(f1_values, weights=supports))
    confidences = probabilities.max(axis=1)
    correctness = predictions == targets
    calibration_error = 0.0
    boundaries = np.linspace(0.0, 1.0, 11)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidences >= lower) & (confidences <= upper if upper == 1.0 else confidences < upper)
        if mask.any():
            calibration_error += float(mask.mean() * abs(correctness[mask].mean() - confidences[mask].mean()))

    return {
        "sample_count": sample_count,
        "accuracy": float(correctness.mean()),
        "macro_f1": float(np.mean(f1_values)),
        "weighted_f1": weighted_f1,
        "expected_calibration_error": calibration_error,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate(model_path: Path, config_path: Path, split: str = "test") -> dict[str, Any]:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = list(checkpoint["class_names"])
    input_size = int(checkpoint["input_size"])
    model_name = str(checkpoint["model_name"])
    model = create_model(model_name, len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    dataset = SkinConditionDataset(
        PROJECT_ROOT / config["data"][f"{split}_csv"],
        build_eval_transform(input_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )
    target_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
            target_batches.append(targets.numpy())

    targets = np.concatenate(target_batches)
    probabilities = np.concatenate(probability_batches)
    metrics = compute_basic_metrics(targets, probabilities, class_names)
    metrics.update(
        {
            "model_name": model_name,
            "checkpoint": model_path.as_posix(),
            "class_names": class_names,
            "split": split,
            "device": str(device),
            "is_smoke_test": bool(checkpoint.get("is_smoke_test", False)),
            "full_split_evaluated": True,
            "validation_accuracy": float(checkpoint.get("val_accuracy", 0.0)),
        }
    )

    report_dir = PROJECT_ROOT / config["outputs"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report_path = report_dir / "classification_report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=["class_name", "precision", "recall", "f1_score", "support"],
        )
        writer.writeheader()
        for class_name, values in metrics["per_class"].items():
            writer.writerow(
                {
                    "class_name": class_name,
                    "precision": values["precision"],
                    "recall": values["recall_sensitivity"],
                    "f1_score": values["f1_score"],
                    "support": values["support"],
                }
            )
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint without scikit-learn.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()
    evaluate(args.model, args.config, args.split)


if __name__ == "__main__":
    main()
