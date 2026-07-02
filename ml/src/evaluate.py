from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from ml.src.augmentations import build_eval_transform
from ml.src.datasets import PROJECT_ROOT, SkinConditionDataset
from ml.src.metrics import compute_multiclass_metrics
from ml.src.models import create_model
from ml.src.train import load_config


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            logits = model(images.to(device, non_blocking=True))
            all_probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_targets.append(targets.numpy())
    if not all_targets:
        raise RuntimeError("No evaluation samples were processed.")
    return np.concatenate(all_targets), np.concatenate(all_probabilities)


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("DermaScan confusion matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_roc_curves(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary_targets = label_binarize(targets, classes=np.arange(len(class_names)))
    figure, axis = plt.subplots(figsize=(8, 7))
    plotted = 0
    for class_index, class_name in enumerate(class_names):
        class_targets = binary_targets[:, class_index]
        if np.unique(class_targets).size < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(
            class_targets,
            probabilities[:, class_index],
        )
        axis.plot(false_positive_rate, true_positive_rate, label=class_name.title())
        plotted += 1
    axis.plot([0, 1], [0, 1], linestyle="--", color="#7b8794", label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("One-vs-rest ROC curves")
    if plotted:
        axis.legend(loc="lower right")
    else:
        axis.text(0.5, 0.5, "Not enough class coverage for ROC curves", ha="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_classification_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["class_name", "precision", "recall", "f1_score", "support"]
    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for class_name, values in report.items():
            if not isinstance(values, dict) or "precision" not in values:
                continue
            writer.writerow(
                {
                    "class_name": class_name,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1_score": values["f1-score"],
                    "support": values["support"],
                }
            )


def evaluate(
    model_path: Path,
    config: dict[str, Any],
    split: str = "test",
    max_batches: int | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = list(checkpoint.get("class_names", config["class_names"]))
    input_size = int(checkpoint.get("input_size", config["input_size"]))
    model_name = str(checkpoint.get("model_name", config["model_name"]))
    model = create_model(model_name, len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    split_key = f"{split}_csv"
    if split_key not in config["data"]:
        raise ValueError(f"Unknown split '{split}'")
    dataset = SkinConditionDataset(
        PROJECT_ROOT / config["data"][split_key],
        build_eval_transform(input_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    targets, probabilities = collect_predictions(model, loader, device, max_batches)
    metrics = compute_multiclass_metrics(targets, probabilities, class_names)
    metrics.update(
        {
            "model_name": model_name,
            "checkpoint": model_path.as_posix(),
            "split": split,
            "device": str(device),
            "is_smoke_test": bool(checkpoint.get("is_smoke_test", False)),
            "full_split_evaluated": max_batches is None,
        }
    )

    report_dir = PROJECT_ROOT / config["outputs"]["report_dir"]
    plot_dir = PROJECT_ROOT / config["outputs"]["plot_dir"]
    prefix = "smoke-" if metrics["is_smoke_test"] else ""
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / f"{prefix}evaluation_metrics.json"
    classification_path = report_dir / f"{prefix}classification_report.csv"
    confusion_path = plot_dir / f"{prefix}confusion_matrix.png"
    roc_path = plot_dir / f"{prefix}roc_curve.png"

    classification_data = metrics.pop("classification_report")
    with metrics_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)
        file_handle.write("\n")
    write_classification_report(classification_data, classification_path)
    plot_confusion_matrix(np.asarray(metrics["confusion_matrix"]), class_names, confusion_path)
    plot_roc_curves(targets, probabilities, class_names, roc_path)

    print(json.dumps(metrics, indent=2))
    print(f"Metrics: {metrics_path}")
    print(f"Classification report: {classification_path}")
    print(f"Confusion matrix: {confusion_path}")
    print(f"ROC curves: {roc_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DermaScan model checkpoint.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.model, load_config(args.config), args.split, args.max_batches)


if __name__ == "__main__":
    main()
