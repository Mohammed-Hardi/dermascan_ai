from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

from ml.src.augmentations import build_eval_transform
from ml.src.datasets import SkinConditionDataset
from ml.src.models import create_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODEL_CANDIDATES = [
    PROJECT_ROOT / "ml" / "outputs" / "models" / "dermascan-acne-eczema-psoriasis-custom-cnn.pt",
    PROJECT_ROOT / "ml" / "outputs" / "models" / "dermascan-acne-eczema-scabies-custom-cnn.pt",
    PROJECT_ROOT / "models" / "dermascan-acne-eczema-scabies-custom-cnn.pt",
]
DEFAULT_TEST_CSV_CANDIDATES = [
    PROJECT_ROOT / "ml" / "data" / "balanced_three_class" / "test.csv",
    PROJECT_ROOT / "ml" / "data" / "splits_three_class" / "test.csv",
]
DEFAULT_TEST_DIR = PROJECT_ROOT / "data" / "cleaned_3_classes" / "test"


class ImageFolderDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, root: Path, class_names: list[str], input_size: int) -> None:
        self.root = root
        self.class_names = class_names
        self.transform = build_eval_transform(input_size)
        self.samples: list[tuple[Path, int]] = []
        for label, class_name in enumerate(class_names):
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.glob("*")):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    self.samples.append((image_path, label))
        if not self.samples:
            raise RuntimeError(f"No test images found in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            return self.transform(image.convert("RGB")), label


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_checkpoint(model_path: Path, device: torch.device) -> tuple[torch.nn.Module, list[str], int, dict]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    required = {"model_name", "model_state_dict", "class_names", "input_size"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"Checkpoint is missing required fields: {sorted(missing)}")

    class_names = list(checkpoint["class_names"])
    input_size = int(checkpoint["input_size"])
    model = create_model(str(checkpoint["model_name"]), len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, input_size, checkpoint


def build_dataset(test_csv: Path | None, test_dir: Path, class_names: list[str], input_size: int) -> Dataset:
    if test_csv and test_csv.exists():
        return SkinConditionDataset(test_csv, build_eval_transform(input_size), PROJECT_ROOT)
    if test_dir.exists():
        return ImageFolderDataset(test_dir, class_names, input_size)
    raise FileNotFoundError(
        "No test dataset found. Expected data/cleaned_3_classes/test/ or one of the existing CSV test splits."
    )


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            batch_probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(batch_probabilities)
            predictions.append(np.argmax(batch_probabilities, axis=1))
            targets.append(labels.numpy())
    if not targets:
        raise RuntimeError("No test samples were processed.")
    return np.concatenate(targets), np.concatenate(predictions), np.concatenate(probabilities)


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
    normalized: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title(title)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks(np.arange(len(class_names)), labels=[name.title() for name in class_names], rotation=35, ha="right")
    axis.set_yticks(np.arange(len(class_names)), labels=[name.title() for name in class_names])

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = f"{value:.2f}" if normalized else f"{int(value)}"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "#071b4c",
                fontsize=10,
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def evaluate_model(model_path: Path, test_csv: Path | None, test_dir: Path, batch_size: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, input_size, checkpoint = load_checkpoint(model_path, device)
    dataset = build_dataset(test_csv, test_dir, class_names, input_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true, y_pred, _ = collect_predictions(model, loader, device)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    normalized_matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        normalize="true",
    )

    metrics = {
        "model_name": str(checkpoint.get("model_name", "unknown")),
        "checkpoint": model_path.as_posix(),
        "class_names": class_names,
        "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(report_dict).transpose().to_csv(RESULTS_DIR / "classification_report.csv")
    (RESULTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_confusion_matrix(matrix, class_names, RESULTS_DIR / "confusion_matrix.png", "DermaScan AI Confusion Matrix")
    save_confusion_matrix(
        normalized_matrix,
        class_names,
        RESULTS_DIR / "confusion_matrix_normalized.png",
        "DermaScan AI Normalized Confusion Matrix",
        normalized=True,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the DermaScan AI trained model.")
    parser.add_argument("--model", type=Path, default=None, help="Path to the trained .pt checkpoint.")
    parser.add_argument("--test-csv", type=Path, default=None, help="Optional CSV split to evaluate.")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR, help="Folder with class subfolders.")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model or first_existing(DEFAULT_MODEL_CANDIDATES)
    if model_path is None:
        print("ERROR: No trained model checkpoint found in the expected model folders.")
        return
    test_csv = args.test_csv or first_existing(DEFAULT_TEST_CSV_CANDIDATES)
    metrics = evaluate_model(model_path, test_csv, args.test_dir, args.batch_size)
    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to: {RESULTS_DIR / 'model_metrics.json'}")
    print(f"Classification report saved to: {RESULTS_DIR / 'classification_report.csv'}")
    print(f"Confusion matrix saved to: {RESULTS_DIR / 'confusion_matrix.png'}")
    print(f"Normalized confusion matrix saved to: {RESULTS_DIR / 'confusion_matrix_normalized.png'}")


if __name__ == "__main__":
    main()
