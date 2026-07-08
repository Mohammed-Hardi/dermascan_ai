from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def add_scalars(self, *args: object, **kwargs: object) -> None:
            pass

        def add_scalar(self, *args: object, **kwargs: object) -> None:
            pass

        def close(self) -> None:
            pass

from ml.src.augmentations import build_eval_transform, build_train_transform
from ml.src.datasets import PROJECT_ROOT, SkinConditionDataset
from ml.src.models import create_model


@dataclass(slots=True)
class EpochMetrics:
    loss: float
    accuracy: float
    samples: int


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)
    required = {
        "model_name",
        "checkpoint_name",
        "num_classes",
        "class_names",
        "input_size",
        "batch_size",
        "epochs",
        "learning_rate",
        "data",
        "outputs",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing training configuration keys: {sorted(missing)}")
    if len(config["class_names"]) != int(config["num_classes"]):
        raise ValueError("class_names length must equal num_classes")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total = len(labels)
    weights = [
        total / (num_classes * counts[class_index]) if counts[class_index] else 0.0
        for class_index in range(num_classes)
    ]
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, SkinConditionDataset]:
    input_size = int(config["input_size"])
    train_dataset = SkinConditionDataset(
        PROJECT_ROOT / config["data"]["train_csv"],
        build_train_transform(input_size, config.get("augmentations")),
    )
    val_dataset = SkinConditionDataset(
        PROJECT_ROOT / config["data"]["val_csv"],
        build_eval_transform(input_size),
    )
    loader_options = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config.get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    return train_loader, val_loader, train_dataset


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("No samples were processed during the epoch.")
    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        samples=total_samples,
    )


def _write_log(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(config: dict[str, Any], smoke_test: bool = False) -> Path:
    active_config = deepcopy(config)
    max_train_batches = None
    max_val_batches = None
    if smoke_test:
        active_config["epochs"] = 1
        active_config["batch_size"] = 4
        active_config["input_size"] = 96
        active_config["pretrained"] = False
        max_train_batches = 2
        max_val_batches = 2

    seed = int(active_config.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_dataset = build_dataloaders(active_config)
    model = create_model(
        active_config["model_name"],
        int(active_config["num_classes"]),
        bool(active_config.get("pretrained", True)),
    ).to(device)

    class_weights = None
    if active_config.get("use_class_weights", True):
        class_weights = compute_class_weights(train_dataset.labels, int(active_config["num_classes"])).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=float(active_config["learning_rate"]),
        weight_decay=float(active_config.get("weight_decay", 0.0)),
    )
    epochs = int(active_config["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    model_dir = PROJECT_ROOT / active_config["outputs"]["model_dir"]
    report_dir = PROJECT_ROOT / active_config["outputs"]["report_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = active_config["checkpoint_name"]
    log_name = "training_log.csv"
    run_name = active_config["model_name"]
    if smoke_test:
        checkpoint_name = f"smoke-{checkpoint_name}"
        log_name = "smoke-training_log.csv"
        run_name = f"smoke-{run_name}"
    checkpoint_path = model_dir / checkpoint_name
    log_path = report_dir / log_name
    writer = SummaryWriter(log_dir=report_dir / "tensorboard" / run_name)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = int(active_config.get("early_stopping_patience", 5))
    history: list[dict[str, float | int]] = []

    print(f"Training {active_config['model_name']} on {device} with {len(train_dataset)} training images")
    try:
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                max_batches=max_train_batches,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                max_batches=max_val_batches,
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch,
                "train_loss": round(train_metrics.loss, 6),
                "train_accuracy": round(train_metrics.accuracy, 6),
                "val_loss": round(val_metrics.loss, 6),
                "val_accuracy": round(val_metrics.accuracy, 6),
                "learning_rate": learning_rate,
            }
            history.append(row)
            _write_log(log_path, history)
            writer.add_scalars("loss", {"train": train_metrics.loss, "val": val_metrics.loss}, epoch)
            writer.add_scalars(
                "accuracy", {"train": train_metrics.accuracy, "val": val_metrics.accuracy}, epoch
            )
            writer.add_scalar("learning_rate", learning_rate, epoch)

            print(
                f"Epoch {epoch}/{epochs} | train loss {train_metrics.loss:.4f} | "
                f"val loss {val_metrics.loss:.4f} | val accuracy {val_metrics.accuracy:.4f}"
            )
            if val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_name": active_config["model_name"],
                        "model_state_dict": model.state_dict(),
                        "class_names": active_config["class_names"],
                        "input_size": active_config["input_size"],
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "val_accuracy": val_metrics.accuracy,
                        "is_smoke_test": smoke_test,
                        "config": active_config,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
            scheduler.step()
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs")
                break
    finally:
        writer.close()

    metadata_path = report_dir / ("smoke-training_metadata.json" if smoke_test else "training_metadata.json")
    with metadata_path.open("w", encoding="utf-8") as file_handle:
        json.dump(
            {
                "checkpoint": checkpoint_path.as_posix(),
                "device": str(device),
                "best_val_loss": best_val_loss,
                "epochs_completed": len(history),
                "is_smoke_test": smoke_test,
            },
            file_handle,
            indent=2,
        )
        file_handle.write("\n")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DermaScan image classifier.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = train(load_config(args.config), smoke_test=args.smoke_test)
    print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
