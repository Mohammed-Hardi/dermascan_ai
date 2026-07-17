import json
import csv
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
import torch

from backend.app.config import PROJECT_ROOT, Settings, get_settings
from backend.app.models.model_loader import ModelUnavailableError, load_model
from backend.app.schemas import ModelInfo
from backend.app.services.inference import CLASS_NAMES


router = APIRouter(tags=["model"])


def _read_json_metrics(path: Path, classes: list[str]) -> dict[str, float | None] | None:
    if not path.exists():
        return None
    try:
        saved_metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    per_class = saved_metrics.get("per_class")
    if isinstance(per_class, dict) and set(per_class) != set(classes):
        return None
    saved_classes = saved_metrics.get("class_names")
    if isinstance(saved_classes, list) and set(saved_classes) != set(classes):
        return None

    return {
        "accuracy": saved_metrics.get("accuracy"),
        "validation_accuracy": saved_metrics.get("validation_accuracy"),
        "weighted_f1": saved_metrics.get("weighted_f1"),
    }


def _read_training_metrics(path: Path) -> dict[str, float | None] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as file_handle:
            rows = list(csv.DictReader(file_handle))
    except OSError:
        return None
    if not rows:
        return None

    def as_float(row: dict[str, str], key: str) -> float | None:
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return None

    best_row = min(rows, key=lambda row: as_float(row, "val_loss") or float("inf"))
    last_row = rows[-1]
    return {
        "accuracy": as_float(best_row, "val_accuracy"),
        "validation_accuracy": as_float(best_row, "val_accuracy"),
        "weighted_f1": None,
    }


def _load_model_metrics(classes: list[str]) -> dict[str, float | None]:
    empty_metrics: dict[str, float | None] = {
        "accuracy": None,
        "validation_accuracy": None,
        "weighted_f1": None,
    }
    reports_dir = PROJECT_ROOT / "ml" / "outputs" / "reports"
    candidate_metric_paths = [
        PROJECT_ROOT / "results" / "model_metrics.json",
        reports_dir / "acne_scabies_psoriasis" / "evaluation_metrics.json",
        reports_dir / "acne_eczema_psoriasis" / "evaluation_metrics.json",
        reports_dir / "evaluation_metrics.json",
        reports_dir / "three_class" / "evaluation_metrics.json",
    ]
    for metrics_path in candidate_metric_paths:
        metrics = _read_json_metrics(metrics_path, classes)
        if metrics is not None:
            return metrics

    training_metrics = _read_training_metrics(reports_dir / "acne_scabies_psoriasis" / "training_log.csv")
    return training_metrics or empty_metrics


@router.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    settings: Settings = get_settings()
    model_name = settings.model_name
    version = settings.model_version
    classes = CLASS_NAMES
    input_size = 224
    is_placeholder = settings.inference_mode == "placeholder"
    is_smoke_test = False
    model_available = True
    last_trained = None
    if not is_placeholder:
        try:
            loaded = load_model(
                settings.model_path,
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                settings.allow_smoke_model,
            )
            model_name = loaded.model_name
            version = loaded.version
            classes = loaded.class_names
            input_size = loaded.input_size
            is_smoke_test = loaded.is_smoke_test
            last_trained = datetime.fromtimestamp(
                loaded.checkpoint_path.stat().st_mtime,
                tz=timezone.utc,
            )
        except ModelUnavailableError:
            model_available = False

    metrics = _load_model_metrics(classes)
    return ModelInfo(
        model_name=model_name,
        version=version,
        classes=classes,
        input_size=[input_size, input_size],
        is_placeholder=is_placeholder,
        is_smoke_test=is_smoke_test,
        model_available=model_available,
        inference_mode=settings.inference_mode,
        metrics=metrics,
        last_trained=last_trained,
    )
