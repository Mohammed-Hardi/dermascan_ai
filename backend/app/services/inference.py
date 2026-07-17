import hashlib
import math
import random
from dataclasses import dataclass

from PIL import Image
import torch

from backend.app.config import Settings
from backend.app.models.model_loader import load_model
from backend.app.schemas import PredictionItem
from ml.src.augmentations import build_eval_transform


CLASS_NAMES = ["acne", "scabies", "psoriasis"]


@dataclass(slots=True)
class InferenceResult:
    top_prediction: PredictionItem | None
    top_k: list[PredictionItem]
    confidence_level: str
    risk_level: str
    model_version: str


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def predict_placeholder(image: Image.Image, image_bytes: bytes, settings: Settings) -> InferenceResult:
    """Return deterministic fake scores until a trained checkpoint is available."""
    digest = hashlib.sha256(image_bytes).digest()
    randomizer = random.Random(int.from_bytes(digest[:8], "big"))
    logits = [randomizer.uniform(-0.4, 0.4) for _ in CLASS_NAMES]
    logits[digest[8] % len(CLASS_NAMES)] += randomizer.uniform(1.4, 2.8)
    probabilities = _softmax(logits)

    ranked = sorted(zip(CLASS_NAMES, probabilities), key=lambda item: item[1], reverse=True)
    top_k = [
        PredictionItem(class_name=name, confidence=round(confidence, 4))
        for name, confidence in ranked[:3]
    ]
    top_confidence = top_k[0].confidence

    if top_confidence < settings.confidence_threshold:
        return InferenceResult(None, top_k, "uncertain", "uncertain", settings.model_version)
    if top_confidence < 0.75:
        confidence_level = "moderate"
    else:
        confidence_level = "confident"

    return InferenceResult(top_k[0], top_k, confidence_level, "low", settings.model_version)


def predict_checkpoint(image: Image.Image, settings: Settings) -> InferenceResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_model(settings.model_path, device, settings.allow_smoke_model)
    tensor = build_eval_transform(loaded.input_size)(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(loaded.model(tensor), dim=1)[0].cpu()
    top_count = min(3, len(loaded.class_names))
    confidences, indices = torch.topk(probabilities, k=top_count)
    top_k = [
        PredictionItem(
            class_name=loaded.class_names[int(class_index)],
            confidence=round(float(confidence), 4),
        )
        for confidence, class_index in zip(confidences, indices)
    ]
    top_confidence = top_k[0].confidence
    if top_confidence < settings.confidence_threshold:
        return InferenceResult(None, top_k, "uncertain", "uncertain", loaded.version)
    confidence_level = "moderate" if top_confidence < 0.75 else "confident"
    return InferenceResult(top_k[0], top_k, confidence_level, "low", loaded.version)


def predict(image: Image.Image, image_bytes: bytes, settings: Settings) -> InferenceResult:
    if settings.inference_mode == "placeholder":
        return predict_placeholder(image, image_bytes, settings)
    if settings.inference_mode == "checkpoint":
        return predict_checkpoint(image, settings)
    raise ValueError("DERMASCAN_INFERENCE_MODE must be 'placeholder' or 'checkpoint'.")
