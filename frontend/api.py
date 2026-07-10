import os
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests
from PIL import Image


API_URL = os.getenv("DERMASCAN_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 45
LOCAL_FALLBACK_ENABLED = os.getenv("DERMASCAN_FRONTEND_FALLBACK", "true").lower() in {"1", "true", "yes"}
LOCAL_CLASSES = ["acne", "eczema", "psoriasis"]
LOCAL_SCANS: dict[str, dict[str, Any]] = {}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHECKPOINT_PATH = PROJECT_ROOT / "ml" / "outputs" / "models" / "dermascan-acne-eczema-psoriasis-efficientnet-b0.pt"
LOCAL_METRICS_PATH = PROJECT_ROOT / "results" / "model_metrics.json"

DISCLAIMER = (
    "The scan result is not a diagnosis. For accurate diagnosis and treatment "
    "recommendations, consult a qualified healthcare professional or dermatologist."
)

EXPLANATIONS = {
    "acne": "The image has visual features that may resemble acne, including blocked or inflamed pores and clustered bump-like patterns.",
    "eczema": "The image has visual features that may resemble eczema, including dry, itchy, cracked, or inflamed-looking patches.",
    "psoriasis": "The image has visual features that may resemble psoriasis, including raised, scaly, or plaque-like patches.",
}

URGENT_WARNING = "Seek urgent medical care if the rash is rapidly spreading, painful, bleeding, infected, or associated with fever."


class ApiError(RuntimeError):
    pass


def _fallback_model_info() -> dict[str, Any]:
    metrics = _load_local_metrics()
    checkpoint_available = LOCAL_CHECKPOINT_PATH.exists()
    return {
        "model_name": metrics.get("model_name", "dermascan-local-checkpoint"),
        "version": "streamlit-local-checkpoint",
        "classes": metrics.get("class_names", LOCAL_CLASSES),
        "input_size": [160, 160],
        "is_placeholder": False,
        "is_smoke_test": False,
        "model_available": checkpoint_available,
        "inference_mode": "streamlit_local_checkpoint" if checkpoint_available else "checkpoint_missing",
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "validation_accuracy": metrics.get("validation_accuracy"),
            "weighted_f1": metrics.get("weighted_f1"),
        },
        "last_trained": None,
    }


def _load_local_metrics() -> dict[str, Any]:
    if not LOCAL_METRICS_PATH.exists():
        return {}
    try:
        import json

        return json.loads(LOCAL_METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=1)
def _load_local_checkpoint() -> dict[str, Any]:
    if not LOCAL_CHECKPOINT_PATH.exists():
        raise ApiError("The trained model checkpoint is missing from this deployment.")
    try:
        import torch

        from ml.src.models import create_model
    except ModuleNotFoundError as exc:
        raise ApiError("The local model dependencies are not installed in this Streamlit deployment.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(LOCAL_CHECKPOINT_PATH, map_location=device, weights_only=False)
    class_names = list(checkpoint["class_names"])
    model = create_model(str(checkpoint["model_name"]), len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return {
        "model": model,
        "device": device,
        "class_names": class_names,
        "input_size": int(checkpoint.get("input_size", 160)),
        "version": f"{checkpoint.get('model_name', 'custom_cnn')}-epoch-{checkpoint.get('epoch', 'unknown')}",
    }


def _fallback_prediction(image_bytes: bytes, filename: str, content_type: str, metadata: dict[str, str]) -> dict[str, Any]:
    try:
        from backend.app.config import get_settings
        from backend.app.services.explanation import build_explanation
        from backend.app.services.image_quality import validate_image
        from ml.src.augmentations import build_eval_transform
        import torch
    except Exception as exc:
        raise ApiError("The local analysis service is not available in this deployment.") from exc

    try:
        validated = validate_image(image_bytes, get_settings())
    except ValueError as exc:
        raise ApiError(str(exc)) from exc
    if not validated.quality.is_acceptable:
        raise ApiError(validated.quality.reason or "The image quality is not acceptable for analysis.")

    loaded = _load_local_checkpoint()
    tensor = build_eval_transform(loaded["input_size"])(validated.image).unsqueeze(0).to(loaded["device"])
    with torch.inference_mode():
        probabilities_tensor = torch.softmax(loaded["model"](tensor), dim=1)[0].cpu()
    confidences, indices = torch.topk(probabilities_tensor, k=min(3, len(loaded["class_names"])))
    top_k = [
        {
            "class_name": loaded["class_names"][int(class_index)],
            "confidence": round(float(confidence), 4),
        }
        for confidence, class_index in zip(confidences, indices)
    ]
    top_prediction = top_k[0] if top_k and top_k[0]["confidence"] >= 0.60 else None
    scan_id = str(uuid4())
    explanation = build_explanation(
        top_prediction["class_name"] if top_prediction else None,
        top_prediction is None,
    )
    result = {
        "scan_id": scan_id,
        "status": "success",
        "top_prediction": top_prediction,
        "top_k": top_k,
        "confidence_level": "uncertain" if top_prediction is None else ("moderate" if top_prediction["confidence"] < 0.75 else "confident"),
        "risk_level": "uncertain" if top_prediction is None else "low",
        "explanation": explanation.model_dump(),
        "disclaimer": DISCLAIMER,
        "model_version": loaded["version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    LOCAL_SCANS[scan_id] = {
        "result": result,
        "filename": filename,
        "content_type": content_type,
        "metadata": metadata,
        "image_bytes": validated.image_bytes,
    }
    return result


def _fallback_report(scan_id: str) -> bytes:
    record = LOCAL_SCANS.get(scan_id)
    if record is None:
        raise ApiError("The report could not be generated because the scan has expired.")
    result = record["result"]
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ModuleNotFoundError as exc:
        raise ApiError("The PDF report service is unavailable in this deployment.") from exc

    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=letter)
    pdf.setTitle("DermaScan AI Educational Report")
    pdf.drawString(72, 740, "DermaScan AI Educational Report")
    top_prediction = result.get("top_prediction")
    if top_prediction:
        pdf.drawString(72, 710, f"Possible condition: {top_prediction['class_name'].title()}")
        pdf.drawString(72, 690, f"Confidence score: {top_prediction['confidence'] * 100:.1f}%")
    else:
        pdf.drawString(72, 710, "Possible condition: Uncertain")
        pdf.drawString(72, 690, "Confidence score: below decision threshold")
    pdf.drawString(72, 660, "Model probabilities:")
    y = 640
    for item in result["top_k"]:
        pdf.drawString(90, y, f"{item['class_name'].title()}: {item['confidence'] * 100:.1f}%")
        y -= 20
    pdf.drawString(72, y - 15, "Important:")
    text = pdf.beginText(72, y - 35)
    text.textLines(DISCLAIMER)
    pdf.drawText(text)
    pdf.save()
    return output.getvalue()


def health_check() -> bool:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.ok
    except requests.RequestException:
        return LOCAL_FALLBACK_ENABLED and LOCAL_CHECKPOINT_PATH.exists()


def get_model_info() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}/api/model-info", timeout=5)
        return response.json() if response.ok else None
    except (requests.RequestException, ValueError):
        return _fallback_model_info() if LOCAL_FALLBACK_ENABLED else None


def predict_image(
    image_bytes: bytes,
    filename: str,
    content_type: str,
    metadata: dict[str, str],
) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{API_URL}/api/predict",
            files={"image": (filename, image_bytes, content_type)},
            data=metadata,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        if LOCAL_FALLBACK_ENABLED:
            return _fallback_prediction(image_bytes, filename, content_type, metadata)
        raise ApiError("The analysis service is unavailable. Check that the backend is running.") from exc

    if not response.ok:
        try:
            detail = response.json().get("detail", "The image could not be analyzed.")
        except ValueError:
            detail = "The image could not be analyzed."
        raise ApiError(str(detail))
    return response.json()


def download_report(scan_id: str) -> bytes:
    try:
        response = requests.get(f"{API_URL}/api/report/{scan_id}", timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        if LOCAL_FALLBACK_ENABLED:
            return _fallback_report(scan_id)
        raise ApiError("The PDF report service is unavailable.") from exc
    if not response.ok:
        raise ApiError("The report could not be generated or the scan has expired.")
    return response.content
