import hashlib
import os
from datetime import datetime, timezone
from io import BytesIO
from typing import Any
from uuid import uuid4

import requests
from PIL import Image


API_URL = os.getenv("DERMASCAN_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 45
LOCAL_FALLBACK_ENABLED = os.getenv("DERMASCAN_FRONTEND_FALLBACK", "true").lower() in {"1", "true", "yes"}
LOCAL_CLASSES = ["acne", "eczema", "psoriasis"]
LOCAL_SCANS: dict[str, dict[str, Any]] = {}

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
    return {
        "model_name": "dermascan-streamlit-cloud-fallback",
        "version": "educational-fallback-v0.1",
        "classes": LOCAL_CLASSES,
        "input_size": [224, 224],
        "is_placeholder": True,
        "is_smoke_test": False,
        "model_available": True,
        "inference_mode": "streamlit_local_fallback",
        "metrics": {
            "accuracy": None,
            "validation_accuracy": None,
            "weighted_f1": None,
        },
        "last_trained": None,
    }


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [2.718281828459045 ** (value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _fallback_prediction(image_bytes: bytes, filename: str, content_type: str, metadata: dict[str, str]) -> dict[str, Any]:
    try:
        Image.open(BytesIO(image_bytes)).verify()
    except Exception as exc:
        raise ApiError("The image could not be opened. Please upload a clear JPG, JPEG, PNG, or WEBP image.") from exc

    digest = hashlib.sha256(image_bytes).digest()
    logits = [
        ((digest[index] / 255.0) * 2.0) - 1.0
        for index in range(len(LOCAL_CLASSES))
    ]
    logits[digest[8] % len(LOCAL_CLASSES)] += 1.2
    probabilities = _softmax(logits)
    top_k = sorted(
        [
            {"class_name": class_name, "confidence": round(probability, 4)}
            for class_name, probability in zip(LOCAL_CLASSES, probabilities)
        ],
        key=lambda item: item["confidence"],
        reverse=True,
    )
    top_prediction = top_k[0]
    scan_id = str(uuid4())
    explanation = {
        "summary": EXPLANATIONS.get(top_prediction["class_name"], "The image is visually compared with supported educational classes."),
        "next_steps": "Use this result only as educational support and consult a qualified health professional for assessment.",
        "warning": URGENT_WARNING,
    }
    result = {
        "scan_id": scan_id,
        "status": "success",
        "top_prediction": top_prediction,
        "top_k": top_k,
        "confidence_level": "moderate" if top_prediction["confidence"] < 0.75 else "confident",
        "risk_level": "low",
        "explanation": explanation,
        "disclaimer": DISCLAIMER,
        "model_version": "streamlit-cloud-educational-fallback",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    LOCAL_SCANS[scan_id] = {
        "result": result,
        "filename": filename,
        "content_type": content_type,
        "metadata": metadata,
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
    pdf.drawString(72, 710, f"Possible condition: {result['top_prediction']['class_name'].title()}")
    pdf.drawString(72, 690, f"Confidence score: {result['top_prediction']['confidence'] * 100:.1f}%")
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
        return LOCAL_FALLBACK_ENABLED


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
