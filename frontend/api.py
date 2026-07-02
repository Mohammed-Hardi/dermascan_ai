import os
from typing import Any

import requests


API_URL = os.getenv("DERMASCAN_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 45


class ApiError(RuntimeError):
    pass


def health_check() -> bool:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.ok
    except requests.RequestException:
        return False


def get_model_info() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}/api/model-info", timeout=5)
        return response.json() if response.ok else None
    except (requests.RequestException, ValueError):
        return None


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
        raise ApiError("The PDF report service is unavailable.") from exc
    if not response.ok:
        raise ApiError("The report could not be generated or the scan has expired.")
    return response.content
