from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.app.config import Settings
from backend.app.schemas import QualityResult


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


@dataclass(slots=True)
class ValidatedImage:
    image: Image.Image
    image_bytes: bytes
    quality: QualityResult


def _skin_pixel_ratio(rgb: np.ndarray) -> float:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)

    # Broad skin-color masks are used only as an input-quality guard. They are
    # intentionally permissive to support different skin tones and lighting.
    hsv_mask = (
        (hsv[:, :, 0] <= 35)
        & (hsv[:, :, 1] >= 20)
        & (hsv[:, :, 2] >= 35)
    )
    ycrcb_mask = (
        (ycrcb[:, :, 1] >= 125)
        & (ycrcb[:, :, 1] <= 190)
        & (ycrcb[:, :, 2] >= 65)
        & (ycrcb[:, :, 2] <= 145)
    )
    normalized = rgb.astype(np.float32) / 255.0
    red, green, blue = normalized[:, :, 0], normalized[:, :, 1], normalized[:, :, 2]
    broad_rgb_mask = (
        (red > 0.18)
        & (green > 0.12)
        & (blue > 0.08)
        & ((np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])) > 0.04)
        & (red >= blue * 0.75)
    )
    return float((hsv_mask | ycrcb_mask | broad_rgb_mask).mean())


def _text_region_ratio(gray: np.ndarray) -> tuple[float, int]:
    small = cv2.resize(gray, (640, int(gray.shape[0] * (640 / gray.shape[1])))) if gray.shape[1] > 640 else gray
    threshold = cv2.adaptiveThreshold(
        small,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10,
    )
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    connected = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = small.shape[0] * small.shape[1]
    text_area = 0
    text_like_count = 0
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if height < 8 or width < 18:
            continue
        aspect_ratio = width / max(height, 1)
        area = width * height
        if area > image_area * 0.18:
            continue
        if 1.4 <= aspect_ratio <= 18 and area >= image_area * 0.0004:
            text_area += area
            text_like_count += 1
    return float(text_area / image_area), text_like_count


def validate_image(data: bytes, settings: Settings) -> ValidatedImage:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if not data:
        raise ValueError("The uploaded image is empty.")
    if len(data) > max_bytes:
        raise ValueError(f"The image exceeds the {settings.max_upload_mb} MB upload limit.")

    try:
        with Image.open(BytesIO(data)) as source:
            image_format = source.format
            if image_format not in ALLOWED_FORMATS:
                raise ValueError("Use a JPG, JPEG, PNG, or WEBP image.")
            source.verify()

        with Image.open(BytesIO(data)) as source:
            clean_image = ImageOps.exif_transpose(source).convert("RGB").copy()
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc

    width, height = clean_image.size
    if min(width, height) < settings.min_image_dimension:
        raise ValueError(
            f"The image is too small. Use at least {settings.min_image_dimension} x "
            f"{settings.min_image_dimension} pixels."
        )

    rgb = np.asarray(clean_image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    skin_ratio = _skin_pixel_ratio(rgb)
    text_region_ratio, text_region_count = _text_region_ratio(gray)

    reason = None
    if brightness < settings.min_brightness:
        reason = "The image is too dark. Retake it in brighter, even lighting."
    elif brightness > settings.max_brightness:
        reason = "The image is too bright. Avoid flash glare and direct light."
    elif blur_score < settings.blur_threshold:
        reason = "The image appears blurry. Hold the camera steady and retake it."
    elif text_region_ratio > settings.max_text_region_ratio and text_region_count >= 3:
        reason = "Images containing text, labels, screenshots, or document-like content are not allowed. Upload a clear photo of human skin only."
    elif skin_ratio < settings.min_skin_ratio:
        reason = "The image does not appear to contain enough human skin. Upload or crop a clear photo of the affected human skin area only."

    quality = QualityResult(
        is_acceptable=reason is None,
        reason=reason,
        width=width,
        height=height,
        brightness=round(brightness, 2),
        blur_score=round(blur_score, 2),
        skin_ratio=round(skin_ratio, 4),
        text_region_ratio=round(text_region_ratio, 4),
    )

    output = BytesIO()
    clean_image.save(output, format="JPEG", quality=90, optimize=True)
    return ValidatedImage(clean_image, output.getvalue(), quality)
