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

    reason = None
    if brightness < settings.min_brightness:
        reason = "The image is too dark. Retake it in brighter, even lighting."
    elif brightness > settings.max_brightness:
        reason = "The image is too bright. Avoid flash glare and direct light."
    elif blur_score < settings.blur_threshold:
        reason = "The image appears blurry. Hold the camera steady and retake it."

    quality = QualityResult(
        is_acceptable=reason is None,
        reason=reason,
        width=width,
        height=height,
        brightness=round(brightness, 2),
        blur_score=round(blur_score, 2),
    )

    output = BytesIO()
    clean_image.save(output, format="JPEG", quality=90, optimize=True)
    return ValidatedImage(clean_image, output.getvalue(), quality)
