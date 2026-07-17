from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.app.config import Settings
from backend.app.schemas import QualityResult


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MIN_TEXT_LIKE_REGIONS = 4
FOCUS_CROP_SCALES = (0.9, 0.8, 0.7)
MIN_FOCUS_SKIN_GAIN = 0.06
MIN_FOCUS_TEXT_REGION_RATIO = 0.12
MAX_FOCUS_BORDER_SKIN_RATIO = 0.20


@dataclass(slots=True)
class ValidatedImage:
    image: Image.Image
    image_bytes: bytes
    quality: QualityResult


def _crop_coordinates(width: int, height: int, size: int) -> list[tuple[int, int]]:
    max_left = width - size
    max_top = height - size
    left_values = np.linspace(0, max_left, num=5).round().astype(int)
    top_values = np.linspace(0, max_top, num=5).round().astype(int)
    return [(int(left), int(top)) for left in left_values for top in top_values]


def auto_prepare_image(
    image: Image.Image,
    skin_mask: np.ndarray | None = None,
    text_region_ratio: float = 0.0,
    text_region_count: int = 0,
) -> Image.Image:
    """Crop toward a large skin-rich region while preserving useful context."""
    width, height = image.size
    crop_size = min(width, height)
    center_left = (width - crop_size) // 2
    center_top = (height - crop_size) // 2
    best_crop = (center_left, center_top, crop_size)

    if skin_mask is None:
        skin_mask = _skin_pixel_mask(np.asarray(image.convert("RGB")))
    border_width = max(8, round(crop_size * 0.08))
    minimum_border_skin_ratio = min(
        float(skin_mask[:, :border_width].mean()),
        float(skin_mask[:, -border_width:].mean()),
        float(skin_mask[:border_width, :].mean()),
        float(skin_mask[-border_width:, :].mean()),
    )
    has_text_heavy_border = (
        text_region_ratio >= MIN_FOCUS_TEXT_REGION_RATIO
        and text_region_count >= MIN_TEXT_LIKE_REGIONS
        and minimum_border_skin_ratio <= MAX_FOCUS_BORDER_SKIN_RATIO
    )
    if not has_text_heavy_border:
        return image.crop(
            (
                center_left,
                center_top,
                center_left + crop_size,
                center_top + crop_size,
            )
        ).copy()

    default_region = skin_mask[
        center_top : center_top + crop_size,
        center_left : center_left + crop_size,
    ]
    default_skin_ratio = float(default_region.mean())
    best_skin_ratio = default_skin_ratio
    best_score = default_skin_ratio + 0.02

    for scale in FOCUS_CROP_SCALES:
        candidate_size = max(224, round(crop_size * scale))
        if candidate_size >= crop_size:
            continue
        for left, top in _crop_coordinates(width, height, candidate_size):
            region = skin_mask[top : top + candidate_size, left : left + candidate_size]
            skin_ratio = float(region.mean())
            center_x = left + candidate_size / 2
            center_y = top + candidate_size / 2
            center_distance = (
                abs(center_x - width / 2) / width
                + abs(center_y - height / 2) / height
            )
            score = skin_ratio + (0.02 * scale) - (0.01 * center_distance)
            if score > best_score:
                best_score = score
                best_skin_ratio = skin_ratio
                best_crop = (left, top, candidate_size)

    if best_skin_ratio < default_skin_ratio + MIN_FOCUS_SKIN_GAIN:
        best_crop = (center_left, center_top, crop_size)

    left, top, selected_size = best_crop
    prepared = image.crop((left, top, left + selected_size, top + selected_size))
    if selected_size != crop_size:
        prepared = prepared.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    return prepared.copy()


def _skin_pixel_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    channel_range = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)

    # Broad skin-color masks are used only as an input-quality guard. They are
    # intentionally permissive to support different skin tones and lighting.
    # The channel-range condition keeps neutral paper and screenshots from
    # passing the YCrCb mask simply because gray has centered chroma values.
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
        & (channel_range >= 8)
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
    return hsv_mask | ycrcb_mask | broad_rgb_mask


def _skin_pixel_ratio(rgb: np.ndarray) -> float:
    return float(_skin_pixel_mask(rgb).mean())


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


def _has_text_like_content(
    region_ratio: float,
    region_count: int,
    max_region_ratio: float,
    skin_ratio: float,
    min_skin_ratio: float,
) -> bool:
    """Reject text structure only when the image also lacks human-skin evidence."""
    has_repeated_text_structure = (
        region_ratio > max_region_ratio
        and region_count >= MIN_TEXT_LIKE_REGIONS
    )
    return has_repeated_text_structure and skin_ratio < min_skin_ratio


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
    skin_mask = _skin_pixel_mask(rgb)
    skin_ratio = float(skin_mask.mean())
    text_region_ratio, text_region_count = _text_region_ratio(gray)

    reason = None
    if brightness < settings.min_brightness:
        reason = "The image is too dark. Retake it in brighter, even lighting."
    elif brightness > settings.max_brightness:
        reason = "The image is too bright. Avoid flash glare and direct light."
    elif blur_score < settings.blur_threshold:
        reason = "The image appears blurry. Hold the camera steady and retake it."
    elif _has_text_like_content(
        text_region_ratio,
        text_region_count,
        settings.max_text_region_ratio,
        skin_ratio,
        settings.min_skin_ratio,
    ):
        reason = "The image appears to be text-only rather than a human skin photo. Upload a clear photo of the affected skin area."
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

    prepared_image = auto_prepare_image(
        clean_image,
        skin_mask,
        text_region_ratio,
        text_region_count,
    )
    output = BytesIO()
    prepared_image.save(output, format="JPEG", quality=90, optimize=True)
    return ValidatedImage(prepared_image, output.getvalue(), quality)
