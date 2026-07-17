from io import BytesIO

import pytest
import numpy as np
from PIL import Image, ImageDraw

from backend.app.config import get_settings
from backend.app.services.image_quality import (
    _has_text_like_content,
    _skin_pixel_ratio,
    validate_image,
)


def test_rejects_small_image() -> None:
    output = BytesIO()
    Image.new("RGB", (100, 100), color=(120, 90, 80)).save(output, format="PNG")

    with pytest.raises(ValueError, match="too small"):
        validate_image(output.getvalue(), get_settings())


def test_strips_to_rgb_jpeg(valid_image_bytes) -> None:
    result = validate_image(valid_image_bytes, get_settings())
    assert result.quality.is_acceptable is True
    assert result.image.mode == "RGB"
    assert result.image_bytes.startswith(b"\xff\xd8")


def test_backend_auto_center_crops_valid_image() -> None:
    randomizer = np.random.default_rng(24)
    base = np.array([180, 120, 95], dtype=np.int16)
    noise = randomizer.integers(-18, 19, size=(320, 420, 3), dtype=np.int16)
    pixels = np.clip(base + noise, 0, 255).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(pixels).save(output, format="JPEG", quality=92)

    result = validate_image(output.getvalue(), get_settings())

    assert result.quality.is_acceptable is True
    assert result.image.size == (320, 320)


def test_rejects_image_with_text() -> None:
    image = Image.new("RGB", (420, 320), color=(205, 205, 205))
    draw = ImageDraw.Draw(image)
    for row, text in enumerate(["DERMASCAN REPORT", "Patient: Example", "Diagnosis text", "Not a skin photo"]):
        draw.text((24, 35 + row * 52), text, fill=(5, 5, 5))
    output = BytesIO()
    image.save(output, format="JPEG", quality=92)

    result = validate_image(output.getvalue(), get_settings())

    assert result.quality.is_acceptable is False
    assert result.quality.reason is not None
    assert "text" in result.quality.reason.lower()


def test_accepts_skin_image_with_incidental_text() -> None:
    randomizer = np.random.default_rng(84)
    base = np.array([176, 126, 98], dtype=np.int16)
    noise = randomizer.integers(-18, 19, size=(320, 420, 3), dtype=np.int16)
    pixels = np.clip(base + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    for row, text in enumerate(["camera date", "clinic", "sample", "skin photo"]):
        draw.text((24, 35 + row * 52), text, fill=(5, 5, 5))
    output = BytesIO()
    image.save(output, format="JPEG", quality=92)

    result = validate_image(output.getvalue(), get_settings())

    assert result.quality.is_acceptable is True


def test_text_filter_requires_repeated_text_and_missing_skin() -> None:
    settings = get_settings()

    assert _has_text_like_content(
        0.08,
        3,
        settings.max_text_region_ratio,
        0.0,
        settings.min_skin_ratio,
    ) is False
    assert _has_text_like_content(
        0.08,
        4,
        settings.max_text_region_ratio,
        0.5,
        settings.min_skin_ratio,
    ) is False
    assert _has_text_like_content(
        0.08,
        4,
        settings.max_text_region_ratio,
        0.0,
        settings.min_skin_ratio,
    ) is True


def test_neutral_document_background_is_not_skin() -> None:
    image = np.full((320, 420, 3), 205, dtype=np.uint8)

    assert _skin_pixel_ratio(image) == 0.0


def test_rejects_non_skin_image() -> None:
    randomizer = np.random.default_rng(12)
    base = np.array([30, 90, 180], dtype=np.int16)
    noise = randomizer.integers(-35, 36, size=(320, 320, 3), dtype=np.int16)
    pixels = np.clip(base + noise, 0, 255).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(pixels).save(output, format="JPEG", quality=92)

    result = validate_image(output.getvalue(), get_settings())

    assert result.quality.is_acceptable is False
    assert result.quality.reason is not None
    assert "human skin" in result.quality.reason.lower()
