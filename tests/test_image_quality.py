from io import BytesIO

import pytest
from PIL import Image

from backend.app.config import get_settings
from backend.app.services.image_quality import validate_image


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
