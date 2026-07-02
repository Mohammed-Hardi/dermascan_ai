from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def valid_image_bytes() -> bytes:
    randomizer = np.random.default_rng(42)
    pixels = randomizer.integers(35, 220, size=(320, 320, 3), dtype=np.uint8)
    output = BytesIO()
    Image.fromarray(pixels).save(output, format="JPEG", quality=92)
    return output.getvalue()


@pytest.fixture
def dark_image_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (320, 320), color=(5, 5, 5)).save(output, format="JPEG")
    return output.getvalue()
