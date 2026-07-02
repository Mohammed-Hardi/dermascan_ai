from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from backend.app.config import Settings
from backend.app.models.model_loader import ModelUnavailableError, load_model
from backend.app.services.inference import predict_checkpoint
from ml.src.export_model import export_torchscript
from ml.src.gradcam import GradCAM
from ml.src.models import create_model
from ml.src.predict import predict_image


@pytest.fixture
def smoke_checkpoint(tmp_path: Path) -> Path:
    model = create_model("custom_cnn", num_classes=6, pretrained=False)
    path = tmp_path / "smoke-custom.pt"
    torch.save(
        {
            "model_name": "custom_cnn",
            "model_state_dict": model.state_dict(),
            "class_names": ["acne", "eczema", "tinea", "scabies", "psoriasis", "other"],
            "input_size": 32,
            "epoch": 1,
            "is_smoke_test": True,
        },
        path,
    )
    return path


def test_smoke_checkpoint_is_blocked_by_default(smoke_checkpoint: Path) -> None:
    with pytest.raises(ModelUnavailableError, match="smoke-test model"):
        load_model(smoke_checkpoint, torch.device("cpu"), allow_smoke_model=False)


def test_checkpoint_inference_when_explicitly_allowed(smoke_checkpoint: Path) -> None:
    settings = Settings(
        inference_mode="checkpoint",
        model_path=smoke_checkpoint,
        allow_smoke_model=True,
        confidence_threshold=0.0,
    )
    image = Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8))
    result = predict_checkpoint(image, settings)
    assert len(result.top_k) == 3
    assert result.top_prediction is not None
    assert result.model_version.startswith("custom_cnn-epoch-")


def test_torchscript_export_reloads(smoke_checkpoint: Path, tmp_path: Path) -> None:
    output_path = export_torchscript(smoke_checkpoint, tmp_path / "model.torchscript.pt")
    exported = torch.jit.load(str(output_path), map_location="cpu")
    with torch.inference_mode():
        logits = exported(torch.zeros(1, 3, 32, 32))
    assert logits.shape == (1, 6)
    assert output_path.with_suffix(".json").is_file()


def test_cli_prediction_contract(smoke_checkpoint: Path, tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), color=(100, 120, 140)).save(image_path)
    result = predict_image(smoke_checkpoint, image_path)
    assert len(result["top_k"]) == 3
    assert result["is_smoke_test"] is True
    assert 0.0 <= result["top_prediction"]["confidence"] <= 1.0


def test_gradcam_returns_normalized_heatmap() -> None:
    model = create_model("custom_cnn", num_classes=6, pretrained=False)
    input_tensor = torch.randn(1, 3, 64, 64)
    with GradCAM(model) as gradcam:
        heatmap, class_index = gradcam.generate(input_tensor)
    assert heatmap.shape == (64, 64)
    assert 0 <= class_index < 6
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0
