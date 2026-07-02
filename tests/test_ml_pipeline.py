import csv
from pathlib import Path

import pytest
import torch
from PIL import Image

from ml.src.augmentations import build_eval_transform, build_train_transform
from ml.src.datasets import SkinConditionDataset
from ml.src.models import create_model
from ml.src.train import compute_class_weights, load_config


def test_transforms_return_expected_shape() -> None:
    image = Image.new("RGB", (320, 240), color=(120, 90, 70))
    train_tensor = build_train_transform(96)(image)
    eval_transform = build_eval_transform(96)
    eval_tensor = eval_transform(image)
    assert train_tensor.shape == (3, 96, 96)
    assert eval_tensor.shape == (3, 96, 96)
    assert torch.equal(eval_tensor, eval_transform(image))


def test_csv_dataset_loads_image(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), color=(100, 120, 140)).save(image_path)
    csv_path = tmp_path / "split.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["image_path", "label", "class_name", "case_id"])
        writer.writeheader()
        writer.writerow({"image_path": "image.png", "label": 1, "class_name": "eczema", "case_id": "1"})
    dataset = SkinConditionDataset(csv_path, build_eval_transform(32), project_root=tmp_path)
    tensor, label = dataset[0]
    assert tensor.shape == (3, 32, 32)
    assert label == 1
    assert dataset.labels == [1]


@pytest.mark.parametrize("model_name", ["custom_cnn", "efficientnet_b0", "mobilenet_v2", "resnet50"])
def test_model_factory_outputs_six_logits(model_name: str) -> None:
    model = create_model(model_name, num_classes=6, pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 6)


def test_class_weights_upweight_minority_class() -> None:
    weights = compute_class_weights([0, 0, 0, 1], num_classes=2)
    assert weights[1] > weights[0]


def test_production_config_is_valid() -> None:
    config = load_config(Path("ml/configs/efficientnet_b0.yaml"))
    assert config["model_name"] == "efficientnet_b0"
    assert len(config["class_names"]) == config["num_classes"]
