import csv
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ml.src.augmentations import build_eval_transform, build_train_transform
from ml.src.datasets import SkinConditionDataset
from ml.src.evaluate_basic import compute_basic_metrics
from ml.src.models import create_model
from ml.src.prepare_balanced_three_class_dataset import infer_source_and_case_id, split_rows_by_case
from ml.src.train import compute_class_weights, freeze_feature_extractor, load_config


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


def test_freeze_feature_extractor_keeps_classifier_trainable() -> None:
    model = create_model("torchvision_efficientnet_b0", num_classes=3, pretrained=False)

    trainable_parameters = freeze_feature_extractor(model)

    assert trainable_parameters > 0
    assert all(
        parameter.requires_grad == name.startswith("classifier.")
        for name, parameter in model.named_parameters()
    )


def test_basic_metrics_reports_weighted_f1() -> None:
    targets = np.asarray([0, 1, 2, 2])
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.6, 0.3],
            [0.1, 0.2, 0.7],
        ]
    )

    metrics = compute_basic_metrics(targets, probabilities, ["acne", "scabies", "psoriasis"])

    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["weighted_f1"] <= 1.0
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]


def test_active_three_class_config_uses_scabies() -> None:
    config = load_config(Path("ml/configs/accuracy_target_acne_scabies_psoriasis.yaml"))

    assert config["class_names"] == ["acne", "scabies", "psoriasis"]


def test_dataset_source_inference_preserves_patient_id() -> None:
    source, case_id = infer_source_and_case_id(
        Path("skindisnet_scabies_p_241__sc_1.jpg"), {}
    )

    assert source == "SkinDisNet (CC BY-NC 4.0)"
    assert case_id == "skindisnet:skindisnet_scabies_p_241"


def test_case_grouped_split_prevents_patient_leakage() -> None:
    rows = [
        {"case_id": case_id, "image_path": f"{case_id}_{index}.jpg"}
        for case_id in ["patient-a", "patient-b", "patient-c", "patient-d", "patient-e"]
        for index in range(2)
    ]

    splits = split_rows_by_case(rows, random.Random(42))
    assigned = {
        row["case_id"]: split
        for split, split_rows in splits.items()
        for row in split_rows
    }
    for case_id in {row["case_id"] for row in rows}:
        assert len(
            {
                split
                for split, split_rows in splits.items()
                if any(row["case_id"] == case_id for row in split_rows)
            }
        ) == 1
        assert assigned[case_id] in {"train", "val", "test"}


def test_production_config_is_valid() -> None:
    config = load_config(Path("ml/configs/efficientnet_b0.yaml"))
    assert config["model_name"] == "efficientnet_b0"
    assert len(config["class_names"]) == config["num_classes"]
