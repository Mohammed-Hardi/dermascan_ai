from __future__ import annotations

import warnings

import torch
from torch import nn


TIMM_MODEL_NAMES = {
    "efficientnet_b0": "efficientnet_b0",
    "mobilenet_v2": "mobilenetv2_100",
    "resnet50": "resnet50",
}

TORCHVISION_MODEL_NAMES = {
    "torchvision_efficientnet_b0",
    "torchvision_mobilenet_v2",
    "torchvision_resnet18",
    "torchvision_resnet50",
}


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        channels = [3, 32, 64, 128, 256]
        blocks: list[nn.Module] = []
        for input_channels, output_channels in zip(channels, channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name == "custom_cnn":
        return CustomCNN(num_classes=num_classes)

    if model_name in TORCHVISION_MODEL_NAMES:
        return _create_torchvision_model(model_name, num_classes, pretrained)

    if model_name not in TIMM_MODEL_NAMES:
        supported = ["custom_cnn", *TIMM_MODEL_NAMES, *TORCHVISION_MODEL_NAMES]
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {supported}.")

    try:
        import timm

        return timm.create_model(
            TIMM_MODEL_NAMES[model_name],
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except ModuleNotFoundError:
        warnings.warn(
            f"timm is not installed; using torchvision fallback for '{model_name}'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _create_torchvision_model(f"torchvision_{model_name}", num_classes, pretrained)


def _create_torchvision_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    from torchvision import models

    weights = None
    if model_name == "torchvision_efficientnet_b0":
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "torchvision_mobilenet_v2":
        if pretrained:
            weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "torchvision_resnet18":
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "torchvision_resnet50":
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported torchvision model '{model_name}'.")
