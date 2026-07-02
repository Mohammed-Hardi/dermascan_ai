from __future__ import annotations

import timm
import torch
from torch import nn


TIMM_MODEL_NAMES = {
    "efficientnet_b0": "efficientnet_b0",
    "mobilenet_v2": "mobilenetv2_100",
    "resnet50": "resnet50",
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
    if model_name not in TIMM_MODEL_NAMES:
        supported = ["custom_cnn", *TIMM_MODEL_NAMES]
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {supported}.")
    return timm.create_model(
        TIMM_MODEL_NAMES[model_name],
        pretrained=pretrained,
        num_classes=num_classes,
    )
