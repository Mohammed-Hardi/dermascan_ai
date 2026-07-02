from __future__ import annotations

from typing import Any

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(input_size: int, config: dict[str, Any] | None = None) -> transforms.Compose:
    options = config or {}
    crop_scale = tuple(options.get("crop_scale", [0.75, 1.0]))
    horizontal_flip_probability = float(options.get("horizontal_flip_probability", 0.5))
    rotation_degrees = float(options.get("rotation_degrees", 15))
    color_jitter = options.get(
        "color_jitter",
        {"brightness": 0.2, "contrast": 0.2, "saturation": 0.15, "hue": 0.03},
    )
    blur_probability = float(options.get("blur_probability", 0.1))

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=crop_scale, antialias=True),
            transforms.RandomHorizontalFlip(p=horizontal_flip_probability),
            transforms.RandomRotation(rotation_degrees),
            transforms.ColorJitter(**color_jitter),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                p=blur_probability,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(input_size: int) -> transforms.Compose:
    resize_size = round(input_size / 0.875)
    return transforms.Compose(
        [
            transforms.Resize(resize_size, antialias=True),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
