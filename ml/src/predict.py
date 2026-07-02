from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from ml.src.augmentations import build_eval_transform
from ml.src.models import create_model


def predict_image(checkpoint_path: Path, image_path: Path, top_k: int = 3) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = list(checkpoint["class_names"])
    model = create_model(checkpoint["model_name"], len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    tensor = build_eval_transform(int(checkpoint["input_size"]))(image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    count = min(top_k, len(class_names))
    confidences, indices = torch.topk(probabilities, k=count)
    predictions = [
        {"class_name": class_names[int(index)], "confidence": round(float(confidence), 6)}
        for confidence, index in zip(confidences, indices)
    ]
    return {
        "image": image_path.as_posix(),
        "checkpoint": checkpoint_path.as_posix(),
        "is_smoke_test": bool(checkpoint.get("is_smoke_test", False)),
        "top_prediction": predictions[0],
        "top_k": predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DermaScan checkpoint prediction.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(predict_image(args.model, args.image, args.top_k), indent=2))


if __name__ == "__main__":
    main()
