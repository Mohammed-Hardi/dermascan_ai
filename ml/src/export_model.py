from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ml.src.models import create_model


def export_torchscript(checkpoint_path: Path, output_path: Path | None = None) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model_name", "model_state_dict", "class_names", "input_size"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")

    model = create_model(
        str(checkpoint["model_name"]),
        len(checkpoint["class_names"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    input_size = int(checkpoint["input_size"])
    example = torch.zeros(1, 3, input_size, input_size)
    with torch.inference_mode():
        exported = torch.jit.trace(model, example)
        exported = torch.jit.freeze(exported)

    if output_path is None:
        output_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.torchscript.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(exported, output_path)

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "format": "torchscript",
                "source_checkpoint": checkpoint_path.as_posix(),
                "model_name": checkpoint["model_name"],
                "class_names": checkpoint["class_names"],
                "input_size": input_size,
                "is_smoke_test": bool(checkpoint.get("is_smoke_test", False)),
                "epoch": checkpoint.get("epoch"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a DermaScan checkpoint for deployment.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--format", choices=["torchscript"], default="torchscript")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_torchscript(args.model, args.output)
    print(f"Exported model: {output_path}")
    print(f"Metadata: {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
