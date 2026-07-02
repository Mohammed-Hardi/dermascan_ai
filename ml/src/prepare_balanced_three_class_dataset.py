from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

from ml.src.datasets import PROJECT_ROOT
from ml.src.prepare_three_class_dataset import prepare_three_class_splits


CLASSES = ["acne", "eczema", "psoriasis"]
SOURCE_SPLIT_DIR = PROJECT_ROOT / "ml" / "data" / "splits_three_class"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "data" / "balanced_three_class"
TRAIN_IMAGE_DIR = OUTPUT_DIR / "train_images"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    output = ImageOps.exif_transpose(image).convert("RGB")
    if rng.random() < 0.5:
        output = ImageOps.mirror(output)
    angle = rng.uniform(-18, 18)
    output = output.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    width, height = output.size
    crop_scale = rng.uniform(0.82, 1.0)
    crop_w = max(1, int(width * crop_scale))
    crop_h = max(1, int(height * crop_scale))
    left = rng.randint(0, max(0, width - crop_w))
    top = rng.randint(0, max(0, height - crop_h))
    output = output.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.Resampling.LANCZOS)
    output = ImageEnhance.Brightness(output).enhance(rng.uniform(0.86, 1.14))
    output = ImageEnhance.Contrast(output).enhance(rng.uniform(0.86, 1.14))
    output = ImageEnhance.Color(output).enhance(rng.uniform(0.9, 1.1))
    return output


def build_balanced_training_set(images_per_class: int = 900, seed: int = 42) -> Path:
    prepare_three_class_splits(class_names=CLASSES)
    train_rows = read_rows(SOURCE_SPLIT_DIR / "train.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        grouped[row["class_name"]].append(row)

    label_map = {class_name: str(index) for index, class_name in enumerate(CLASSES)}
    balanced_rows: list[dict[str, str]] = []
    rng = random.Random(seed)
    TRAIN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:
        source_rows = grouped[class_name]
        if not source_rows:
            raise ValueError(f"No training images found for {class_name}")
        class_dir = TRAIN_IMAGE_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(images_per_class):
            source = source_rows[index % len(source_rows)]
            source_path = PROJECT_ROOT / source["image_path"]
            with Image.open(source_path) as image:
                output = augment_image(image, random.Random(rng.randint(0, 10_000_000)))
            output_name = f"{class_name}_{index:04d}.jpg"
            output_path = class_dir / output_name
            output.save(output_path, format="JPEG", quality=90, optimize=True)
            row = dict(source)
            row["image_path"] = output_path.relative_to(PROJECT_ROOT).as_posix()
            row["label"] = label_map[class_name]
            row["class_name"] = class_name
            row["case_id"] = f"{source['case_id']}__aug_{index:04d}"
            row["source"] = "SCIN_augmented"
            row["split"] = "train"
            balanced_rows.append(row)

    write_rows(OUTPUT_DIR / "train.csv", balanced_rows)
    for split in ["val", "test"]:
        rows = read_rows(SOURCE_SPLIT_DIR / f"{split}.csv")
        for row in rows:
            row["label"] = label_map[row["class_name"]]
        write_rows(OUTPUT_DIR / f"{split}.csv", rows)

    summary = {
        "classes": CLASSES,
        "images_per_class_train": images_per_class,
        "train_total": len(balanced_rows),
        "unique_source_train_counts": {class_name: len(grouped[class_name]) for class_name in CLASSES},
        "validation_total": len(read_rows(OUTPUT_DIR / "val.csv")),
        "test_total": len(read_rows(OUTPUT_DIR / "test.csv")),
        "note": "Training images are balanced with deterministic augmentation from SCIN train split only.",
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced acne/eczema/psoriasis training set.")
    parser.add_argument("--images-per-class", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = build_balanced_training_set(args.images_per_class, args.seed)
    print(f"Balanced three-class dataset summary: {summary_path}")


if __name__ == "__main__":
    main()
