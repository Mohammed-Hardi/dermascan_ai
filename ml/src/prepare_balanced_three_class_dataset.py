from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["acne", "scabies", "psoriasis"]
SOURCE_SPLIT_DIR = PROJECT_ROOT / "ml" / "data" / "splits_acne_scabies_psoriasis"
RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "data" / "balanced_acne_scabies_psoriasis"
TRAIN_IMAGE_DIR = OUTPUT_DIR / "train_images"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SCIN_MANIFEST = PROJECT_ROOT / "ml" / "data" / "source" / "scin" / "download_manifest.csv"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def read_rows(path: Path) -> list[dict[str, str]]:
    # utf-8-sig handles source manifests that include a byte-order mark.
    with path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scin_case_map(path: Path = SCIN_MANIFEST) -> dict[str, str]:
    if not path.is_file():
        return {}
    rows = read_rows(path)
    return {
        Path(row["destination"]).name: row["case_id"]
        for row in rows
        if row.get("destination") and row.get("case_id")
    }


def infer_source_and_case_id(image_path: Path, scin_case_map: dict[str, str]) -> tuple[str, str]:
    name = image_path.name.lower()
    stem = image_path.stem.lower()
    if name.startswith("mendeley_scabies_benchmark_"):
        return "Mendeley Scabies Benchmark (CC BY 4.0)", f"mendeley-benchmark:{stem}"
    if name.startswith("skindisnet_scabies_"):
        patient_stem = stem.split("__", maxsplit=1)[0]
        return "SkinDisNet (CC BY-NC 4.0)", f"skindisnet:{patient_stem}"
    if name.startswith("dermnet_"):
        return "DermNet", f"dermnet:{stem}"
    if image_path.name in scin_case_map:
        return "SCIN", f"scin:{scin_case_map[image_path.name]}"
    return "Unknown local source", f"local:{stem}"


def scan_raw_images(class_names: list[str], seed: int) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(seed)
    label_map = {class_name: str(index) for index, class_name in enumerate(class_names)}
    scin_case_map = load_scin_case_map()
    grouped: dict[str, list[dict[str, str]]] = {}

    for class_name in class_names:
        class_dir = RAW_DIR / class_name
        if not class_dir.is_dir():
            raise ValueError(f"Missing raw class directory: {class_dir}")

        rows: list[dict[str, str]] = []
        seen_hashes: set[str] = set()
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
            except OSError:
                continue

            image_hash = file_sha256(image_path)
            if image_hash in seen_hashes:
                continue
            seen_hashes.add(image_hash)
            relative_path = image_path.relative_to(PROJECT_ROOT).as_posix()
            source, case_id = infer_source_and_case_id(image_path, scin_case_map)
            rows.append(
                {
                    "image_path": relative_path,
                    "label": label_map[class_name],
                    "class_name": class_name,
                    "case_id": case_id,
                    "source": source,
                    "width": str(width),
                    "height": str(height),
                    "sha256": image_hash,
                    "dhash": "",
                    "phash": "",
                    "duplicate_group": image_path.stem,
                    "split": "",
                }
            )

        if not rows:
            raise ValueError(f"No valid images found for {class_name}")
        rng.shuffle(rows)
        grouped[class_name] = rows

    return grouped


def split_rows_by_case(
    rows: list[dict[str, str]], rng: random.Random
) -> dict[str, list[dict[str, str]]]:
    rows_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_case[row["case_id"]].append(row)

    case_ids = sorted(rows_by_case)
    rng.shuffle(case_ids)
    case_count = len(case_ids)
    train_count = max(1, round(case_count * SPLIT_RATIOS["train"]))
    val_count = max(1, round(case_count * SPLIT_RATIOS["val"])) if case_count >= 3 else 0
    if train_count + val_count >= case_count:
        train_count = max(1, case_count - 2)
        val_count = 1 if case_count > 1 else 0

    case_to_split: dict[str, str] = {}
    for index, case_id in enumerate(case_ids):
        if index < train_count:
            split = "train"
        elif index < train_count + val_count:
            split = "val"
        else:
            split = "test"
        case_to_split[case_id] = split

    split_rows = {split: [] for split in SPLIT_RATIOS}
    for row in rows:
        split_rows[case_to_split[row["case_id"]]].append(row)
    return split_rows


def split_raw_images(class_names: list[str], seed: int) -> dict[str, dict[str, list[dict[str, str]]]]:
    grouped = scan_raw_images(class_names, seed)
    splits = {split: {} for split in ["train", "val", "test"]}
    rng = random.Random(seed)

    for class_name, rows in grouped.items():
        class_splits = split_rows_by_case(rows, rng)
        for split, split_rows in class_splits.items():
            for row in split_rows:
                row["split"] = split
            splits[split][class_name] = split_rows

    case_splits: dict[str, set[str]] = defaultdict(set)
    for split in SPLIT_RATIOS:
        for class_name in class_names:
            for row in splits[split][class_name]:
                case_splits[row["case_id"]].add(split)
    leaking_cases = {case_id: values for case_id, values in case_splits.items() if len(values) > 1}
    if leaking_cases:
        raise RuntimeError(f"Case leakage detected for {len(leaking_cases)} cases")

    SOURCE_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        rows = [row for class_name in class_names for row in splits[split][class_name]]
        write_rows(SOURCE_SPLIT_DIR / f"{split}.csv", rows)

    summary = {
        "source": RAW_DIR.as_posix(),
        "classes": class_names,
        "splits": {
            split: {
                class_name: len(splits[split][class_name])
                for class_name in class_names
            }
            for split in ["train", "val", "test"]
        },
        "total_class_counts": {class_name: len(grouped[class_name]) for class_name in class_names},
        "total_case_counts": {
            class_name: len({row["case_id"] for row in grouped[class_name]})
            for class_name in class_names
        },
    }
    (SOURCE_SPLIT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return splits


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
    return ImageOps.fit(output, (320, 320), method=Image.Resampling.LANCZOS)


def build_balanced_training_set(images_per_class: int = 900, seed: int = 42) -> Path:
    split_raw_images(CLASSES, seed)
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
            output.save(output_path, format="JPEG", quality=88)
            row = dict(source)
            row["image_path"] = output_path.relative_to(PROJECT_ROOT).as_posix()
            row["label"] = label_map[class_name]
            row["class_name"] = class_name
            row["case_id"] = f"{source['case_id']}__aug_{index:04d}"
            row["source"] = f"{source['source']}_augmented"
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
        "unique_source_train_case_counts": {
            class_name: len({row["case_id"] for row in grouped[class_name]})
            for class_name in CLASSES
        },
        "validation_total": len(read_rows(OUTPUT_DIR / "val.csv")),
        "test_total": len(read_rows(OUTPUT_DIR / "test.csv")),
        "note": "Training images are balanced with deterministic augmentation from raw real images.",
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced acne/scabies/psoriasis training set.")
    parser.add_argument("--images-per-class", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = build_balanced_training_set(args.images_per_class, args.seed)
    print(f"Balanced three-class dataset summary: {summary_path}")


if __name__ == "__main__":
    main()
