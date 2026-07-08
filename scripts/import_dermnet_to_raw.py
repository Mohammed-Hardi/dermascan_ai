from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw"
DATASET_SOURCES = PROJECT_ROOT / "dataset_sources" / "dermnet"
TARGET_CLASSES = ["acne", "eczema", "psoriasis"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


CLASS_KEYWORDS = {
    "acne": ["acne"],
    "eczema": ["eczema", "atopic", "dermatitis"],
    "psoriasis": ["psoriasis"],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def extract_archive(archive_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(output_dir)
    return output_dir


def class_for_path(path: Path) -> str | None:
    searchable = " ".join(part.lower().replace("_", " ").replace("-", " ") for part in path.parts)
    for class_name, keywords in CLASS_KEYWORDS.items():
        if any(keyword in searchable for keyword in keywords):
            return class_name
    return None


def existing_hashes() -> set[str]:
    hashes: set[str] = set()
    for class_name in TARGET_CLASSES:
        class_dir = RAW_DIR / class_name
        if not class_dir.exists():
            continue
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                hashes.add(file_sha256(image_path))
    return hashes


def import_images(source_dir: Path, target_per_class: int) -> dict[str, int]:
    hashes = existing_hashes()
    copied = {class_name: 0 for class_name in TARGET_CLASSES}
    current_counts = {
        class_name: len([path for path in (RAW_DIR / class_name).rglob("*") if path.is_file()])
        for class_name in TARGET_CLASSES
    }
    for image_path in source_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        class_name = class_for_path(image_path)
        if class_name is None or current_counts[class_name] >= target_per_class:
            continue
        if not is_valid_image(image_path):
            continue
        digest = file_sha256(image_path)
        if digest in hashes:
            continue
        destination_dir = RAW_DIR / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"dermnet_{digest[:16]}{image_path.suffix.lower()}"
        shutil.copy2(image_path, destination)
        hashes.add(digest)
        copied[class_name] += 1
        current_counts[class_name] += 1
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import real DermNet images into DermaScan raw folders.")
    parser.add_argument("--source-dir", type=Path, help="Extracted DermNet dataset folder.")
    parser.add_argument("--archive", type=Path, help="Optional complete Kaggle .archive/.zip file to extract first.")
    parser.add_argument("--target-per-class", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    if args.archive:
        source_dir = extract_archive(args.archive, DATASET_SOURCES)
    if source_dir is None or not source_dir.exists():
        raise SystemExit("Provide --source-dir or a complete --archive file.")
    copied = import_images(source_dir, args.target_per_class)
    print("Imported real image counts:")
    for class_name, count in copied.items():
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    main()
