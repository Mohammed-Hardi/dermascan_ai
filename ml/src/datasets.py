from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ["acne", "eczema", "tinea", "scabies", "psoriasis", "other"]
CLASS_TO_LABEL = {name: index for index, name in enumerate(CLASS_NAMES)}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass(slots=True)
class DatasetRecord:
    image_path: str
    label: int
    class_name: str
    case_id: str
    source: str
    width: int
    height: int
    sha256: str
    dhash: str
    phash: str
    duplicate_group: str = ""
    split: str = ""


@dataclass(slots=True)
class InvalidRecord:
    image_path: str
    case_id: str
    class_name: str
    reason: str


@dataclass(slots=True)
class DuplicatePair:
    image_path_a: str
    image_path_b: str
    case_id_a: str
    case_id_b: str
    class_name_a: str
    class_name_b: str
    dhash_distance: int
    phash_distance: int
    match_type: str


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first != root_second:
            self.parent[root_second] = root_first


class SkinConditionDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        csv_path: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        self.csv_path = csv_path
        self.transform = transform
        self.project_root = project_root
        with csv_path.open("r", encoding="utf-8", newline="") as file_handle:
            self.rows = list(csv.DictReader(file_handle))
        required = {"image_path", "label", "class_name", "case_id"}
        if not self.rows:
            raise ValueError(f"Dataset split is empty: {csv_path}")
        missing_columns = required.difference(self.rows[0])
        if missing_columns:
            raise ValueError(f"Missing dataset columns: {sorted(missing_columns)}")
        self.labels = [int(row["label"]) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image_path = self.project_root / row["image_path"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        return self.transform(image), int(row["label"])


def compute_dhash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.int16)
    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)
    return value


def compute_phash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float32)
    low_frequencies = cv2.dct(pixels)[:8, :8]
    median = float(np.median(low_frequencies.flatten()[1:]))
    value = 0
    for bit in (low_frequencies > median).flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def build_index(
    manifest_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> tuple[list[DatasetRecord], list[InvalidRecord]]:
    records: list[DatasetRecord] = []
    invalid: list[InvalidRecord] = []
    seen_paths: set[str] = set()

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        for row in csv.DictReader(file_handle):
            normalized_destination = row["destination"].replace("\\", "/")
            if normalized_destination in seen_paths:
                continue
            seen_paths.add(normalized_destination)

            image_path = project_root / normalized_destination
            class_name = row["class_name"].strip().lower()
            case_id = row["case_id"].strip()
            if class_name not in CLASS_TO_LABEL:
                invalid.append(InvalidRecord(normalized_destination, case_id, class_name, "unsupported class"))
                continue
            if not image_path.is_file():
                invalid.append(InvalidRecord(normalized_destination, case_id, class_name, "missing file"))
                continue

            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                    if width < 1 or height < 1:
                        raise ValueError("invalid image dimensions")
                    dhash_value = compute_dhash(image)
                    phash_value = compute_phash(image)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                invalid.append(InvalidRecord(normalized_destination, case_id, class_name, str(exc)))
                continue

            records.append(
                DatasetRecord(
                    image_path=_relative_path(image_path, project_root),
                    label=CLASS_TO_LABEL[class_name],
                    class_name=class_name,
                    case_id=case_id,
                    source="SCIN",
                    width=width,
                    height=height,
                    sha256=_file_sha256(image_path),
                    dhash=f"{dhash_value:016x}",
                    phash=f"{phash_value:016x}",
                )
            )
    return records, invalid


def find_duplicates(
    records: list[DatasetRecord],
    dhash_distance: int = 4,
    phash_distance: int = 6,
) -> list[DuplicatePair]:
    pairs: list[DuplicatePair] = []
    exact_groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        exact_groups[record.sha256].append(index)

    exact_pairs: set[tuple[int, int]] = set()
    for indexes in exact_groups.values():
        if len(indexes) < 2:
            continue
        for position, first_index in enumerate(indexes):
            for second_index in indexes[position + 1 :]:
                exact_pairs.add((first_index, second_index))

    dhashes = [int(record.dhash, 16) for record in records]
    phashes = [int(record.phash, 16) for record in records]
    for first_index, first_record in enumerate(records):
        for second_index in range(first_index + 1, len(records)):
            second_record = records[second_index]
            if first_record.case_id == second_record.case_id:
                continue
            is_exact = (first_index, second_index) in exact_pairs
            current_dhash_distance = (
                0 if is_exact else hamming_distance(dhashes[first_index], dhashes[second_index])
            )
            current_phash_distance = (
                0 if is_exact else hamming_distance(phashes[first_index], phashes[second_index])
            )
            if not is_exact and (
                current_dhash_distance > dhash_distance or current_phash_distance > phash_distance
            ):
                continue
            pairs.append(
                DuplicatePair(
                    image_path_a=first_record.image_path,
                    image_path_b=second_record.image_path,
                    case_id_a=first_record.case_id,
                    case_id_b=second_record.case_id,
                    class_name_a=first_record.class_name,
                    class_name_b=second_record.class_name,
                    dhash_distance=current_dhash_distance,
                    phash_distance=current_phash_distance,
                    match_type="exact" if is_exact else "near",
                )
            )
    return pairs


def _split_counts(item_count: int) -> tuple[int, int, int]:
    if item_count <= 0:
        return 0, 0, 0
    if item_count == 1:
        return 1, 0, 0
    if item_count == 2:
        return 1, 1, 0

    train_count = max(1, round(item_count * SPLIT_RATIOS["train"]))
    val_count = max(1, round(item_count * SPLIT_RATIOS["val"]))
    if train_count + val_count >= item_count:
        train_count = item_count - 2
        val_count = 1
    test_count = item_count - train_count - val_count
    return train_count, val_count, test_count


def assign_splits(
    records: list[DatasetRecord],
    duplicate_pairs: list[DuplicatePair],
    seed: int = 42,
) -> None:
    case_ids = sorted({record.case_id for record in records})
    case_groups = UnionFind(case_ids)
    for pair in duplicate_pairs:
        if pair.class_name_a == pair.class_name_b:
            case_groups.union(pair.case_id_a, pair.case_id_b)

    records_by_group: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        group_id = case_groups.find(record.case_id)
        record.duplicate_group = group_id
        records_by_group[group_id].append(record)

    groups_by_class: dict[str, list[str]] = defaultdict(list)
    for group_id, group_records in records_by_group.items():
        classes = {record.class_name for record in group_records}
        group_class = next(iter(classes)) if len(classes) == 1 else "mixed"
        groups_by_class[group_class].append(group_id)

    randomizer = random.Random(seed)
    group_to_split: dict[str, str] = {}
    for class_name in sorted(groups_by_class):
        group_ids = sorted(groups_by_class[class_name])
        randomizer.shuffle(group_ids)
        train_count, val_count, _ = _split_counts(len(group_ids))
        for index, group_id in enumerate(group_ids):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"
            group_to_split[group_id] = split

    for record in records:
        record.split = group_to_split[record.duplicate_group]


def assert_no_leakage(records: list[DatasetRecord]) -> None:
    case_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        case_splits[record.case_id].add(record.split)
        group_splits[record.duplicate_group].add(record.split)

    leaking_cases = {case_id: splits for case_id, splits in case_splits.items() if len(splits) > 1}
    leaking_groups = {group_id: splits for group_id, splits in group_splits.items() if len(splits) > 1}
    if leaking_cases or leaking_groups:
        raise RuntimeError(
            f"Split leakage detected: {len(leaking_cases)} cases and "
            f"{len(leaking_groups)} duplicate groups."
        )


def build_summary(records: list[DatasetRecord], invalid: list[InvalidRecord], duplicates: list[DuplicatePair]) -> dict:
    summary: dict[str, object] = {
        "total_images": len(records),
        "total_cases": len({record.case_id for record in records}),
        "invalid_or_missing": len(invalid),
        "duplicate_pairs": {
            "exact": sum(pair.match_type == "exact" for pair in duplicates),
            "near": sum(pair.match_type == "near" for pair in duplicates),
            "cross_class": sum(pair.class_name_a != pair.class_name_b for pair in duplicates),
        },
        "splits": {},
    }
    split_summary: dict[str, object] = {}
    for split in SPLIT_RATIOS:
        split_records = [record for record in records if record.split == split]
        split_summary[split] = {
            "images": len(split_records),
            "cases": len({record.case_id for record in split_records}),
            "classes": {
                class_name: sum(record.class_name == class_name for record in split_records)
                for class_name in CLASS_NAMES
            },
        }
    summary["splits"] = split_summary
    return summary


def _write_csv(path: Path, rows: list[object], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_outputs(
    records: list[DatasetRecord],
    invalid: list[InvalidRecord],
    duplicates: list[DuplicatePair],
    processed_dir: Path,
    split_dir: Path,
) -> dict:
    record_fields = list(DatasetRecord.__dataclass_fields__)
    invalid_fields = list(InvalidRecord.__dataclass_fields__)
    duplicate_fields = list(DuplicatePair.__dataclass_fields__)

    _write_csv(processed_dir / "dataset_index.csv", records, record_fields)
    _write_csv(processed_dir / "invalid_images.csv", invalid, invalid_fields)
    _write_csv(processed_dir / "duplicate_pairs.csv", duplicates, duplicate_fields)
    for split in SPLIT_RATIOS:
        split_records = [record for record in records if record.split == split]
        _write_csv(split_dir / f"{split}.csv", split_records, record_fields)

    summary = build_summary(records, invalid, duplicates)
    with (split_dir / "summary.json").open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2)
        file_handle.write("\n")
    return summary


def prepare_dataset(
    manifest_path: Path,
    processed_dir: Path,
    split_dir: Path,
    seed: int = 42,
    dhash_distance: int = 4,
    phash_distance: int = 6,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    records, invalid = build_index(manifest_path, project_root)
    duplicates = find_duplicates(records, dhash_distance, phash_distance)
    assign_splits(records, duplicates, seed)
    assert_no_leakage(records)
    return write_outputs(records, invalid, duplicates, processed_dir, split_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DermaScan image data and grouped splits.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "ml" / "data" / "source" / "scin" / "download_manifest.csv",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "ml" / "data" / "processed",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=PROJECT_ROOT / "ml" / "data" / "splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dhash-distance", type=int, default=4)
    parser.add_argument("--phash-distance", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(
        manifest_path=args.manifest,
        processed_dir=args.processed_dir,
        split_dir=args.split_dir,
        seed=args.seed,
        dhash_distance=args.dhash_distance,
        phash_distance=args.phash_distance,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
