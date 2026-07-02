from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from ml.src.datasets import PROJECT_ROOT


DEFAULT_CLASSES = ["eczema", "tinea", "psoriasis"]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "ml" / "data" / "splits"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ml" / "data" / "splits_three_class"


def convert_split(source_csv: Path, output_csv: Path, class_names: list[str]) -> Counter[str]:
    label_map = {class_name: index for index, class_name in enumerate(class_names)}
    counts: Counter[str] = Counter()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with source_csv.open("r", encoding="utf-8", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"Split has no header: {source_csv}")
        required = {"label", "class_name", "split"}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(f"Split {source_csv} is missing columns: {sorted(missing)}")

        with output_csv.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                class_name = row["class_name"].strip().lower()
                if class_name not in label_map:
                    continue
                row["class_name"] = class_name
                row["label"] = str(label_map[class_name])
                row["split"] = output_csv.stem
                writer.writerow(row)
                counts[class_name] += 1
    return counts


def prepare_three_class_splits(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    class_names: list[str] | None = None,
) -> Path:
    active_classes = class_names or DEFAULT_CLASSES
    summary: dict[str, object] = {
        "source_splits": source_dir.as_posix(),
        "output_splits": output_dir.as_posix(),
        "class_names": active_classes,
        "label_map": {class_name: index for index, class_name in enumerate(active_classes)},
        "splits": {},
    }

    total_counts: Counter[str] = Counter()
    for split in ["train", "val", "test"]:
        counts = convert_split(source_dir / f"{split}.csv", output_dir / f"{split}.csv", active_classes)
        total_counts.update(counts)
        summary["splits"][split] = {
            "total_images": int(sum(counts.values())),
            "class_counts": {class_name: int(counts[class_name]) for class_name in active_classes},
        }

    summary["total_images"] = int(sum(total_counts.values()))
    summary["total_class_counts"] = {
        class_name: int(total_counts[class_name]) for class_name in active_classes
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create DermaScan three-class SCIN splits.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = prepare_three_class_splits(args.source_dir, args.output_dir, args.classes)
    print(f"Three-class split summary: {summary_path}")


if __name__ == "__main__":
    main()
