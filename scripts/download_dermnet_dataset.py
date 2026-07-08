from __future__ import annotations

import argparse
import time
from pathlib import Path

import kagglehub


DATASETS = {
    "small": "chetantirumala/skin-diseases-dermnet",
    "large": "shubhamgoel27/dermnet",
}


def download_with_retries(dataset_slug: str, attempts: int, wait_seconds: int) -> Path:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        print(f"Download attempt {attempt}/{attempts}: {dataset_slug}")
        try:
            dataset_path = Path(kagglehub.dataset_download(dataset_slug))
            print(f"Dataset downloaded to: {dataset_path}")
            return dataset_path
        except Exception as exc:
            last_error = exc
            print(f"Download failed: {exc}")
            if attempt < attempts:
                print(f"Waiting {wait_seconds} seconds before resuming...")
                time.sleep(wait_seconds)
    raise RuntimeError(f"Dataset download failed after {attempts} attempts.") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume-download a DermNet-based dataset from Kaggle.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="small",
        help="Use 'small' first. Use 'large' only if you need more images.",
    )
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--wait-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_with_retries(DATASETS[args.dataset], args.attempts, args.wait_seconds)


if __name__ == "__main__":
    main()
