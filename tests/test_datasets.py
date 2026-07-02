from pathlib import Path

from PIL import Image

from ml.src.datasets import (
    DatasetRecord,
    DuplicatePair,
    assert_no_leakage,
    assign_splits,
    compute_dhash,
    compute_phash,
    hamming_distance,
)


def make_record(case_id: str, class_name: str, index: int) -> DatasetRecord:
    labels = {"acne": 0, "eczema": 1, "tinea": 2, "scabies": 3, "psoriasis": 4, "other": 5}
    return DatasetRecord(
        image_path=f"ml/data/raw/{class_name}/{index}.png",
        label=labels[class_name],
        class_name=class_name,
        case_id=case_id,
        source="test",
        width=224,
        height=224,
        sha256=f"sha-{index}",
        dhash=f"{index:016x}",
        phash=f"{index:016x}",
    )


def test_dhash_is_deterministic() -> None:
    image = Image.new("RGB", (32, 32), color=(90, 120, 150))
    assert compute_dhash(image) == compute_dhash(image.copy())
    assert compute_phash(image) == compute_phash(image.copy())
    assert hamming_distance(0b1010, 0b0011) == 2


def test_case_images_remain_in_one_split() -> None:
    records = []
    for index in range(20):
        class_name = "acne" if index < 10 else "eczema"
        records.append(make_record(f"case-{index}", class_name, index * 2))
        records.append(make_record(f"case-{index}", class_name, index * 2 + 1))

    assign_splits(records, [], seed=42)
    assert_no_leakage(records)
    for case_id in {record.case_id for record in records}:
        assert len({record.split for record in records if record.case_id == case_id}) == 1


def test_near_duplicate_cases_remain_in_one_split() -> None:
    records = [make_record(f"case-{index}", "acne", index) for index in range(10)]
    duplicate = DuplicatePair(
        image_path_a=records[0].image_path,
        image_path_b=records[1].image_path,
        case_id_a=records[0].case_id,
        case_id_b=records[1].case_id,
        class_name_a="acne",
        class_name_b="acne",
        dhash_distance=2,
        phash_distance=3,
        match_type="near",
    )
    assign_splits(records, [duplicate], seed=10)
    assert records[0].split == records[1].split
    assert_no_leakage(records)


def test_split_assignment_is_deterministic() -> None:
    first = [make_record(f"case-{index}", "psoriasis", index) for index in range(20)]
    second = [make_record(f"case-{index}", "psoriasis", index) for index in range(20)]
    assign_splits(first, [], seed=99)
    assign_splits(second, [], seed=99)
    assert [record.split for record in first] == [record.split for record in second]
