from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from vlm_bench.dataset import prepare_dataset


def _line(text: str, boxes: list[tuple[int, int, int, int]]) -> str:
    cmps = "".join(
        f'<cmp x="{x}" y="{y}" width="{width}" height="{height}" />'
        for x, y, width, height in boxes
    )
    return f'<line text="{text}"><word>{cmps}</word></line>'


def _sample(
    data_dir: Path,
    sample_id: str,
    lines: list[str] | None = None,
    *,
    size: tuple[int, int] = (120, 100),
    writer_id: str = "writer-7",
) -> None:
    image_dir = data_dir / "images" / "nested"
    xml_dir = data_dir / "xml" / "nested"
    image_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(image_dir / f"{sample_id}.png")
    if lines is None:
        lines = [_line("The quick brown fox", [(10, 20, 30, 8)])]
    xml = (
        f'<form writer-id="{writer_id}" width="{size[0]}" height="{size[1]}">'
        "<machine-print-part>"
        + _line("Printed prompt should not appear", [(0, 0, 90, 10)])
        + "</machine-print-part><handwritten-part>"
        + "".join(lines)
        + "</handwritten-part></form>"
    )
    (xml_dir / f"{sample_id}.xml").write_text(xml, encoding="utf-8")


def test_crops_union_of_retained_handwriting_and_uses_same_lines_for_xml_reference(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    _sample(
        data_dir,
        "form-1",
        [
            _line("  First handwritten line.  ", [(10, 20, 20, 5), (40, 25, 10, 7)]),
            _line("Second line", [(15, 40, 25, 8)]),
            _line("J Smith", [(8, 91, 30, 5)]),  # short signature in bottom 15%
            _line("Name: Jane Doe", [(5, 94, 50, 4)]),  # final Name line
        ],
    )

    [sample] = prepare_dataset(data_dir, tmp_path / "run")

    assert sample["id"] == "form-1"
    assert sample["reference"] == "First handwritten line.\nSecond line"
    assert sample["reference_source"] == "xml"
    assert sample["writer_id"] == "writer-7"
    assert sample["crop_bbox"] == [0, 0, 70, 68]
    assert Path(sample["image_path"]).is_absolute()
    assert Path(sample["crop_path"]).is_absolute()
    with Image.open(sample["crop_path"]) as crop:
        assert crop.size == (70, 68)
    assert all(len(value) == 64 for value in sample["hashes"].values())


def test_txt_reference_is_authoritative_and_never_overwritten(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _sample(data_dir, "form-1")
    reference_dir = data_dir / "references"
    reference_dir.mkdir()
    reference_bytes = b"  my reviewed reference\r\n"
    (reference_dir / "form-1.txt").write_bytes(reference_bytes)

    [sample] = prepare_dataset(data_dir, tmp_path / "run", write_references=True)

    assert sample["reference"] == reference_bytes.decode("utf-8")
    assert sample["reference_source"] == "txt"
    assert sample["hashes"]["reference"] == hashlib.sha256(reference_bytes).hexdigest()
    assert (reference_dir / "form-1.txt").read_bytes() == reference_bytes


def test_write_references_generates_missing_txt_from_filtered_xml(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _sample(
        data_dir,
        "form-1",
        [_line("Copy this line", [(4, 10, 40, 9)]), _line("A B", [(4, 90, 20, 5)])],
    )

    [sample] = prepare_dataset(data_dir, tmp_path / "run", write_references=True)

    created = data_dir / "references" / "form-1.txt"
    assert created.read_text(encoding="utf-8") == "Copy this line"
    assert sample["reference_source"] == "xml"


def test_write_references_covers_full_validated_dataset_before_sampling(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    for sample_id in ("form-a", "form-b", "form-c"):
        _sample(data_dir, sample_id)

    selected = prepare_dataset(data_dir, tmp_path / "run", limit=1, seed=2, write_references=True)

    assert len(selected) == 1
    assert sorted(path.stem for path in (data_dir / "references").glob("*.txt")) == [
        "form-a",
        "form-b",
        "form-c",
    ]


def test_rejects_invalid_component_coordinates_even_when_limit_would_exclude_sample(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    _sample(data_dir, "form-a")
    _sample(data_dir, "form-b", [_line("Out of bounds", [(115, 20, 10, 5)])])

    with pytest.raises(ValueError, match="exceeds image bounds"):
        prepare_dataset(data_dir, tmp_path / "run", limit=1, seed=3)


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ("missing_xml", "Missing XML"),
        ("duplicate_xml", "Duplicate XML ID"),
        ("duplicate_image", "Duplicate image ID"),
        ("empty_reference", "Reference.*empty"),
    ],
)
def test_rejects_missing_duplicate_and_empty_dataset_inputs(
    tmp_path: Path, setup: str, message: str
) -> None:
    data_dir = tmp_path / "data"
    _sample(data_dir, "form-1")
    if setup == "missing_xml":
        (data_dir / "xml" / "nested" / "form-1.xml").unlink()
    elif setup == "duplicate_xml":
        duplicate = data_dir / "xml" / "form-1.xml"
        duplicate.parent.mkdir(parents=True, exist_ok=True)
        duplicate.write_text((data_dir / "xml" / "nested" / "form-1.xml").read_text())
    elif setup == "duplicate_image":
        (data_dir / "images" / "form-1.jpg").parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (120, 100), "white").save(data_dir / "images" / "form-1.jpg")
    else:
        reference_dir = data_dir / "references"
        reference_dir.mkdir()
        (reference_dir / "form-1.txt").write_text(" \n\t", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        prepare_dataset(data_dir, tmp_path / "run")


def test_subset_is_reproducible_sorted_and_rejects_oversized_limit(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    for sample_id in ("form-d", "form-b", "form-c", "form-a"):
        _sample(data_dir, sample_id)

    first = prepare_dataset(data_dir, tmp_path / "run-1", limit=2, seed=17)
    second = prepare_dataset(data_dir, tmp_path / "run-2", limit=2, seed=17)

    assert [sample["id"] for sample in first] == [sample["id"] for sample in second]
    assert [sample["id"] for sample in first] == sorted(sample["id"] for sample in first)
    assert len(first) == 2
    with pytest.raises(ValueError, match="only 4 are available"):
        prepare_dataset(data_dir, tmp_path / "run-3", limit=5)
