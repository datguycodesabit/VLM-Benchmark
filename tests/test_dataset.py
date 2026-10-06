from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from vlm_bench.dataset import check_dataset, prepare_dataset, split_dataset


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


def test_auto_detects_pasted_iam_word_archive(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "archive" / "iam_words" / "words" / "a01" / "a01-000u"
    image_dir.mkdir(parents=True)
    Image.new("L", (32, 18), "white").save(image_dir / "a01-000u-00-00.png")
    (data_dir / "archive" / "iam_words" / "words.txt").write_text(
        "# IAM words\na01-000u-00-00 ok 154 1 1 1 10 10 NN hello\n",
        encoding="utf-8",
    )

    [sample] = prepare_dataset(data_dir, tmp_path / "run")

    assert sample["id"] == "a01-000u-00-00"
    assert sample["reference"] == "hello"
    assert sample["reference_source"] == "words.txt"
    assert sample["crop_bbox"] == [0, 0, 32, 18]
    with Image.open(sample["crop_path"]) as crop:
        assert crop.size == (32, 18)
        assert crop.mode == "L"


def test_auto_detects_pasted_iam_line_archive_and_preserves_crop_by_default(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "archive" / "iam_lines" / "lines" / "a01"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (80, 24), "white").save(image_dir / "a01-000u-00-00.png")
    (data_dir / "archive" / "iam_lines" / "lines.txt").write_text(
        "# IAM lines\na01-000u-00-00 ok 154 1 0 0 80 24 Hello handwritten line\n",
        encoding="utf-8",
    )

    [sample] = prepare_dataset(data_dir, tmp_path / "run")

    assert sample["id"] == "a01-000u-00-00"
    assert sample["reference"] == "Hello handwritten line"
    assert sample["reference_source"] == "lines.txt"
    assert sample["writer_id"] is None
    assert sample["sample_type"] == "line"
    assert sample["crop_bbox"] == [0, 0, 80, 24]
    with Image.open(sample["crop_path"]) as crop:
        assert crop.size == (80, 24)
        assert crop.mode == "RGB"


def _paired_sample(data_dir: Path, relative_id: str, text: str, *, size=(32, 16)) -> None:
    image_path = data_dir / "images" / f"{relative_id}.png"
    text_path = data_dir / "text" / f"{relative_id}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(image_path)
    text_path.write_text(text, encoding="utf-8")


def test_paired_layout_matches_relative_paths_and_freezes_original_inputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _paired_sample(data_dir, "exam-2006/page-03/line-02", "Reviewed words")

    [sample] = prepare_dataset(data_dir, tmp_path / "prepared")

    assert sample["id"] == "exam-2006/page-03/line-02"
    assert sample["reference"] == "Reviewed words"
    assert sample["writer_id"] is None
    assert sample["sample_type"] == "line"
    assert sample["preprocess"] == "original"
    assert Path(sample["crop_path"]).read_bytes() == Path(sample["image_path"]).read_bytes()
    manifest = json.loads((tmp_path / "prepared" / "dataset-manifest.json").read_text())
    assert manifest["sample_ids"] == [sample["id"]]
    assert manifest["samples"][0]["reference"] == "Reviewed words"
    assert manifest["samples"][0]["hashes"]["image"] == sample["hashes"]["image"]


def test_paired_layout_accepts_references_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _paired_sample(data_dir, "exam-2006/line-02", "A reviewed line")
    (data_dir / "text").rename(data_dir / "references")

    [sample] = prepare_dataset(data_dir, tmp_path / "prepared")

    assert sample["reference_source"] == "references"


def test_paired_layout_rejects_ambiguous_reference_directories(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _paired_sample(data_dir, "exam-2006/line-02", "A reviewed line")
    (data_dir / "references").mkdir()

    with pytest.raises(ValueError, match="ambiguous"):
        prepare_dataset(data_dir, tmp_path / "prepared")


def test_paired_validation_reports_missing_corrupt_and_empty_samples(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _paired_sample(data_dir, "doc/valid", "Readable")
    (data_dir / "images" / "doc" / "missing.png").write_bytes(b"not an image")
    (data_dir / "text" / "doc" / "empty.txt").write_text(" \n", encoding="utf-8")

    report = check_dataset(data_dir)

    codes = {issue["code"] for issue in report["issues"]}
    assert not report["valid"]
    assert {"missing_reference", "missing_image", "corrupt_image", "empty_reference"} <= codes
    assert report["counts"]["samples"] == 1
    with pytest.raises(ValueError, match="files do not match"):
        prepare_dataset(data_dir, tmp_path / "prepared")


def test_paired_metadata_filters_without_relabeling_and_enhanced_is_opt_in(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    _paired_sample(data_dir, "exam-a/line-01", "Prose sample")
    _paired_sample(data_dir, "exam-b/line-01", "Equation sample")
    metadata = [
        {
            "id": "exam-a/line-01",
            "source_document": "exam-a",
            "split": "train",
            "content_type": "prose",
            "verification_status": "verified",
        },
        {
            "id": "exam-b/line-01",
            "source_document": "exam-b",
            "split": "test",
            "content_type": "equation",
        },
    ]
    (data_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in metadata) + "\n", encoding="utf-8"
    )

    [sample] = prepare_dataset(
        data_dir,
        tmp_path / "test-set",
        split="test",
        content_type="equation",
        preprocess="enhanced",
    )

    assert sample["id"] == "exam-b/line-01"
    assert sample["split"] == "test"
    assert sample["content_type"] == "equation"
    assert sample["verification_status"] is None
    with Image.open(sample["crop_path"]) as crop:
        assert crop.size == (32 * 3 + 32, 16 * 3 + 32)
        assert crop.mode == "L"
    with pytest.raises(ValueError, match="No samples match"):
        prepare_dataset(data_dir, tmp_path / "wrong-filter", split="train", content_type="equation")


def test_iam_line_parser_honors_quality_flags_and_pipe_separator(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    image_dir = data_dir / "lines"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (80, 24), "white").save(image_dir / "a01-000u-00-00.png")
    Image.new("RGB", (80, 24), "white").save(image_dir / "a01-000u-00-01.png")
    annotations = "\n".join(
        (
            "a01-000u-00-00 | ok | 154 | 1 | 0 | 0 | 80 | 24 | Hello | there",
            "a01-000u-00-01 | err | 154 | 1 | 0 | 0 | 80 | 24 | Exclude me",
        )
    )
    (data_dir / "lines.txt").write_text(annotations + "\n", encoding="utf-8")

    [sample] = prepare_dataset(data_dir, tmp_path / "prepared", layout="iam-lines")

    assert sample["reference"] == "Hello there"
    assert sample["sample_type"] == "line"


def test_split_dataset_keeps_documents_together_and_rejects_image_leakage() -> None:
    samples = [
        {
            "id": f"doc-{doc}/line-{index}",
            "source_document": f"doc-{doc}",
            "hashes": {"image": f"hash-{doc}-{index}"},
        }
        for doc in range(5)
        for index in range(2)
    ]

    first = split_dataset(samples, seed=8)
    second = split_dataset(samples, seed=8)

    assert first == second
    assert {row["split"] for row in first} == {"train", "validation", "test"}
    by_document = {}
    for row in first:
        by_document.setdefault(row["source_document"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_document.values())
    leaked = [dict(sample) for sample in samples]
    leaked[2] = {**leaked[2], "hashes": {"image": samples[0]["hashes"]["image"]}}
    with pytest.raises(ValueError, match="crosses documents"):
        split_dataset(leaked)


def test_writer_disjoint_split_keeps_shared_writers_and_multi_writer_documents_together():
    samples = [
        {
            "id": "a-1",
            "source_document": "doc-a",
            "writer_id": "writer-1",
            "hashes": {"image": "a1"},
        },
        {
            "id": "a-2",
            "source_document": "doc-a",
            "writer_id": "writer-2",
            "hashes": {"image": "a2"},
        },
        {
            "id": "b-1",
            "source_document": "doc-b",
            "writer_id": "writer-2",
            "hashes": {"image": "b1"},
        },
        {
            "id": "c-1",
            "source_document": "doc-c",
            "writer_id": "writer-3",
            "hashes": {"image": "c1"},
        },
        {
            "id": "d-1",
            "source_document": "doc-d",
            "writer_id": "writer-4",
            "hashes": {"image": "d1"},
        },
        {
            "id": "e-1",
            "source_document": "doc-e",
            "writer_id": "writer-5",
            "hashes": {"image": "e1"},
        },
    ]

    first = split_dataset(samples, seed=11, protocol="writer-disjoint")
    second = split_dataset(samples, seed=11, protocol="writer-disjoint")

    assert first == second
    by_writer = {}
    by_document = {}
    for row in first:
        by_writer.setdefault(row["writer_id"], set()).add(row["split"])
        by_document.setdefault(row["source_document"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_writer.values())
    assert all(len(splits) == 1 for splits in by_document.values())
    assert by_document["doc-a"] == by_document["doc-b"]


def test_writer_disjoint_split_requires_writer_metadata():
    with pytest.raises(ValueError, match="writer_id metadata"):
        split_dataset(
            [
                {
                    "id": "a",
                    "source_document": "doc-a",
                    "hashes": {"image": "image-a"},
                }
            ],
            protocol="writer-disjoint",
        )


def test_dataset_check_marks_cross_split_document_overlap_invalid(tmp_path: Path):
    data_dir = tmp_path / "data"
    _sample(data_dir, "line-a")
    _sample(data_dir, "line-b")
    metadata = [
        {
            "id": "line-a",
            "source_document": "exam-shared",
            "split": "train",
        },
        {
            "id": "line-b",
            "source_document": "exam-shared",
            "split": "test",
        },
    ]
    (data_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8"
    )

    report = check_dataset(data_dir)

    assert report["valid"] is False
    assert any(issue["code"] == "document_split_overlap" for issue in report["issues"])
    assert any(
        finding["type"] == "document_split_overlap"
        and finding["sample_ids"] == ["line-a", "line-b"]
        for finding in report["findings"]
    )


def test_dataset_check_flags_perceptual_duplicates_for_review_only(tmp_path: Path):
    data_dir = tmp_path / "data"
    for sample_id, color, reference in (
        ("white", (255, 255, 255), "white page"),
        ("off-white", (254, 254, 254), "off-white page"),
    ):
        image = data_dir / "images" / f"{sample_id}.png"
        text = data_dir / "text" / f"{sample_id}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        text.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 30), color).save(image)
        text.write_text(reference, encoding="utf-8")

    report = check_dataset(data_dir)

    near_duplicates = [
        finding
        for finding in report["findings"]
        if finding["type"] == "perceptual_duplicate_review"
    ]
    assert report["valid"] is True
    assert len(near_duplicates) == 1
    assert near_duplicates[0]["sample_ids"] == ["off-white", "white"]


def test_task_annotations_are_preserved_and_schema_checked(tmp_path: Path):
    data_dir = tmp_path / "data"
    _sample(data_dir, "equation-1")
    annotations = {
        "critical_expressions": [r"x^2", r"= 0"],
        "reading_order": [[r"x^2", r"= 0"]],
    }
    (data_dir / "metadata.jsonl").write_text(
        json.dumps({"id": "equation-1", "annotations": annotations}) + "\n",
        encoding="utf-8",
    )

    [sample] = prepare_dataset(data_dir, tmp_path / "prepared")

    assert sample["metadata"]["annotations"] == annotations


@pytest.mark.parametrize(
    ("annotations", "message"),
    [
        ([], "annotations must be a JSON object"),
        ({"critical_expressions": ["x", " "]}, "critical_expressions"),
        ({"critical_expressions": ["x", "x"]}, "duplicate critical expressions"),
        ({"reading_order": [["x"]]}, "entries must be pairs"),
    ],
)
def test_task_annotation_schema_errors_are_explicit(
    tmp_path: Path, annotations: object, message: str
):
    data_dir = tmp_path / "data"
    _sample(data_dir, "equation-1")
    (data_dir / "metadata.jsonl").write_text(
        json.dumps({"id": "equation-1", "annotations": annotations}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        prepare_dataset(data_dir, tmp_path / "prepared")
