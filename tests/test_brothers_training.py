from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_brothers_training.py"
SPEC = importlib.util.spec_from_file_location("brothers_training", SCRIPT)
assert SPEC and SPEC.loader
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


def fixture_dataset(root: Path) -> list[dict]:
    records = []
    for year in [2006, 2008, 2009, 2011, 2012, 2013, 2014, 2015]:
        for kind in ["prose", "equation"]:
            for origin in ["real", "synthetic"]:
                pair = f"exam{year}/page01-{kind}01"
                sid = origin + "/" + pair
                image = root / "images" / (sid + ".png")
                text = root / "text" / (sid + ".txt")
                image.parent.mkdir(parents=True, exist_ok=True)
                text.parent.mkdir(parents=True, exist_ok=True)
                im = Image.new("RGB", (200, 50), "white")
                ImageDraw.Draw(im).rectangle((10, 10, 90, 30), fill=(len(records), 20, 30))
                im.save(image)
                text.write_text(f"Example {year} {kind} {origin}.\n")
                records.append({
                    "id": sid, "origin": origin, "split": "test", "pdf_page": 1,
                    "source_document": ("synthetic-inspired-by-" if origin == "synthetic" else "")
                    + f"Exam 1 Fall {year}.pdf", "paired_source_id": "real/" + pair,
                    "content_type": kind, "sample_type": "line", "writer_id": "unknown",
                    "verification_status": "verified",
                    "annotations": {"pair_id": pair, "verification_method": "fixture"},
                })
    training.write_jsonl(root / "metadata.jsonl", records)
    return records


def test_complete_package_is_grouped_portable_and_does_not_change_source(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "output"
    fixture_dataset(source)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    report = training.build(source, output)
    assert report["split_counts"] == {"train": 20, "validation": 4, "test": 8}
    assert report["cross_split_pairs"] == report["cross_split_exams"] == 0
    assert report["frozen_test_samples"] == 8
    assert before == {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    moved = tmp_path / "moved"
    shutil.copytree(output, moved)
    assert training.verify_package(moved)["test_fingerprint"] == report["test_fingerprint"]
    inputs = [json.loads(line) for line in (moved / "manifests/test.inputs.jsonl").read_text().splitlines()]
    assert all("text" not in row and "reference_path" not in row for row in inputs)
    with pytest.raises(FileExistsError):
        training.build(source, output)


def test_wrong_synthetic_lineage_is_rejected(tmp_path: Path):
    records = fixture_dataset(tmp_path / "source")
    records[1]["annotations"]["pair_id"] = "exam2015/page01-prose01"
    with pytest.raises(ValueError, match="Pair lineage"):
        training.assign_splits(records)


def test_unresolved_label_prevents_publishing(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "output"
    records = fixture_dataset(source)
    (source / "text" / (records[0]["id"] + ".txt")).write_text("ILLEGIBLE\n")
    with pytest.raises(ValueError, match="Label audit failed"):
        training.build(source, output)
    assert not output.exists()


def test_changed_test_image_fails_integrity_check(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "output"
    fixture_dataset(source)
    training.build(source, output)
    next((output / "frozen-test/crops").rglob("*.png")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Package file changed"):
        training.verify_package(output)


@pytest.mark.parametrize("text,kind,issue", [
    ("x_{i", "equation", "unbalanced_latex_braces"),
    ("Vector $A", "prose", "unbalanced_inline_math"),
    ("value \ufffd", "prose", "unresolved_label"),
    ("x\x00", "prose", "control_character"),
])
def test_label_audit_flags_invalid_targets(text: str, kind: str, issue: str):
    assert issue in training.audit_label(text, kind)
