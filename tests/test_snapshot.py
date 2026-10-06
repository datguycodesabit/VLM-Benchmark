from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from vlm_bench.dataset import prepare_dataset
from vlm_bench.snapshot import copy_inputs, fingerprint, freeze, load


def _paired_data(root: Path) -> Path:
    data = root / "source"
    image = data / "images" / "exam-a" / "page-2" / "line-3.png"
    reference = data / "text" / "exam-a" / "page-2" / "line-3.txt"
    image.parent.mkdir(parents=True)
    reference.parent.mkdir(parents=True)
    Image.new("RGB", (27, 13), "white").save(image)
    reference.write_text("A frozen reference", encoding="utf-8")
    (data / "metadata.jsonl").write_text(
        json.dumps(
            {
                "id": "exam-a/page-2/line-3",
                "writer_id": "writer-4",
                "content_type": "equation",
                "sample_type": "word",
                "split": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return data


def _rewrite_integrity(path: Path) -> None:
    manifest_path = path / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in manifest.items() if key != "integrity"}
    canonical = json.dumps(
        body,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    manifest["integrity"] = hashlib.sha256(canonical).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_snapshot_moves_and_runs_without_original_dataset(tmp_path: Path) -> None:
    data = _paired_data(tmp_path)
    snapshot_path = tmp_path / "snapshot"
    frozen = freeze(data, snapshot_path)

    assert frozen["manifest_version"] == 2
    assert frozen["layout"] == "paired"
    assert frozen["preprocess"] == "original"
    assert frozen["samples"][0]["crop_path"].startswith(str(snapshot_path.resolve()))
    assert frozen["sample_ids"] == ["exam-a/page-2/line-3"]
    assert frozen["benchmark_fingerprint"] == fingerprint(frozen["samples"], "original")
    changed_label = dict(frozen["samples"][0], content_type="prose")
    assert fingerprint([changed_label], "original") != frozen["benchmark_fingerprint"]

    disk_manifest = json.loads((snapshot_path / "dataset-manifest.json").read_text())
    assert disk_manifest["samples"][0]["crop_path"] == "crops/exam-a/page-2/line-3.png"
    assert "image_path" not in disk_manifest["samples"][0]

    moved = tmp_path / "moved" / "snapshot"
    moved.parent.mkdir()
    snapshot_path.rename(moved)
    shutil.rmtree(data)

    loaded = load(moved)
    assert loaded["benchmark_fingerprint"] == frozen["benchmark_fingerprint"]
    assert loaded["samples"][0]["crop_path"] == str(
        (moved / "crops" / "exam-a" / "page-2" / "line-3.png").resolve()
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / ".lock").write_text("held", encoding="utf-8")
    [sample], copied_manifest = copy_inputs(moved, run_dir)
    assert copied_manifest["benchmark_fingerprint"] == loaded["benchmark_fingerprint"]
    assert sample["id"] == "exam-a/page-2/line-3"
    assert sample["reference"] == "A frozen reference"
    assert sample["reference_source"] == "text"
    assert sample["content_type"] == "equation"
    assert sample["sample_type"] == "word"
    assert sample["writer_id"] == "writer-4"
    assert sample["hashes"]["crop"] == frozen["samples"][0]["hashes"]["crop"]
    assert Path(sample["crop_path"]).is_file()
    assert (
        Path(sample["crop_path"]).read_bytes()
        == Path(loaded["samples"][0]["crop_path"]).read_bytes()
    )
    assert (run_dir / ".lock").read_text(encoding="utf-8") == "held"


def test_task_annotations_survive_freeze_load_copy_and_affect_fingerprint(tmp_path: Path) -> None:
    data = _paired_data(tmp_path)
    metadata_path = data / "metadata.jsonl"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["annotations"] = {
        "critical_expressions": [r"x^2"],
        "reading_order": [[r"x^2", r"= 0"]],
    }
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    snapshot_path = tmp_path / "snapshot"
    frozen = freeze(data, snapshot_path)
    annotations = frozen["samples"][0]["metadata"]["annotations"]
    changed_annotation = dict(frozen["samples"][0])
    changed_annotation["metadata"] = dict(changed_annotation["metadata"])
    changed_annotation["metadata"]["annotations"] = {"critical_expressions": [r"x^3"]}
    assert fingerprint([changed_annotation], "original") != frozen["benchmark_fingerprint"]
    assert load(snapshot_path)["samples"][0]["metadata"]["annotations"] == annotations

    copied_samples, _ = copy_inputs(snapshot_path, tmp_path / "run")
    assert copied_samples[0]["metadata"]["annotations"] == annotations


def test_freeze_refuses_to_overwrite_any_existing_output(tmp_path: Path) -> None:
    data = _paired_data(tmp_path)
    output = tmp_path / "snapshot"
    freeze(data, output)
    before = (output / "dataset-manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        freeze(data, output)

    assert (output / "dataset-manifest.json").read_bytes() == before


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("seed", True, "seed must be an integer"),
        ("seed", -1, "seed must be an integer"),
        ("seed", 2**32, "seed must be an integer"),
        ("limit", False, "limit must be a positive integer"),
        ("limit", 0, "limit must be a positive integer"),
        ("layout", "unknown", "layout must be"),
        ("preprocess", "unknown", "preprocess must be"),
        ("split", "development", "split must be"),
        ("content_type", "unknown", "content_type must be"),
    ],
)
def test_freeze_validates_selection_options_before_creating_output(
    tmp_path: Path, option: str, value: object, message: str
) -> None:
    output = tmp_path / "not-created" / "snapshot"

    with pytest.raises(ValueError, match=message):
        freeze(tmp_path / "missing-source", output, **{option: value})

    assert not output.parent.exists()


def test_load_detects_tampered_crop_bytes(tmp_path: Path) -> None:
    data = _paired_data(tmp_path)
    snapshot = tmp_path / "snapshot"
    freeze(data, snapshot)
    crop = snapshot / "crops" / "exam-a" / "page-2" / "line-3.png"
    crop.write_bytes(crop.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="crop hash mismatch"):
        load(snapshot)


def test_load_rejects_legacy_manifest_with_upgrade_guidance(tmp_path: Path) -> None:
    data = _paired_data(tmp_path)
    legacy = tmp_path / "legacy"
    prepare_dataset(data, legacy)

    with pytest.raises(ValueError, match="legacy.*v1.*prepare"):
        load(legacy)


def test_load_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    data = _paired_data(tmp_path)
    snapshot = tmp_path / "snapshot"
    freeze(data, snapshot)
    manifest_path = snapshot / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"][0]["crop_path"] = "../outside.png"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _rewrite_integrity(snapshot)

    with pytest.raises(ValueError, match="crop_path escapes"):
        load(snapshot)


def test_load_rejects_symlinked_crop_that_escapes_snapshot(tmp_path: Path) -> None:
    data = _paired_data(tmp_path)
    snapshot = tmp_path / "snapshot"
    freeze(data, snapshot)
    crop = snapshot / "crops" / "exam-a" / "page-2" / "line-3.png"
    original = crop.read_bytes()
    crop.unlink()
    outside = tmp_path / "outside.png"
    outside.write_bytes(original)
    try:
        crop.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="escapes the snapshot"):
        load(snapshot)
