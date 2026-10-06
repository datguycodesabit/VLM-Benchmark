from __future__ import annotations

import hashlib
import json

import pytest
from PIL import Image

from vlm_bench.cli import main
from vlm_bench.inspection import (
    group_report,
    inspect_run,
    minimum_edit_alignment,
    write_inspection_html,
)
from vlm_bench.metrics import score
from vlm_bench.research import score_equation


def _sample(root, sample_id, reference, **metadata):
    crop_path = root / "crops" / f"{sample_id}.png"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 24), "white").save(crop_path)
    return {
        "id": sample_id,
        "reference": reference,
        "crop_path": str(crop_path),
        "metadata": metadata,
    }


def _success(model, sample_id, prediction, reference, *, equation=False, **values):
    metrics = score_equation(prediction, reference) if equation else score(prediction, reference)
    return {
        "model": model,
        "sample_id": sample_id,
        "status": "success",
        "prediction": prediction,
        "reference": reference,
        "metrics": metrics,
        "truncated": False,
        "error": None,
        **values,
    }


def _make_run(root):
    root.mkdir()
    samples = [
        _sample(
            root,
            "s1",
            "hello",
            source_document="doc-1",
            writer_id="writer-1",
            split="test",
            difficulty="easy",
            sample_type="line",
            verification_status="verified",
            content_type="prose",
        ),
        _sample(
            root,
            "s2",
            "abc",
            source_document="doc-2",
            writer_id="writer-2",
            split="test",
            difficulty="hard",
            sample_type="line",
            verification_status="verified",
            content_type="prose",
        ),
        _sample(
            root,
            "s3",
            "excluded",
            source_document="doc-3",
            writer_id="writer-3",
            split="train",
            difficulty="easy",
            sample_type="line",
            verification_status="verified",
            content_type="prose",
        ),
        _sample(
            root,
            "s4",
            "e\u0301",
            source_document="doc-4",
            writer_id="writer-4",
            split="test",
            sample_type="page",
            verification_status="verified",
            content_type="equation",
        ),
    ]
    records = [
        _success(
            "model-a",
            "s1",
            "<script>alert(1)</script> Transcription: hello hello hello",
            "hello",
        ),
        _success("model-a", "s2", "", "abc", truncated=True),
        _success("model-a", "s3", "excluded", "excluded"),
        _success("model-a", "s4", "é", "e\u0301", equation=True),
        {
            "model": "model-b",
            "sample_id": "s2",
            "status": "error",
            "prediction": None,
            "reference": "abc",
            "error": "provider unavailable",
            "metrics": None,
        },
        {
            "model": "model-b",
            "sample_id": "s4",
            "status": "unsupported",
            "prediction": None,
            "reference": "e\u0301",
            "error": "unsupported task",
        },
    ]
    manifest = {
        "schema_version": 1,
        "status": "complete_with_errors",
        "models": ["model-a", "model-b"],
        "samples": samples,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    return root


def test_alignment_counts_match_metric_tie_breaking_and_normalizes_equations():
    tied = minimum_edit_alignment("aa", "a")
    assert tied["counts"] == {
        "substitutions": 0,
        "deletions": 1,
        "insertions": 0,
        "edits": 1,
    }
    assert tied["counts"]["edits"] == score("a", "aa")["char_edits"]
    assert tied["operations"][0]["operation"] == "deletion"
    equation = minimum_edit_alignment("e\u0301", "é", equation=True)
    assert equation["counts"]["edits"] == score_equation("é", "e\u0301")["char_edits"] == 0


def test_alignment_cell_limit_omits_matrix_without_losing_explicit_status():
    alignment = minimum_edit_alignment("abcdef", "uvwxyz", max_cells=10)
    assert alignment["status"] == "omitted_cell_limit"
    assert alignment["cell_count"] == 49
    assert alignment["operations"] == []


def test_inspection_uses_frozen_references_eligibility_and_review_flags(tmp_path):
    run = _make_run(tmp_path / "run")
    report = inspect_run(run, worst=10)

    assert report["eligible_sample_count"] == 3
    assert report["excluded_sample_count"] == 1
    assert report["ignored_unsupported_result_count"] == 1
    assert report["eligible_result_count"] == 4
    assert all(item["sample_id"] != "s3" for item in report["results"])
    assert report["results"][0]["status"] == "error"
    by_sample = {
        item["sample_id"]: item for item in report["results"] if item["model"] == "model-a"
    }
    assert by_sample["s1"]["reference"] == "hello"
    assert {"repeated_text", "extra_commentary"}.issubset(set(by_sample["s1"]["flags"]))
    assert {"empty_output", "truncated"}.issubset(set(by_sample["s2"]["flags"]))
    assert by_sample["s4"]["cer"] == 0


def test_inspection_rejects_unknown_models_and_reference_mismatch(tmp_path):
    run = _make_run(tmp_path / "run")
    results_path = run / "results.jsonl"
    records = [json.loads(line) for line in results_path.read_text().splitlines()]
    records[0]["model"] = "unexpected"
    results_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="unknown model"):
        inspect_run(run)

    run = _make_run(tmp_path / "second-run")
    records = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    records[0]["reference"] = "stale reference"
    (run / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="differs from the frozen reference"):
        inspect_run(run)


def test_html_report_escapes_text_and_uses_local_images(tmp_path):
    run = _make_run(tmp_path / "run")
    output = tmp_path / "review" / "inspection.html"
    written, report = write_inspection_html(run, output, worst=4)
    document = written.read_text(encoding="utf-8")

    assert report["selected_count"] == 4
    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert 'src="../run/crops/s1.png"' in document
    assert "heuristic prompts" in document
    with pytest.raises(FileExistsError, match="already exists"):
        write_inspection_html(run, output, worst=4)


def test_group_report_has_coverage_document_counts_and_unknown_bucket(tmp_path):
    run = _make_run(tmp_path / "run")
    report = group_report(run, "difficulty")

    assert report["eligible_sample_count"] == 3
    assert report["excluded_sample_count"] == 1
    assert report["eligible_document_count"] == 3
    groups = {item["value"]: item for item in report["groups"]}
    assert groups["easy"]["eligible_sample_count"] == 1
    assert groups["easy"]["document_count"] == 1
    assert groups["hard"]["eligible_sample_count"] == 1
    assert groups["Unknown"]["is_unknown"] is True
    assert groups["Unknown"]["eligible_sample_count"] == 1
    model_a = {item["model"]: item for item in groups["Unknown"]["models"]}["model-a"]
    assert model_a["sample_coverage"] == 1
    assert model_a["cer"] == 0


def test_report_cli_emits_json_and_inspect_cli_writes_html(tmp_path, capsys):
    run = _make_run(tmp_path / "run")
    assert main(["report", "--run", str(run), "--group-by", "difficulty"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["group_by"] == "difficulty"
    assert main(["inspect", "--run", str(run), "--worst", "2"]) == 0
    output = capsys.readouterr().out
    assert "model/sample results selected" in output
    assert (run / "inspection.html").is_file()


def test_inspection_rejects_unknown_model_and_bad_worst_argument(tmp_path, capsys):
    run = _make_run(tmp_path / "run")
    with pytest.raises(ValueError, match="positive integer"):
        inspect_run(run, worst=0)
    with pytest.raises(SystemExit) as error:
        main(["inspect", "--run", str(run), "--worst", "0"])
    assert error.value.code == 2
    assert "positive integer" in capsys.readouterr().err


def test_manifest_integrity_and_crop_hash_are_checked(tmp_path):
    run = _make_run(tmp_path / "run")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["schema_version"] = 2
    manifest["integrity"] = "invalid"
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity check failed"):
        inspect_run(run)

    run = _make_run(tmp_path / "second-run")
    manifest = json.loads((run / "manifest.json").read_text())
    image_path = run / "crops" / "s1.png"
    manifest["samples"][0]["hashes"] = {"crop": hashlib.sha256(image_path.read_bytes()).hexdigest()}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    image_path.write_bytes(b"changed image content")
    with pytest.raises(ValueError, match="image hash does not match"):
        write_inspection_html(run, tmp_path / "review.html", worst=4)
