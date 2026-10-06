from __future__ import annotations

import csv
import json

import pytest
from PIL import Image

from vlm_bench import engine
from vlm_bench.comparison import compare
from vlm_bench.export import export_run
from vlm_bench.import_predictions import import_predictions
from vlm_bench.inspection import inspect_run
from vlm_bench.research import research_reports
from vlm_bench.runner import read_records
from vlm_bench.snapshot import freeze


def _prepared_snapshot(root):
    data = root / "source"
    (data / "images").mkdir(parents=True)
    (data / "text").mkdir()
    Image.new("RGB", (80, 24), "white").save(data / "images" / "s1.png")
    Image.new("RGB", (80, 24), "white").save(data / "images" / "s2.png")
    (data / "text" / "s1.txt").write_text("hello", encoding="utf-8")
    (data / "text" / "s2.txt").write_text("x = 1", encoding="utf-8")
    metadata = [
        {
            "id": "s1",
            "content_type": "prose",
            "split": "test",
            "verified": True,
            "source_document": "doc-1",
        },
        {
            "id": "s2",
            "content_type": "equation",
            "split": "test",
            "verified": True,
            "source_document": "doc-2",
            "annotations": {
                "critical_expressions": ["x"],
                "reading_order": [],
            },
        },
    ]
    (data / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8"
    )
    prepared = root / "prepared"
    freeze(data, prepared)
    return prepared


def _prediction_file(path, *, partial=False):
    rows = [
        {
            "sample_id": "s1",
            "prediction": "hello",
            "latency_seconds": 1.25,
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    ]
    if not partial:
        rows.append({"sample_id": "s2", "prediction": "x = 1", "latency_seconds": 2.5})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class NativeBackend:
    def __init__(self, provider, *args):
        self.provider = provider

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def validate_model(self, _model):
        return {"revision": "native-test"}

    def transcribe(self, _model, image_path, _prompt, _options):
        prediction = {"s1": "hello", "s2": "x = 1"}[image_path.stem]
        return {"message": {"content": prediction}, "done_reason": "stop"}

    def unload(self, _model):
        pass


def test_import_matches_native_scoring_and_supports_downstream_tools(tmp_path):
    prepared = _prepared_snapshot(tmp_path)
    predictions = _prediction_file(tmp_path / "predictions.jsonl")
    provenance_file = tmp_path / "predictions.jsonl.provenance.json"
    provenance = {
        "schema_version": 1,
        "version": "5.4.0",
        "config": {"page_segmentation_mode": 6},
        "preprocessing": "grayscale, threshold=180",
        "scope": "line crops",
    }
    provenance_file.write_text(json.dumps(provenance), encoding="utf-8")

    imported = import_predictions(
        prepared,
        predictions,
        tmp_path / "imported",
        system="tesseract",
        provenance_file=provenance_file,
    )
    native = engine.run(
        None,
        ["ollama:native-test"],
        tmp_path / "native",
        prepared=prepared,
        warmup=False,
        backend_factory=NativeBackend,
        progress=lambda _message: None,
    )
    imported_rows = {row["sample_id"]: row for row in read_records(imported / "results.jsonl")}
    native_rows = {row["sample_id"]: row for row in read_records(native / "results.jsonl")}
    assert set(imported_rows) == set(native_rows)
    for sample_id in imported_rows:
        assert imported_rows[sample_id]["reference"] == native_rows[sample_id]["reference"]
        assert imported_rows[sample_id]["metrics"] == native_rows[sample_id]["metrics"]

    manifest = json.loads((imported / "manifest.json").read_text(encoding="utf-8"))
    selector = "external:tesseract"
    assert manifest["models"] == [selector]
    assert manifest["model_info"][selector]["provenance"] == provenance
    assert manifest["external_predictions"]["provenance_file_sha256"]
    assert imported_rows["s1"]["timing_source"] == "external"
    assert imported_rows["s1"]["usage_source"] == "external"
    assert imported_rows["s1"]["reference"] == "hello"

    assert inspect_run(imported)["eligible_result_count"] == 2
    assert (imported / "task_metrics.csv") in export_run(imported, ["csv"])
    assert (imported / "research.json").is_file()
    with (imported / "task_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert next(csv.DictReader(handle))["track"] in {"equation", "prose", "word"}

    comparison = compare([imported, native], tmp_path / "comparison")
    assert "external:tesseract@imported" in comparison["entries"]
    with pytest.raises(FileExistsError, match="already exists"):
        import_predictions(
            prepared,
            predictions,
            imported,
            system="tesseract",
            provenance="Tesseract 5.4.0, line mode",
        )


def test_incomplete_import_is_coverage_incomplete_and_unranked(tmp_path):
    prepared = _prepared_snapshot(tmp_path)
    predictions = _prediction_file(tmp_path / "partial.jsonl", partial=True)
    imported = import_predictions(
        prepared,
        predictions,
        tmp_path / "partial-run",
        system="paddleocr",
        provenance="PaddleOCR 3.x, line crops",
    )
    records = read_records(imported / "results.jsonl")
    manifest = json.loads((imported / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete"
    report = research_reports(records, manifest)
    equation_result = report["tracks"]["equation"]["model_results"][0]
    assert equation_result["status"] == "incomplete"
    assert equation_result["rank"] is None
    assert equation_result["missing_sample_ids"] == ["s2"]
    assert inspect_run(imported)["eligible_sample_count"] == 2


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"sample_id":"s1","prediction":"a"}\n{"sample_id":"s1","prediction":"b"}\n', "Duplicate"),
        ('{"sample_id":"missing","prediction":"a"}\n', "unknown sample_id"),
        ('{"sample_id":"s1","prediction":"a","reference":"forged"}\n', "unsupported fields"),
        ('{"sample_id":"s1","prediction":"a","latency_seconds":NaN}\n', "non-finite"),
        ('{"sample_id":"s1","prediction":"a","latency_seconds":Infinity}\n', "non-finite"),
        ('{"sample_id":"s1","prediction":"a","usage":{"cost":NaN}}\n', "non-finite"),
    ],
)
def test_import_rejects_invalid_prediction_files_before_creating_output(tmp_path, payload, message):
    prepared = _prepared_snapshot(tmp_path)
    predictions = tmp_path / "invalid.jsonl"
    predictions.write_text(payload, encoding="utf-8")
    output = tmp_path / "new-run"
    with pytest.raises(ValueError, match=message):
        import_predictions(
            prepared,
            predictions,
            output,
            system="test-system",
            provenance="test source",
        )
    assert not output.exists()


def test_cli_import_requires_provenance_and_uses_external_model_identity(
    tmp_path, monkeypatch, capsys
):
    from vlm_bench import cli

    prepared = _prepared_snapshot(tmp_path)
    predictions = _prediction_file(tmp_path / "predictions.jsonl")
    monkeypatch.setattr(cli, "_report", lambda _run: None)
    assert (
        cli.main(
            [
                "import",
                "--prepared",
                str(prepared),
                "--predictions",
                str(predictions),
                "--output",
                str(tmp_path / "cli-import"),
                "--system",
                "external-ocr",
                "--provenance",
                "versioned command line",
            ]
        )
        == 0
    )
    assert "Imported run:" in capsys.readouterr().out
