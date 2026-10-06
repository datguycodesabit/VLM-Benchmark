from __future__ import annotations

import copy
import csv
import hashlib
import json

import pytest

from vlm_bench import comparison, runner
from vlm_bench.snapshot import fingerprint


@pytest.fixture
def samples():
    return [
        {
            "id": "p1",
            "reference": "hello",
            "hashes": {"crop": hashlib.sha256(b"crop-p1").hexdigest()},
            "crop_bbox": [0, 0, 100, 30],
            "preprocess": "enhanced",
            "content_type": "prose",
            "metadata": {"content_type": "prose", "split": "test", "source_document": "doc-a"},
            "verified": True,
            "split": "test",
            "document_id": "doc-a",
        },
        {
            "id": "p2",
            "reference": "world",
            "hashes": {"crop": hashlib.sha256(b"crop-p2").hexdigest()},
            "crop_bbox": [0, 0, 100, 30],
            "preprocess": "enhanced",
            "content_type": "prose",
            "metadata": {"content_type": "prose", "split": "test", "source_document": "doc-b"},
            "verified": True,
            "split": "test",
            "document_id": "doc-b",
        },
        {
            "id": "eq1",
            "reference": "x = 1",
            "hashes": {"crop": hashlib.sha256(b"crop-eq1").hexdigest()},
            "crop_bbox": [0, 0, 100, 30],
            "preprocess": "enhanced",
            "content_type": "equation",
            "metadata": {"content_type": "equation", "split": "test", "source_document": "doc-c"},
            "verified": True,
            "split": "test",
            "document_id": "doc-c",
        },
    ]


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _record(model, sample_id, prediction, reference="row reference", **values):
    return {
        "model": model,
        "sample_id": sample_id,
        "status": "success",
        "prediction": prediction,
        "reference": reference,
        "metrics": {"cer": 0, "exact_match": True},
        "latency_seconds": 0.2,
        "truncated": False,
        **values,
    }


def _make_run(tmp_path, name, samples, models, records, **manifest_values):
    path = tmp_path / name
    path.mkdir()
    # Normal benchmark runs already have this file. Keeping it present also
    # makes the fixture's before/after directory contents directly comparable.
    (path / ".lock").touch()
    manifest = {
        "schema_version": 2,
        "benchmark_version": "test-version",
        "created_at": f"2026-10-01T00:00:0{len(name)}+00:00",
        "status": "complete",
        "preprocess": "enhanced",
        "scoring_version": "1",
        "samples": copy.deepcopy(samples),
        "models": models,
        "model_info": {model: {"digest": f"revision-{model}"} for model in models},
        "options": {"temperature": 0, "seed": 42, "num_predict": 256},
        "settings": {},
        "provider_controls": {},
        "prompt": "Read handwriting exactly.",
        "math_prompt": "Read equations.",
        "prompt_hashes": {"prose": "hash-prose", "equation": "hash-equation"},
        "timeout": 60,
        "warmup": False,
        "research": {"bootstrap_replicates": 20, "bootstrap_seed": 7},
        **manifest_values,
    }
    manifest["benchmark_fingerprint"] = fingerprint(manifest["samples"], manifest["preprocess"])
    manifest["integrity"] = runner._digest(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"integrity", "status", "updated_at"}
        }
    )
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_jsonl(path / "results.jsonl", records)
    _write_jsonl(path / "warmups.jsonl", [])
    return path


def _entry_result(report, track, model):
    return next(row for row in report["tracks"][track]["model_results"] if row["model"] == model)


def test_compare_matched_runs_relabels_entries_and_preserves_sources(tmp_path, samples):
    first = _make_run(
        tmp_path,
        "run-a",
        samples,
        ["ollama:local", "trocr:entry-id"],
        [
            _record(
                "ollama:local",
                "p1",
                "hello",
                cache_hit=True,
                cache_lookup_seconds=0.003,
                cache_bypass_reason=None,
                latency_seconds=None,
                inference_latency_seconds=None,
            ),
            _record("ollama:local", "p2", "world", cache_hit=False),
            _record("ollama:local", "eq1", "x = 1"),
            _record("trocr:entry-id", "p1", "hello"),
            _record("trocr:entry-id", "p2", "world"),
            _record("trocr:entry-id", "eq1", "x = 1"),
        ],
        options={"temperature": 0, "seed": 42, "num_predict": 256},
        settings={"ollama:local": {"num_predict": 256}},
        provider_controls={
            "ollama:local": {
                "requested": {"temperature": 0, "seed": 42, "num_predict": 256},
                "effective": {"temperature": 0, "seed": 42, "num_predict": 256},
                "unsupported": [],
            }
        },
        prompt="Prompt A",
        strict_research=True,
        protocol="writer-disjoint",
        research_protocol_version=1,
    )
    second = _make_run(
        tmp_path,
        "run-b",
        samples,
        ["ollama:local"],
        [
            _record("ollama:local", "p1", "hellx"),
            _record("ollama:local", "p2", "world"),
            _record("ollama:local", "eq1", "x = 1"),
        ],
        options={"temperature": 0.2, "seed": 99, "num_predict": 512},
        settings={"ollama:local": {"num_predict": 512}},
        provider_controls={
            "ollama:local": {
                "requested": {"temperature": 0.2, "seed": 99, "num_predict": 512},
                "effective": {"temperature": 0.2, "seed": 99, "num_predict": 512},
                "unsupported": [],
            }
        },
        prompt="Prompt B",
        model_info={"ollama:local": {"digest": "revision-b"}},
        strict_research=True,
        protocol="writer-disjoint",
        research_protocol_version=1,
    )
    before = {
        path: {child.name: child.read_bytes() for child in path.iterdir() if child.is_file()}
        for path in (first, second)
    }
    output = tmp_path / "comparison"

    report = comparison.compare([first, second], output)

    first_entry = "ollama:local@run-a"
    second_entry = "ollama:local@run-b"
    trocr_entry = "trocr:entry-id@run-a"
    assert report["benchmark_fingerprint"]
    assert report["runs"][0]["evaluation"]["strict_research"] is True
    assert report["runs"][0]["evaluation"]["protocol"] == "writer-disjoint"
    assert report["runs"][0]["evaluation"]["research_protocol_version"] == 1
    assert report["entries"][first_entry]["prompt"] == "Prompt A"
    assert report["entries"][first_entry]["settings"] == {"num_predict": 256}
    assert report["entries"][second_entry]["settings"] == {"num_predict": 512}
    assert report["entries"][second_entry]["effective_settings"]["options"]["seed"] == 99
    assert report["entries"][second_entry]["provider_controls"]["effective"]["num_predict"] == 512
    assert report["entries"][second_entry]["prompt_hashes"] == {
        "prose": "hash-prose",
        "equation": "hash-equation",
    }
    assert report["entries"][second_entry]["prose_prompt_hash"] == "hash-prose"
    assert report["entries"][second_entry]["math_prompt_hash"] == "hash-equation"
    assert report["entries"][second_entry]["model_revision"] == {"digest": "revision-b"}
    assert report["model_info"][trocr_entry] == {"digest": "revision-trocr:entry-id"}
    assert _entry_result(report, "prose", first_entry)["cer"] == 0
    assert _entry_result(report, "prose", first_entry)["cache_hit_count"] == 1
    assert _entry_result(report, "prose", first_entry)["measured_latency_sample_count"] == 1
    assert _entry_result(report, "prose", second_entry)["cer"] == pytest.approx(1 / 10)
    assert _entry_result(report, "equation", trocr_entry)["status"] == "unsupported"
    assert _entry_result(report, "equation", trocr_entry)["rank"] is None
    assert (output / "comparison.json").is_file()
    assert (output / "comparison.csv").is_file()
    assert (output / "paired.csv").is_file()
    assert (output / "costs.csv").is_file()
    comparison_header = (output / "comparison.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert all(
        field in comparison_header
        for field in (
            "run",
            "original_model",
            "provider_controls",
            "effective_settings",
            "prompt_hashes",
            "prose_prompt_hash",
            "math_prompt_hash",
            "model_revision",
            "cache_hit_count",
            "measured_latency_sample_count",
        )
    )
    with (output / "comparison.csv").open(encoding="utf-8-sig", newline="") as file:
        comparison_rows = csv.DictReader(file)
        local_prose = next(
            row
            for row in comparison_rows
            if row["track"] == "prose" and row["model"] == first_entry
        )
    assert local_prose["cache_hit_count"] == "1"
    assert local_prose["measured_latency_sample_count"] == "1"
    assert "ci95_low" in (output / "paired.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    after = {
        path: {child.name: child.read_bytes() for child in path.iterdir() if child.is_file()}
        for path in (first, second)
    }
    assert before == after


def test_compare_rejects_snapshot_mismatch(tmp_path, samples):
    changed = copy.deepcopy(samples)
    changed[0]["reference"] = "hullo"
    first = _make_run(tmp_path, "run-a", samples, ["one"], [])
    second = _make_run(tmp_path, "run-b", changed, ["two"], [])
    output = tmp_path / "comparison"

    with pytest.raises(ValueError, match="same benchmark snapshot"):
        comparison.compare([first, second], output)
    assert not output.exists()


def test_compare_rescores_from_frozen_reference_and_marks_incomplete(tmp_path, samples):
    first = _make_run(
        tmp_path,
        "run-a",
        samples,
        ["ollama:local"],
        [
            _record("ollama:local", "p1", "hellx", reference="forged row reference"),
        ],
        scoring_version="0",
    )
    second = _make_run(tmp_path, "run-b", samples, ["openai:other-model"], [])

    report = comparison.compare([first, second], tmp_path / "comparison")
    result = _entry_result(report, "prose", "ollama:local@run-a")
    assert report["scoring_version"] == "1"
    assert report["runs"][0]["scoring_version"] == "0"
    assert result["cer"] == pytest.approx(1 / 5)
    assert result["status"] == "incomplete"
    assert result["failed_sample_count"] == 0
    assert result["missing_sample_ids"] == ["p2"]
    assert _entry_result(report, "prose", "openai:other-model@run-b")["status"] == "not_run"


@pytest.mark.parametrize("duplicate", [False, True])
def test_compare_rejects_unknown_or_duplicate_result_pairs(tmp_path, samples, duplicate):
    first_records = [_record("one", "p1", "hello")]
    if duplicate:
        first_records.append(_record("one", "p1", "hello"))
    else:
        first_records.append(_record("one", "not-in-manifest", "hello"))
    first = _make_run(tmp_path, "run-a", samples, ["one"], first_records)
    second = _make_run(tmp_path, "run-b", samples, ["two"], [])

    expected = "duplicate model/sample pair" if duplicate else "unknown model/sample pair"
    with pytest.raises(ValueError, match=expected):
        comparison.compare([first, second], tmp_path / "comparison")


def test_compare_rejects_legacy_runs_with_actionable_message(tmp_path, samples):
    first = _make_run(tmp_path, "run-a", samples, ["one"], [])
    second = _make_run(tmp_path, "run-b", samples, ["two"], [])
    manifest_path = first / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("benchmark_fingerprint")
    manifest["integrity"] = runner._digest(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"integrity", "status", "updated_at"}
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy run.*Existing runs can still be exported"):
        comparison.compare([first, second], tmp_path / "comparison")


def test_compare_refuses_duplicate_runs_and_existing_output(tmp_path, samples):
    first = _make_run(tmp_path, "run-a", samples, ["one"], [])
    with pytest.raises(ValueError, match="at least two distinct"):
        comparison.compare([first, first], tmp_path / "comparison")
    second = _make_run(tmp_path, "run-b", samples, ["two"], [])
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        comparison.compare([first, second], output)


def test_compare_canonicalizes_bare_ollama_model_ids(tmp_path, samples):
    first = _make_run(
        tmp_path,
        "run-a",
        samples,
        ["qwen:tag"],
        [_record("qwen:tag", "p1", "hello")],
    )
    second = _make_run(tmp_path, "run-b", samples, ["ollama:other"], [])

    report = comparison.compare([first, second], tmp_path / "comparison")

    assert "ollama:qwen:tag@run-a" in report["entries"]
    assert report["entries"]["ollama:qwen:tag@run-a"]["model"] == "qwen:tag"


def test_compare_preserves_task_scores_and_separates_per_run_scorer_options(
    tmp_path, samples, monkeypatch
):
    from vlm_bench import task_metrics

    def saved_task_score(status, value):
        return {
            "version": 1,
            "scores": {
                "formula_render_similarity": {"status": status, "value": value},
                "critical_expression_accuracy": {"status": "not_annotated", "value": None},
                "reading_order_accuracy": {"status": "not_annotated", "value": None},
            },
            "errors": [],
        }

    first_record = _record("ollama:literal", "eq1", "x = 1", reference="wrong")
    first_record["metrics"] = {"cer": 0.8, "task_metrics": saved_task_score("disabled", None)}
    second_record = _record("openai:rendered", "eq1", r"x\;=\;1", reference="wrong")
    second_record["metrics"] = {"cer": 0.1, "task_metrics": saved_task_score("scored", 0.95)}
    first = _make_run(
        tmp_path,
        "run-a",
        samples,
        ["ollama:literal"],
        [first_record],
        formula_rendering=False,
        task_metrics_version=1,
    )
    second = _make_run(
        tmp_path,
        "run-b",
        samples,
        ["openai:rendered"],
        [second_record],
        formula_rendering=True,
        task_metrics_version=1,
        formula_renderer={"name": "matplotlib-mathtext", "version": "3.10.3"},
    )
    monkeypatch.setattr(
        task_metrics,
        "score_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must use saved scores")),
    )

    output = tmp_path / "comparison"
    report = comparison.compare([first, second], output)
    assert "task_metrics" not in report
    assert report["runs"][0]["task_metrics"]["formula_rendering_enabled"] is False
    assert report["runs"][1]["task_metrics"]["formula_rendering_enabled"] is True
    assert report["runs"][1]["evaluation"]["formula_renderer"]["version"] == "3.10.3"
    assert (
        report["entries"]["ollama:literal@run-a"]["task_metrics_options"]["formula_rendering"]
        is False
    )
    assert (
        report["entries"]["openai:rendered@run-b"]["task_metrics_options"]["formula_rendering"]
        is True
    )
    equation_results = report["tracks"]["equation"]["model_results"]
    ranked = sorted(equation_results, key=lambda row: row["rank"])
    assert ranked[0]["model"] == "ollama:literal@run-a"
    metric_report = next(
        model
        for model in report["runs"][1]["task_metrics"]["tracks"]["equation"]["models"]
        if model["model"] == "openai:rendered"
    )
    assert metric_report["metrics"]["formula_render_similarity"]["status_counts"] == {"scored": 1}
    with (output / "comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "task_metrics_options" in rows[0]
