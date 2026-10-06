import json
from pathlib import Path

import pytest
from PIL import Image

from vlm_bench import engine
from vlm_bench.runner import read_records


@pytest.fixture
def paired(tmp_path):
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "text").mkdir()
    for i in range(3):
        Image.new("RGB", (80 + i, 20), "white").save(data / "images" / f"s{i}.png")
        (data / "text" / f"s{i}.txt").write_text("hello")
    return data


class Backend:
    calls = []
    failure = None
    identity = "fixed"

    def __init__(self, provider, *args):
        self.provider = provider

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def validate_model(self, model):
        return {"model": model, "revision": self.identity}

    def transcribe(self, model, path, prompt, options):
        assert "hello" not in prompt
        if self.failure == "interrupt" and len(self.calls) == 1:
            raise KeyboardInterrupt
        if self.failure == "error" and path.stem == "s1":
            raise RuntimeError("unavailable")
        if self.failure == "quota" and self.provider == "chatgpt":
            from vlm_bench.backends import BackendPaused

            raise BackendPaused(self.provider, model, "limit")
        self.calls.append((self.provider, path.stem))
        return {"message": {"content": "hello"}, "done_reason": "stop"}

    def unload(self, model):
        pass


@pytest.fixture(autouse=True)
def reset():
    Backend.calls = []
    Backend.failure = None
    Backend.identity = "fixed"


def test_cross_provider_same_frozen_inputs(paired, tmp_path):
    directory = engine.run(
        paired,
        ["ollama:first", "trocr:second"],
        tmp_path / "runs",
        warmup=False,
        backend_factory=Backend,
    )
    rows = read_records(directory / "results.jsonl")
    assert len(rows) == 6
    assert {r["sample_id"] for r in rows if r["provider"] == "ollama"} == {
        r["sample_id"] for r in rows if r["provider"] == "trocr"
    }
    assert all(r["metrics"]["cer"] == 0 for r in rows)


def test_resume_frozen_inputs_without_live_dataset(paired, tmp_path):
    Backend.failure = "interrupt"
    with pytest.raises(KeyboardInterrupt):
        engine.run(
            paired, ["ollama:first"], tmp_path / "runs", warmup=False, backend_factory=Backend
        )
    directory = next((tmp_path / "runs").iterdir())
    (paired / "text" / "s0.txt").write_text("changed after freezing")
    Backend.failure = None
    engine.resume(directory, backend_factory=Backend)
    assert len(read_records(directory / "results.jsonl")) == 3
    assert len(Backend.calls) == 3


def test_retry_failed_preserves_history_and_successes(paired, tmp_path):
    Backend.failure = "error"
    directory = engine.run(
        paired, ["ollama:first"], tmp_path / "runs", warmup=False, backend_factory=Backend
    )
    Backend.failure = None
    engine.resume(directory, retry_failed=True, backend_factory=Backend)
    assert len(Backend.calls) == 3
    assert len(read_records(directory / "results.jsonl")) == 3
    history = read_records(directory / "attempts.jsonl")
    assert len([event for event in history if "event_kind" not in event]) == 1


def test_quota_pauses_only_cloud_and_never_switches_billing(paired, tmp_path):
    Backend.failure = "quota"
    directory = engine.run(
        paired,
        ["chatgpt:first", "ollama:first"],
        tmp_path / "runs",
        warmup=False,
        backend_factory=Backend,
    )
    assert json.loads((directory / "manifest.json").read_text())["status"] == "paused"
    assert len(read_records(directory / "results.jsonl")) == 3
    assert all(p == "ollama" for p, _ in Backend.calls)


def test_resume_rejects_changed_model_or_image(paired, tmp_path):
    directory = engine.run(
        paired, ["ollama:first"], tmp_path / "runs", warmup=False, backend_factory=Backend
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    Path(manifest["samples"][0]["crop_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Saved image changed"):
        engine.resume(directory, backend_factory=Backend)


def test_dry_run_never_transcribes(paired):
    report = engine.preview(paired, ["ollama:first", "openai:second"], backend_factory=Backend)
    assert report["sample_count"] == 3
    assert report["models"][1]["execution"] == "cloud"
    assert report["strict_research"] is False
    assert report["protocol"] == "document-disjoint"
    assert Backend.calls == []


def test_strict_source_run_uses_test_split_and_persists_audit(paired, tmp_path):
    metadata = [
        {
            "id": f"s{i}",
            "split": "test",
            "verification_status": "verified",
            "source_document": f"document-{i}",
            "writer_id": f"writer-{i}",
        }
        for i in range(3)
    ]
    (paired / "metadata.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in metadata)
    )

    directory = engine.run(
        paired,
        ["ollama:first"],
        tmp_path / "runs",
        warmup=False,
        strict_research=True,
        backend_factory=Backend,
    )

    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["research_protocol_version"] == 1
    assert manifest["strict_research"] is True
    assert manifest["protocol"] == "document-disjoint"
    assert manifest["split"] == "test"
    assert manifest["research"]["eligible_sample_count"] == 3
    assert manifest["source_audit"]["sample_count"] == 3
    assert manifest["validation_scope"] == "source-dataset-and-selected-test-samples"


@pytest.mark.parametrize("protocol", ["document-disjoint", "writer-disjoint"])
def test_strict_source_run_rejects_cross_split_document_leakage(paired, tmp_path, protocol):
    metadata = [
        {
            "id": "s0",
            "split": "train",
            "verification_status": "verified",
            "source_document": "shared-document",
            "writer_id": "writer-0",
        },
        {
            "id": "s1",
            "split": "test",
            "verification_status": "verified",
            "source_document": "shared-document",
            "writer_id": "writer-1",
        },
        {
            "id": "s2",
            "split": "test",
            "verification_status": "verified",
            "source_document": "document-2",
            "writer_id": "writer-2",
        },
    ]
    (paired / "metadata.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in metadata)
    )

    with pytest.raises(ValueError, match="split leakage"):
        engine.run(
            paired,
            ["ollama:first"],
            tmp_path / "runs",
            strict_research=True,
            protocol=protocol,
            backend_factory=Backend,
        )

    assert Backend.calls == []
    assert not (tmp_path / "runs").exists()


def test_writer_disjoint_protocol_runs_with_distinct_documents_and_writers(paired, tmp_path):
    metadata = [
        {
            "id": "s0",
            "split": "train",
            "verification_status": "verified",
            "source_document": "document-0",
            "writer_id": "writer-0",
        },
        {
            "id": "s1",
            "split": "test",
            "verification_status": "verified",
            "source_document": "document-1",
            "writer_id": "writer-1",
        },
        {
            "id": "s2",
            "split": "test",
            "verification_status": "verified",
            "source_document": "document-2",
            "writer_id": "writer-2",
        },
    ]
    (paired / "metadata.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in metadata)
    )

    directory = engine.run(
        paired,
        ["ollama:first"],
        tmp_path / "runs",
        warmup=False,
        strict_research=True,
        protocol="writer-disjoint",
        backend_factory=Backend,
    )

    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["protocol"] == "writer-disjoint"
    assert manifest["research"]["eligible_sample_count"] == 2


def test_writer_disjoint_source_audit_requires_writer_ids_in_every_split(paired, tmp_path):
    metadata = [
        {
            "id": "s0",
            "split": "train",
            "verification_status": "verified",
            "source_document": "document-0",
        },
        {
            "id": "s1",
            "split": "test",
            "verification_status": "verified",
            "source_document": "document-1",
            "writer_id": "writer-1",
        },
        {
            "id": "s2",
            "split": "test",
            "verification_status": "verified",
            "source_document": "document-2",
            "writer_id": "writer-2",
        },
    ]
    (paired / "metadata.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in metadata)
    )

    with pytest.raises(ValueError, match="missing writer_id.*s0"):
        engine.run(
            paired,
            ["ollama:first"],
            tmp_path / "runs",
            strict_research=True,
            protocol="writer-disjoint",
            backend_factory=Backend,
        )

    assert Backend.calls == []
    assert not (tmp_path / "runs").exists()


def test_strict_manifest_preserves_dataset_duplicate_review(paired, tmp_path, monkeypatch):
    from vlm_bench import dataset

    metadata = [
        {
            "id": f"s{i}",
            "split": "test",
            "verification_status": "verified",
            "source_document": f"document-{i}",
            "writer_id": f"writer-{i}",
        }
        for i in range(3)
    ]
    (paired / "metadata.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in metadata)
    )
    check_dataset = dataset.check_dataset

    def check_with_duplicate_findings(data_dir, layout="auto"):
        report = check_dataset(data_dir, layout=layout)
        report["findings"].append(
            {
                "type": "perceptual_duplicate_review",
                "sample_ids": ["s0", "s1"],
                "pairs": [{"sample_ids": ["s0", "s1"], "hamming_distance": 2}],
            }
        )
        report["duplicate_content"]["images"] = [["s0", "s1"]]
        return report

    monkeypatch.setattr(dataset, "check_dataset", check_with_duplicate_findings)
    directory = engine.run(
        paired,
        ["ollama:first"],
        tmp_path / "runs",
        warmup=False,
        strict_research=True,
        backend_factory=Backend,
    )

    source_audit = json.loads((directory / "manifest.json").read_text())["source_audit"]
    assert source_audit["sample_count"] == 3
    assert any(
        finding["type"] == "perceptual_duplicate_review"
        for finding in source_audit["dataset_findings"]
    )
    assert source_audit["duplicate_content"]["images"] == [["s0", "s1"]]


def test_strict_prepared_run_rejects_missing_metadata_before_backend(paired, tmp_path):
    from vlm_bench.snapshot import freeze

    snapshot = tmp_path / "prepared"
    freeze(paired, snapshot)
    backend_constructions = []

    def unreachable_backend(*args):
        backend_constructions.append(args)
        raise AssertionError("backend must not be constructed")

    with pytest.raises(ValueError):
        engine.run(
            models=["ollama:first"],
            prepared=snapshot,
            output_dir=tmp_path / "runs",
            strict_research=True,
            backend_factory=unreachable_backend,
        )

    assert backend_constructions == []


def test_formula_rendering_preflights_before_backend_and_saves_task_metrics(
    paired, tmp_path, monkeypatch
):
    from vlm_bench import task_metrics

    (paired / "metadata.jsonl").write_text(
        json.dumps(
            {
                "id": "s0",
                "annotations": {
                    "critical_expressions": ["hello"],
                    "reading_order": [["heading", "body"]],
                },
            }
        )
        + "\n"
    )
    events = []
    scored_samples = []
    renderer = {
        "name": "matplotlib-mathtext",
        "version": "3.10.3",
        "task_metrics_version": 1,
        "engine": "MathTextParser(agg), DejaVu Sans, 18 pt, 96 dpi",
    }

    def preflight_renderer():
        events.append("preflight")
        return renderer

    def score_task(prediction, reference, sample, *, formula_rendering=False):
        scored_samples.append((sample, formula_rendering))
        return {
            "version": 1,
            "scores": {"formula_render_similarity": {"status": "scored", "value": 1.0}},
            "errors": [],
        }

    monkeypatch.setattr(task_metrics, "preflight_renderer", preflight_renderer)
    monkeypatch.setattr(task_metrics, "score_task", score_task)

    class OrderedBackend(Backend):
        def __init__(self, provider, *args):
            events.append("backend")
            super().__init__(provider, *args)

    preview = engine.preview(
        paired,
        ["ollama:first"],
        formula_rendering=True,
        backend_factory=OrderedBackend,
    )
    assert preview["formula_rendering"] is True
    assert preview["task_metrics_version"] == 1
    assert preview["formula_renderer"] == renderer
    events.clear()

    directory = engine.run(
        paired,
        ["ollama:first"],
        tmp_path / "runs",
        warmup=False,
        formula_rendering=True,
        backend_factory=OrderedBackend,
    )

    manifest = json.loads((directory / "manifest.json").read_text())
    rows = read_records(directory / "results.jsonl")
    assert events[0] == "preflight"
    assert events[1] == "backend"
    assert manifest["schema_version"] == 2
    assert manifest["scoring_version"] == engine.SCORING_VERSION
    assert manifest["formula_rendering"] is True
    assert manifest["task_metrics_version"] == 1
    assert manifest["formula_renderer"] == renderer
    assert all(row["metrics"]["task_metrics"]["version"] == 1 for row in rows)
    assert all("cer" in row["metrics"] for row in rows)
    assert any(
        sample["metadata"]["annotations"]["critical_expressions"] == ["hello"] and formula_rendering
        for sample, formula_rendering in scored_samples
    )


def test_formula_renderer_preflight_failure_precedes_backend_creation(
    paired, tmp_path, monkeypatch
):
    from vlm_bench import task_metrics

    def failed_preflight():
        raise RuntimeError("renderer unavailable")

    backend_constructions = []

    def unreachable_backend(*args):
        backend_constructions.append(args)
        raise AssertionError("backend must not be constructed")

    monkeypatch.setattr(task_metrics, "preflight_renderer", failed_preflight)
    with pytest.raises(RuntimeError, match="renderer unavailable"):
        engine.run(
            paired,
            ["ollama:first"],
            tmp_path / "runs",
            formula_rendering=True,
            backend_factory=unreachable_backend,
        )

    assert backend_constructions == []
    assert not (tmp_path / "runs").exists()


def test_resume_legacy_manifest_defaults_formula_rendering_off(paired, tmp_path, monkeypatch):
    from vlm_bench import task_metrics

    scored_options = []

    def score_task(prediction, reference, sample, *, formula_rendering=False):
        scored_options.append(formula_rendering)
        return {"version": 1, "scores": {}, "errors": []}

    def unexpected_preflight():
        raise AssertionError("legacy resume must not preflight formula rendering")

    monkeypatch.setattr(task_metrics, "score_task", score_task)
    monkeypatch.setattr(task_metrics, "preflight_renderer", unexpected_preflight)
    interrupt = {"once": True}

    class InterruptingBackend(Backend):
        def transcribe(self, model, path, prompt, options):
            if path.stem == "s1" and interrupt["once"]:
                interrupt["once"] = False
                raise KeyboardInterrupt
            return super().transcribe(model, path, prompt, options)

    with pytest.raises(KeyboardInterrupt):
        engine.run(
            paired,
            ["ollama:first"],
            tmp_path / "runs",
            warmup=False,
            formula_rendering=False,
            backend_factory=InterruptingBackend,
        )

    directory = next((tmp_path / "runs").iterdir())
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for key in ("formula_rendering", "task_metrics_version", "formula_renderer"):
        manifest.pop(key, None)
    immutable = {
        k: v for k, v in manifest.items() if k not in {"integrity", "status", "updated_at"}
    }
    manifest["integrity"] = engine._digest(immutable)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False) + "\n")

    engine.resume(directory, backend_factory=InterruptingBackend)

    assert scored_options == [False, False, False]


def test_local_request_timing_separates_known_model_load(paired, tmp_path):
    class TimedBackend(Backend):
        def transcribe(self, *args):
            raw = super().transcribe(*args)
            raw["load_duration"] = 0
            return raw

    directory = engine.run(
        paired, ["ollama:first"], tmp_path / "runs", warmup=False, backend_factory=TimedBackend
    )
    for row in read_records(directory / "results.jsonl"):
        assert row["inference_latency_seconds"] == row["latency_seconds"]
        assert row["load_duration_seconds"] == 0


def test_prepared_runs_share_bytes_without_source_or_snapshot(paired, tmp_path):
    import shutil

    from vlm_bench.snapshot import freeze

    snapshot = tmp_path / "benchmark"
    freeze(paired, snapshot, limit=2, seed=42)
    shutil.rmtree(paired)
    seen = []

    class RecordingBackend(Backend):
        def transcribe(self, model, path, prompt, options):
            seen.append((self.provider, path.read_bytes()))
            return super().transcribe(model, path, prompt, options)

    first = engine.run(
        models=["ollama:first"],
        prepared=snapshot,
        output_dir=tmp_path / "runs",
        warmup=False,
        backend_factory=RecordingBackend,
    )
    second = engine.run(
        models=["trocr:second"],
        prepared=snapshot,
        output_dir=tmp_path / "runs",
        warmup=False,
        backend_factory=RecordingBackend,
    )
    a, b = [json.loads((d / "manifest.json").read_text()) for d in (first, second)]
    assert a["benchmark_fingerprint"] == b["benchmark_fingerprint"]
    assert [x[1] for x in seen[:2]] == [x[1] for x in seen[2:]]
    assert a["scoring_version"] == "1"
    assert a["host"]["dependencies"]["Pillow"]
    assert a["prompt_hashes"]["prose"]
    shutil.rmtree(snapshot)
    engine.resume(first, backend_factory=RecordingBackend)
    assert len(Backend.calls) == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": 42},
        {"limit": 1},
        {"preprocess": "original"},
        {"layout": "paired"},
    ],
)
def test_prepared_rejects_selection_before_provider(paired, tmp_path, kwargs):
    with pytest.raises(ValueError, match="--prepared cannot"):
        engine.run(
            models=["ollama:first"], prepared=tmp_path / "absent", backend_factory=Backend, **kwargs
        )
    assert Backend.calls == []


def test_invalid_controls_fail_before_creating_run(paired, tmp_path):
    with pytest.raises(ValueError, match="num_beams"):
        engine.run(
            paired,
            ["trocr:second"],
            tmp_path / "runs",
            settings={"trocr:second": {"num_beams": 11}},
            backend_factory=Backend,
        )
    assert not (tmp_path / "runs").exists()
    assert Backend.calls == []


def test_cloud_dry_run_records_unsupported_seed_and_token_cap(paired):
    report = engine.preview(paired, ["chatgpt:first"], backend_factory=Backend)
    controls = report["models"][0]["controls"]
    assert {"seed", "temperature", "num_predict"} <= set(controls["unsupported"])
    assert "seed" not in controls["effective"]
    assert report["verification_status_counts"] == {"unknown": 3}


@pytest.mark.parametrize("command", ["run", "preview"])
def test_external_selector_is_rejected_before_inference_setup(command, paired, tmp_path):
    backend_calls = []

    def backend_factory(*args, **kwargs):
        backend_calls.append((args, kwargs))
        raise AssertionError("external predictions must not open a backend")

    with pytest.raises(ValueError, match="vlm-bench import"):
        if command == "run":
            engine.run(
                paired,
                ["external:tesseract"],
                tmp_path / "runs",
                backend_factory=backend_factory,
            )
        else:
            engine.preview(paired, ["external:tesseract"], backend_factory=backend_factory)

    assert backend_calls == []
    assert not (tmp_path / "runs").exists()


def test_custom_prompts_and_suite_metadata_match_preview_and_run(paired, tmp_path):
    (paired / "metadata.jsonl").write_text(
        json.dumps({"id": "s0", "content_type": "equation"}) + "\n"
    )
    prose_prompt = "Carefully transcribe the handwritten prose."
    math_prompt = "Return the handwritten expression as LaTeX only."
    calls = []

    class PromptBackend(Backend):
        def transcribe(self, model, path, prompt, options):
            calls.append((path.stem, prompt, options.get("temperature"), options.get("seed")))
            return super().transcribe(model, path, prompt, options)

    settings = {"ollama:first": {"temperature": 0.6, "seed": 99}}
    preview = engine.preview(
        paired,
        ["ollama:first"],
        settings=settings,
        prose_prompt=prose_prompt,
        math_prompt=math_prompt,
        suite_condition="prompt-variant",
        repetition=1,
        repeated_measurement=True,
        backend_factory=PromptBackend,
    )
    directory = engine.run(
        paired,
        ["ollama:first"],
        tmp_path / "runs",
        warmup=False,
        settings=settings,
        prose_prompt=prose_prompt,
        math_prompt=math_prompt,
        suite_condition="prompt-variant",
        repetition=1,
        repeated_measurement=True,
        backend_factory=PromptBackend,
    )

    manifest = json.loads((directory / "manifest.json").read_text())
    expected_hashes = engine._prompt_hashes(prose_prompt, math_prompt)
    assert preview["prompt_version"] == engine.PROMPT_VERSION
    assert preview["prompt_hashes"] == manifest["prompt_hashes"] == expected_hashes
    assert preview["prompt"] == manifest["prompt"] == prose_prompt
    assert preview["math_prompt"] == manifest["math_prompt"] == math_prompt
    assert {
        key: preview[key] for key in ("suite_condition", "repetition", "repeated_measurement")
    } == {
        "suite_condition": "prompt-variant",
        "repetition": 1,
        "repeated_measurement": True,
    }
    assert {
        key: manifest[key] for key in ("suite_condition", "repetition", "repeated_measurement")
    } == {
        "suite_condition": "prompt-variant",
        "repetition": 1,
        "repeated_measurement": True,
    }
    assert manifest["provider_controls"]["ollama:first"]["effective"]["temperature"] == 0.6
    assert manifest["provider_controls"]["ollama:first"]["effective"]["seed"] == 99
    assert calls == [
        ("s0", math_prompt, 0.6, 99),
        ("s1", prose_prompt, 0.6, 99),
        ("s2", prose_prompt, 0.6, 99),
    ]


def test_resume_uses_saved_custom_prompts_for_remaining_samples(paired, tmp_path):
    prompts = []
    fail_once = {"value": True}

    class InterruptingPromptBackend(Backend):
        def transcribe(self, model, path, prompt, options):
            prompts.append((path.stem, prompt))
            if path.stem == "s1" and fail_once["value"]:
                fail_once["value"] = False
                raise KeyboardInterrupt
            return super().transcribe(model, path, prompt, options)

    custom_prose = "Transcribe this input without normalizing spelling."
    custom_math = "Transcribe symbols exactly and return LaTeX."
    with pytest.raises(KeyboardInterrupt):
        engine.run(
            paired,
            ["ollama:first"],
            tmp_path / "runs",
            warmup=False,
            prose_prompt=custom_prose,
            math_prompt=custom_math,
            backend_factory=InterruptingPromptBackend,
        )

    directory = next((tmp_path / "runs").iterdir())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    engine.resume(directory, backend_factory=InterruptingPromptBackend)

    assert prompts == [
        ("s0", custom_prose),
        ("s1", custom_prose),
        ("s1", custom_prose),
        ("s2", custom_prose),
    ]
    assert len(read_records(directory / "results.jsonl")) == 3


def test_resume_rejects_saved_prompt_hash_mismatch(paired, tmp_path):
    from vlm_bench.runner import _digest

    directory = engine.run(
        paired,
        ["ollama:first"],
        tmp_path / "runs",
        warmup=False,
        prose_prompt="Original prompt.",
        backend_factory=Backend,
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prompt"] = "Modified prompt with the old hash."
    immutable = {
        key: value
        for key, value in manifest.items()
        if key not in {"integrity", "status", "updated_at"}
    }
    manifest["integrity"] = _digest(immutable)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Saved prompt hashes"):
        engine.resume(directory, backend_factory=Backend)


def test_resume_rejects_imported_external_prediction_run(paired, tmp_path):
    from vlm_bench.runner import _digest

    directory = engine.run(
        paired, ["ollama:first"], tmp_path / "runs", warmup=False, backend_factory=Backend
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["models"] = ["external:tesseract"]
    manifest["model_info"] = {
        "external:tesseract": {
            "provider": "external",
            "system": "tesseract",
            "provenance": {"description": "imported predictions"},
        }
    }
    immutable = {
        key: value
        for key, value in manifest.items()
        if key not in {"integrity", "status", "updated_at"}
    }
    manifest["integrity"] = _digest(immutable)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="external-prediction runs cannot be resumed"):
        engine.resume(directory, backend_factory=Backend)


def test_subscription_ignored_token_cap_does_not_mark_completion_truncated(paired, tmp_path):
    class CompletedSubscription(Backend):
        def transcribe(self, *args):
            raw = super().transcribe(*args)
            raw["eval_count"] = 100
            return raw

    directory = engine.run(
        paired,
        ["chatgpt:first"],
        tmp_path / "runs",
        num_predict=1,
        warmup=False,
        backend_factory=CompletedSubscription,
    )
    assert all(not row["truncated"] for row in read_records(directory / "results.jsonl"))


def test_managed_retries_are_bounded_and_permanent_errors_are_not_replayed(paired, tmp_path):
    from vlm_bench.execution import RetryableProviderError

    attempts = {}

    class TransientThenSuccess(Backend):
        def transcribe(self, model, path, prompt, options):
            attempts[path.stem] = attempts.get(path.stem, 0) + 1
            if path.stem == "s0" and attempts[path.stem] < 3:
                raise RetryableProviderError(
                    "connection unavailable",
                    retry_after_seconds=0,
                    category="connection_error",
                )
            if path.stem == "s1":
                raise RuntimeError("permanent provider error")
            return super().transcribe(model, path, prompt, options)

    directory = engine.run(
        paired,
        ["ollama:first"],
        tmp_path / "runs",
        warmup=False,
        max_retries=2,
        backend_factory=TransientThenSuccess,
    )

    events = read_records(directory / "attempts.jsonl")
    starts = [event for event in events if event["event_kind"] == "request_started"]
    finishes = [event for event in events if event["event_kind"] == "request_finished"]
    assert attempts == {"s0": 3, "s1": 1, "s2": 1}
    assert len(starts) == len(finishes) == 5
    results = {row["sample_id"]: row["status"] for row in read_records(directory / "results.jsonl")}
    assert results == {"s0": "success", "s1": "error", "s2": "success"}


def test_managed_retry_exhaustion_records_each_physical_attempt(paired, tmp_path):
    from vlm_bench.execution import RetryableProviderError

    attempts = []

    class AlwaysTransient(Backend):
        def transcribe(self, model, path, prompt, options):
            attempts.append(path.stem)
            raise RetryableProviderError(
                "connection unavailable",
                retry_after_seconds=0,
                category="connection_error",
            )

    directory = engine.run(
        paired,
        ["ollama:first"],
        tmp_path / "runs",
        limit=1,
        warmup=False,
        max_retries=2,
        backend_factory=AlwaysTransient,
    )
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert len(read_records(directory / "results.jsonl")) == 1
    assert read_records(directory / "results.jsonl")[0]["status"] == "error"
    assert (
        len(
            [
                event
                for event in read_records(directory / "attempts.jsonl")
                if event["event_kind"] == "request_started"
            ]
        )
        == 3
    )


def test_warmup_consumes_request_budget_and_resume_does_not_repeat_it(paired, tmp_path):
    directory = engine.run(
        paired,
        ["ollama:first"],
        tmp_path / "runs",
        max_requests=1,
        backend_factory=Backend,
    )
    assert json.loads((directory / "execution.json").read_text())["limit_reason"] == "max_requests"
    assert len(read_records(directory / "warmups.jsonl")) == 1
    assert read_records(directory / "warmups.jsonl")[0]["status"] == "success"
    assert read_records(directory / "results.jsonl") == []
    assert len(Backend.calls) == 1

    engine.resume(directory, max_requests=1, backend_factory=Backend)
    assert len(Backend.calls) == 2
    assert len(read_records(directory / "results.jsonl")) == 1


def test_concurrent_request_limit_bounds_submitted_cloud_calls(paired, tmp_path):
    import threading
    import time

    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    class TrackingCloudBackend(Backend):
        def transcribe(self, model, path, prompt, options):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                time.sleep(0.02)
                return super().transcribe(model, path, prompt, options)
            finally:
                with lock:
                    state["active"] -= 1

    directory = engine.run(
        paired,
        ["openai:first"],
        tmp_path / "runs",
        warmup=False,
        concurrency=2,
        max_requests=2,
        backend_factory=TrackingCloudBackend,
    )
    execution = json.loads((directory / "execution.json").read_text())
    assert execution["limit_reason"] == "max_requests"
    assert execution["generation_requests_current"] == 2
    assert len(read_records(directory / "results.jsonl")) == 2
    assert state["peak"] == 2


def test_request_budget_pauses_and_resumes_with_per_invocation_allowance(paired, tmp_path):
    directory = engine.run(
        paired,
        ["ollama:first"],
        tmp_path / "runs",
        warmup=False,
        max_requests=1,
        backend_factory=Backend,
    )
    execution = json.loads((directory / "execution.json").read_text())
    assert execution["limit_reason"] == "max_requests"
    assert execution["generation_requests_current"] == 1
    assert execution["generation_requests_cumulative"] == 1
    assert len(read_records(directory / "results.jsonl")) == 1

    engine.resume(directory, max_requests=1, backend_factory=Backend)
    assert len(read_records(directory / "results.jsonl")) == 2
    engine.resume(directory, max_requests=5, backend_factory=Backend)
    assert json.loads((directory / "manifest.json").read_text())["status"] == "complete"
    execution = json.loads((directory / "execution.json").read_text())
    assert execution["generation_requests_current"] == 1
    assert execution["generation_requests_cumulative"] == 3
    assert len(execution["invocations"]) == 3


def test_spend_limit_requires_prices_and_stops_at_fixed_request_budget(paired, tmp_path):
    costs = {"api": {"model": "openai:first", "cost_per_sample_usd": 0.01}}
    with pytest.raises(ValueError, match="requires an applicable fixed request price"):
        engine.preview(
            paired,
            ["openai:first"],
            max_spend_usd=0.1,
            backend_factory=Backend,
        )

    directory = engine.run(
        paired,
        ["openai:first"],
        tmp_path / "runs",
        warmup=False,
        costs=costs,
        max_spend_usd=0.01,
        backend_factory=Backend,
    )
    execution = json.loads((directory / "execution.json").read_text())
    assert execution["limit_reason"] == "max_spend_usd"
    assert execution["generation_requests_current"] == 1
    assert execution["estimated_spend_usd_current"] == pytest.approx(0.01)
    assert len(read_records(directory / "results.jsonl")) == 1


def test_spend_limit_is_aggregated_across_models(paired, tmp_path):
    costs = {"api": {"cost_per_sample_usd": 0.01}}
    directory = engine.run(
        paired,
        ["openai:first", "chatgpt:second"],
        tmp_path / "runs",
        warmup=False,
        costs=costs,
        max_spend_usd=0.01,
        backend_factory=Backend,
    )
    rows = read_records(directory / "results.jsonl")
    assert len(rows) == 1 and rows[0]["model"] == "openai:first"
    assert json.loads((directory / "execution.json").read_text())["limit_reason"] == (
        "max_spend_usd"
    )


def test_token_spend_limit_stops_when_provider_omits_token_usage(paired, tmp_path):
    costs = {
        "api": {
            "model": "openai:first",
            "input_per_million_usd": 1.0,
            "output_per_million_usd": 2.0,
        }
    }
    directory = engine.run(
        paired,
        ["openai:first"],
        tmp_path / "runs",
        warmup=False,
        costs=costs,
        max_spend_usd=1.0,
        backend_factory=Backend,
    )
    execution = json.loads((directory / "execution.json").read_text())
    assert execution["limit_reason"] == "usage_unavailable"
    assert execution["unknown_spend_count_current"] == 1
    assert execution["generation_requests_current"] == 1
    assert len(read_records(directory / "results.jsonl")) == 1


def test_cache_hits_omit_provider_usage_and_repeated_measurements_bypass(paired, tmp_path):
    class DigestBackend(Backend):
        calls = []

        def validate_model(self, model):
            return {"provider": self.provider, "model": model, "digest": "a" * 64}

        def transcribe(self, model, path, prompt, options):
            self.calls.append((path.stem, prompt))
            return {"message": {"content": "hello"}, "model": model, "usage": {"total_tokens": 10}}

    DigestBackend.calls = []
    cache_dir = tmp_path / "cache"
    engine.run(
        paired,
        ["ollama:first"],
        tmp_path / "first",
        warmup=False,
        cache_dir=cache_dir,
        backend_factory=DigestBackend,
    )
    first_calls = len(DigestBackend.calls)
    second = engine.run(
        paired,
        ["ollama:first"],
        tmp_path / "second",
        warmup=False,
        cache_dir=cache_dir,
        backend_factory=DigestBackend,
    )
    assert len(DigestBackend.calls) == first_calls
    hits = read_records(second / "results.jsonl")
    assert all(row["cache_hit"] for row in hits)
    assert all(row["usage"] is None for row in hits)
    assert all(row["usage_source"] == "cache" for row in hits)
    assert all(row["inference_latency_seconds"] is None for row in hits)
    assert json.loads((second / "execution.json").read_text())["cache_hits_current"] == 3

    changed_prompt = engine.run(
        paired,
        ["ollama:first"],
        tmp_path / "changed-prompt",
        warmup=False,
        cache_dir=cache_dir,
        prose_prompt="A new transcription prompt.",
        backend_factory=DigestBackend,
    )
    assert len(DigestBackend.calls) == first_calls + 3
    assert json.loads((changed_prompt / "execution.json").read_text())["cache_hits_current"] == 0

    repeated = engine.run(
        paired,
        ["ollama:first"],
        tmp_path / "repeated",
        warmup=False,
        cache_dir=cache_dir,
        suite_condition="replicate",
        repetition=1,
        repeated_measurement=True,
        backend_factory=DigestBackend,
    )
    assert len(DigestBackend.calls) == first_calls + 6
    repeated_rows = read_records(repeated / "results.jsonl")
    assert all(not row["cache_hit"] for row in repeated_rows)
    assert all(row["cache_bypass_reason"] == "repeated_measurement" for row in repeated_rows)


def test_cache_write_failure_preserves_started_cloud_responses(paired, tmp_path, monkeypatch):
    from vlm_bench.cache import ResponseCache

    class FixedCloudRevision(Backend):
        def validate_model(self, model):
            return {
                "provider": self.provider,
                "model": model,
                "revision": "abcdef123456",
            }

        def transcribe(self, model, path, prompt, options):
            return {"message": {"content": "hello"}, "model": model}

    def fail_store(self, key, raw_response):
        raise OSError("cache disk unavailable")

    monkeypatch.setattr(ResponseCache, "store", fail_store)
    with pytest.raises(RuntimeError, match="writing its cache entry failed"):
        engine.run(
            paired,
            ["openai:first"],
            tmp_path / "runs",
            warmup=False,
            cache_dir=tmp_path / "cache",
            concurrency=2,
            backend_factory=FixedCloudRevision,
        )

    directory = next((tmp_path / "runs").iterdir())
    rows = read_records(directory / "results.jsonl")
    assert len(rows) == 2
    assert all(row["status"] == "success" for row in rows)
    assert all("cache disk unavailable" in row["cache_error"] for row in rows)
    assert json.loads((directory / "manifest.json").read_text())["status"] == "interrupted"


def test_parallel_interrupt_drains_completed_responses_on_writer_thread(
    paired, tmp_path, monkeypatch
):
    import threading
    import time

    from vlm_bench import task_metrics

    main_thread = threading.get_ident()
    score_calls = {"count": 0}
    real_score_task = task_metrics.score_task

    def interrupt_once(*args, **kwargs):
        assert threading.get_ident() == main_thread
        score_calls["count"] += 1
        if score_calls["count"] == 1:
            raise KeyboardInterrupt
        return real_score_task(*args, **kwargs)

    class SlowCloudBackend(Backend):
        def transcribe(self, model, path, prompt, options):
            time.sleep(0.01 if path.stem == "s0" else 0.04)
            return super().transcribe(model, path, prompt, options)

    monkeypatch.setattr(task_metrics, "score_task", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        engine.run(
            paired,
            ["openai:first"],
            tmp_path / "runs",
            warmup=False,
            concurrency=2,
            backend_factory=SlowCloudBackend,
        )

    directory = next((tmp_path / "runs").iterdir())
    rows = read_records(directory / "results.jsonl")
    assert len(rows) == 2
    assert all(row["status"] == "success" for row in rows)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"


def test_parallel_latency_excludes_main_thread_scoring_delay(paired, tmp_path, monkeypatch):
    import threading
    import time

    from vlm_bench import task_metrics

    main_thread = threading.get_ident()
    first_two_started = threading.Barrier(2)
    real_score_task = task_metrics.score_task
    score_calls = {"count": 0}

    def slow_first_score(*args, **kwargs):
        assert threading.get_ident() == main_thread
        score_calls["count"] += 1
        if score_calls["count"] == 1:
            time.sleep(0.25)
        return real_score_task(*args, **kwargs)

    class SynchronizedCloudBackend(Backend):
        def transcribe(self, model, path, prompt, options):
            if path.stem != "s2":
                first_two_started.wait(timeout=2)
            time.sleep(0.01)
            return super().transcribe(model, path, prompt, options)

    monkeypatch.setattr(task_metrics, "score_task", slow_first_score)
    directory = engine.run(
        paired,
        ["openai:first"],
        tmp_path / "runs",
        warmup=False,
        concurrency=2,
        backend_factory=SynchronizedCloudBackend,
    )

    rows = read_records(directory / "results.jsonl")
    assert len(rows) == 3
    assert score_calls["count"] == 3
    assert max(row["inference_latency_seconds"] for row in rows) < 0.15
