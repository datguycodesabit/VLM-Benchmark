import hashlib
import json

import pytest
from PIL import Image
from test_engine import Backend
from test_runner import FakeClient

from vlm_bench.cli import main
from vlm_bench.runner import read_records, run_benchmark


def test_100_forms_two_models_and_exports(tmp_path):
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    (root / "xml").mkdir()
    for index in range(100):
        stem = f"a01-{index:03}"
        Image.new("RGB", (180, 220), "white").save(root / "images" / f"{stem}.png")
        (root / "xml" / f"{stem}.xml").write_text(
            '<form height="220"><handwritten-part><line text="Hello, world."><word><cmp x="20" y="100" width="120" height="20"/></word></line></handwritten-part></form>'
        )
    assert main(["prepare", "--data", str(root)]) == 0
    assert len(list((root / "references").glob("*.txt"))) == 100
    FakeClient.calls = []
    FakeClient.fail_after = None
    FakeClient.digest = "sha256:fixed"
    directory = run_benchmark(
        root, ["one", "two"], tmp_path / "runs", client_factory=FakeClient, progress=lambda _: None
    )
    assert len(read_records(directory / "results.jsonl")) == 200
    assert len(read_records(directory / "warmups.jsonl")) == 2
    assert main(["export", "--run", str(directory)]) == 0
    assert (directory / "results.xlsx").exists()
    assert (directory / "summary.csv").exists()
    assert (directory / "samples.csv").exists()
    assert main(["rescore", "--run", str(directory)]) == 0
    assert len(FakeClient.calls) == 202


def test_cli_empty_folder_clear_error(tmp_path, capsys):
    assert main(["prepare", "--data", str(tmp_path)]) == 1
    assert "Error:" in capsys.readouterr().err


def test_formula_rendering_uses_config_default_and_cli_override(tmp_path, monkeypatch, capsys):
    from vlm_bench import engine

    config = tmp_path / "experiment.toml"
    config.write_text(
        'version = 1\n[experiment]\nmodels = ["ollama:sample"]\nformula_rendering = true\n',
        encoding="utf-8",
    )
    captured = []
    monkeypatch.setattr(
        engine,
        "preview",
        lambda _data, _models, **options: captured.append(options) or {"ok": True},
    )

    assert main(["run", "--config", str(config), "--dry-run"]) == 0
    capsys.readouterr()
    assert captured[-1]["formula_rendering"] is True
    assert not any(captured[-1]["costs"].values())
    assert main(["run", "--config", str(config), "--dry-run", "--no-formula-rendering"]) == 0
    capsys.readouterr()
    assert captured[-1]["formula_rendering"] is False


def test_suite_cli_exposes_previewable_and_resumable_workflow(tmp_path, monkeypatch, capsys):
    from vlm_bench import suite

    captured = {}

    def fake_suite(config, output, **options):
        captured.update(config=config, output=output, **options)
        return {"schema_version": 1, "conditions": {}, "inference": False}

    monkeypatch.setattr(suite, "run_suite", fake_suite)
    config = tmp_path / "suite.toml"
    output = tmp_path / "suite-output"
    assert (
        main(["suite", "--config", str(config), "--output", str(output), "--dry-run", "--resume"])
        == 0
    )
    assert captured["dry_run"] is True
    assert captured["resume"] is True
    assert json.loads(capsys.readouterr().out)["inference"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["--json", "status", "--run"],
        ["status", "--run"],
    ],
)
def test_json_cli_errors_are_one_parseable_object(tmp_path, capsys, arguments):
    # Missing manifests exercise the runtime error path in either flag position.
    arguments.extend([str(tmp_path)])
    arguments.append("--json")
    assert main(arguments) == 1
    stdout = capsys.readouterr().out
    assert len(stdout.splitlines()) == 1
    payload = json.loads(stdout)
    assert payload["command"] == "status"
    assert payload["status"] == "error"
    assert payload["exit_code"] == 1
    assert payload["error"]


def test_json_dataset_check_returns_invalid_audit_without_repeating_it(
    tmp_path, monkeypatch, capsys
):
    from vlm_bench import dataset

    calls = []
    audit = {
        "layout": "paired",
        "valid": False,
        "counts": {"images": 1, "references": 0},
        "issues": [{"type": "missing_reference", "sample_id": "line-1"}],
        "findings": [{"type": "near_duplicate_image", "sample_ids": ["line-1"]}],
    }

    def check_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return audit

    monkeypatch.setattr(dataset, "check_dataset", check_dataset)
    assert main(["dataset", "check", "--data", str(tmp_path), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["result"] == audit
    assert payload["error"]
    assert len(calls) == 1


def test_json_auth_status_reports_safe_state_without_secret_fields(monkeypatch, capsys):
    from vlm_bench import auth

    monkeypatch.setattr(
        auth,
        "status",
        lambda: {
            "provider": "chatgpt",
            "host_id_configured": True,
            "accounts": [
                {
                    "provider": "chatgpt",
                    "email": "reader@example.test",
                    "expires_at": "2026-10-05T12:00:00Z",
                    "connected": True,
                    "client_id": "private-client-id",
                    "subject": "private-subject",
                    "access_token": "private-access-token",
                    "refresh_token": "private-refresh-token",
                }
            ],
        },
    )
    assert main(["auth", "status", "--json"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["result"]["provider"] == "chatgpt"
    assert payload["result"]["authenticated"] is True
    assert payload["result"]["accounts"] == [
        {
            "provider": "chatgpt",
            "email": "reader@example.test",
            "expires_at": "2026-10-05T12:00:00Z",
            "connected": True,
        }
    ]
    assert "private-access-token" not in output
    assert "private-refresh-token" not in output
    assert "private-subject" not in output


def test_json_run_directory_deduplicates_engine_and_cli_messages(tmp_path, monkeypatch, capsys):
    from vlm_bench import cli, engine

    run_dir = tmp_path / "run"

    def fake_run(*args, **kwargs):
        run_dir.mkdir()
        (run_dir / "results.jsonl").write_text("", encoding="utf-8")
        (run_dir / "manifest.json").write_text('{"status":"complete"}', encoding="utf-8")
        print(f"Run directory: {run_dir}")
        return run_dir

    monkeypatch.setattr(engine, "preview", lambda *args, **kwargs: {"models": []})
    monkeypatch.setattr(engine, "run", fake_run)
    monkeypatch.setattr(cli, "_report", lambda path: None)
    assert main(["run", "--data", str(tmp_path), "--models", "ollama:sample", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_directory"] == str(run_dir.resolve())
    assert not isinstance(payload["run_directory"], list)


def test_json_run_pause_status_and_resume_limit_overrides(tmp_path, monkeypatch, capsys):
    from vlm_bench import cli, engine
    from vlm_bench.runner import _digest

    run_dir = tmp_path / "paused-run"
    captured = {}

    def write_execution(controls, *, limit_reason, generation_requests):
        execution = {
            "schema_version": 1,
            "invocations": [
                {
                    "started_at": "2026-10-05T00:00:00Z",
                    "finished_at": "2026-10-05T00:01:00Z",
                    "status": "paused" if limit_reason else "complete",
                    "generation_requests": generation_requests,
                    "estimated_spend_usd": 0.2,
                    "unknown_spend_count": 0,
                    "cache_hits": 0,
                    "limit_reason": limit_reason,
                    "execution_controls": controls,
                }
            ],
            "generation_requests_current": generation_requests,
            "generation_requests_cumulative": generation_requests,
            "estimated_spend_usd_current": 0.2,
            "estimated_spend_usd_cumulative": 0.2,
            "unknown_spend_count_current": 0,
            "unknown_spend_count_cumulative": 0,
            "cache_hits_current": 0,
            "cache_hits_cumulative": 0,
            "limit_reason": limit_reason,
            "execution_controls": controls,
        }
        (run_dir / "execution.json").write_text(json.dumps(execution), encoding="utf-8")

    def fake_run(_data, models, _output, **kwargs):
        captured["run"] = kwargs
        run_dir.mkdir()
        (run_dir / "results.jsonl").write_text("", encoding="utf-8")
        controls = {
            "max_retries": kwargs["max_retries"],
            "concurrency": kwargs["concurrency"],
            "max_requests": kwargs["max_requests"],
            "max_spend_usd": kwargs["max_spend_usd"],
            "cache_dir": None,
        }
        manifest = {
            "schema_version": 2,
            "status": "paused",
            "models": models,
            "samples": [{"id": "sample-1", "reference": "reference"}],
            "execution_controls": controls,
        }
        manifest["integrity"] = _digest(
            {key: value for key, value in manifest.items() if key not in {"integrity", "status"}}
        )
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        write_execution(controls, limit_reason="max_requests", generation_requests=1)
        print(f"Run directory: {run_dir}")
        return run_dir

    def fake_resume(path, *, retry_failed, max_requests, max_spend_usd):
        captured["resume"] = {
            "path": path,
            "retry_failed": retry_failed,
            "max_requests": max_requests,
            "max_spend_usd": max_spend_usd,
        }
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "complete"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        controls = {
            **manifest["execution_controls"],
            "max_requests": max_requests,
            "max_spend_usd": max_spend_usd,
        }
        write_execution(controls, limit_reason=None, generation_requests=0)
        return path

    monkeypatch.setattr(engine, "preview", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(engine, "run", fake_run)
    monkeypatch.setattr(engine, "resume", fake_resume)
    monkeypatch.setattr(cli, "_report", lambda _path: None)

    assert (
        main(
            [
                "run",
                "--data",
                str(tmp_path),
                "--models",
                "openai:example",
                "--max-requests",
                "1",
                "--max-spend-usd",
                "0.75",
                "--json",
            ]
        )
        == 2
    )
    run_capture = capsys.readouterr()
    assert len(run_capture.out.splitlines()) == 1
    assert "Run directory:" in run_capture.err
    run_payload = json.loads(run_capture.out)
    assert run_payload["status"] == "incomplete"
    assert run_payload["run_directory"] == str(run_dir.resolve())
    assert captured["run"]["max_requests"] == 1
    assert captured["run"]["max_spend_usd"] == 0.75

    assert main(["status", "--run", str(run_dir), "--json"]) == 0
    status_payload = json.loads(capsys.readouterr().out)["result"]
    assert status_payload["limit_reason"] == "max_requests"
    assert status_payload["request_limit"] == 1
    assert status_payload["generation_attempt_count_current"] == 1

    assert (
        main(
            [
                "resume",
                "--run",
                str(run_dir),
                "--max-requests",
                "4",
                "--max-spend-usd",
                "1.5",
                "--json",
            ]
        )
        == 0
    )
    resume_capture = capsys.readouterr()
    assert len(resume_capture.out.splitlines()) == 1
    resume_payload = json.loads(resume_capture.out)
    assert resume_payload["run_directory"] == str(run_dir.resolve())
    assert captured["resume"]["max_requests"] == 4
    assert captured["resume"]["max_spend_usd"] == 1.5
    assert main(["status", "--run", str(run_dir), "--json"]) == 0
    final_status = json.loads(capsys.readouterr().out)["result"]
    assert final_status["request_limit"] == 4
    assert final_status["original_request_limit"] == 1


def test_json_inspection_error_does_not_claim_missing_report(tmp_path, capsys):
    output = tmp_path / "not-created.html"
    assert (
        main(["inspect", "--run", str(tmp_path / "missing-run"), "--output", str(output), "--json"])
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "report_paths" not in payload


def test_json_export_lists_only_files_that_were_created(tmp_path, monkeypatch, capsys):
    import vlm_bench.cli as cli

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_export(_run, _formats):
        report = run_dir / "summary.csv"
        report.write_text("model,cer\n", encoding="utf-8")
        return [report]

    monkeypatch.setattr(cli, "export_run", fake_export)
    assert main(["export", "--run", str(run_dir), "--formats", "csv", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["report_paths"] == [str((run_dir / "summary.csv").resolve())]


def _write_rescore_run(run_dir, *, records=None, tamper_reference=False):
    from vlm_bench.runner import _digest
    from vlm_bench.snapshot import fingerprint

    run_dir.mkdir()
    sample = {
        "id": "s1",
        "reference": "authoritative reference",
        "hashes": {"crop": hashlib.sha256(b"crop").hexdigest()},
        "preprocess": "original",
        "content_type": "prose",
        "metadata": {
            "content_type": "prose",
            "critical_expressions": ["authoritative"],
        },
    }
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "preprocess": "original",
        "models": ["ollama:sample"],
        "samples": [sample],
        "formula_rendering": False,
        "task_metrics_version": 1,
    }
    manifest["benchmark_fingerprint"] = fingerprint(manifest["samples"], "original")
    manifest["integrity"] = _digest(
        {key: value for key, value in manifest.items() if key not in {"status", "updated_at"}}
    )
    if tamper_reference:
        manifest["samples"][0]["reference"] = "tampered reference"
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if records is None:
        records = [
            {
                "model": "ollama:sample",
                "sample_id": "s1",
                "status": "success",
                "prediction": "authoritative reference",
                "reference": "forged row reference",
                "metrics": {"cer": 0.9},
            }
        ]
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_rescore_uses_frozen_annotations_and_refuses_tampered_manifest(
    tmp_path, monkeypatch, capsys
):
    from vlm_bench import cli, task_metrics

    run_dir = tmp_path / "run"
    _write_rescore_run(run_dir)
    captured = {}

    def fake_score_task(prediction, reference, sample, *, formula_rendering=False):
        captured.update(
            prediction=prediction,
            reference=reference,
            sample=sample,
            formula_rendering=formula_rendering,
        )
        return {"version": 1, "scores": {"critical_expression_accuracy": 1.0}, "errors": []}

    monkeypatch.setattr(task_metrics, "score_task", fake_score_task)
    monkeypatch.setattr(cli, "_report", lambda _run: None)
    assert main(["rescore", "--run", str(run_dir)]) == 0
    rescored = read_records(run_dir / "results.jsonl")[0]
    assert captured["reference"] == "authoritative reference"
    assert captured["sample"]["metadata"]["critical_expressions"] == ["authoritative"]
    assert captured["formula_rendering"] is False
    assert rescored["reference"] == "authoritative reference"
    assert rescored["metrics"]["task_metrics"]["scores"]["critical_expression_accuracy"] == 1.0

    before = (run_dir / "results.jsonl").read_bytes()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["samples"][0]["reference"] = "tampered after signing"
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert main(["rescore", "--run", str(run_dir)]) == 1
    assert "integrity check failed" in capsys.readouterr().err
    assert (run_dir / "results.jsonl").read_bytes() == before


def test_rescore_rejects_duplicate_and_unknown_result_pairs(tmp_path, monkeypatch, capsys):
    from vlm_bench import cli

    monkeypatch.setattr(cli, "_report", lambda _run: None)
    good = {
        "model": "ollama:sample",
        "sample_id": "s1",
        "status": "success",
        "prediction": "x",
        "reference": "y",
    }
    cases = [
        ("duplicate", [good, dict(good)], "Duplicate model/sample"),
        ("unknown", [dict(good, sample_id="missing")], "unknown model/sample pair"),
    ]
    for name, records, message in cases:
        run_dir = tmp_path / name
        _write_rescore_run(run_dir, records=records)
        before = (run_dir / "results.jsonl").read_bytes()
        assert main(["rescore", "--run", str(run_dir)]) == 1
        assert message in capsys.readouterr().err
        assert (run_dir / "results.jsonl").read_bytes() == before


def test_paired_check_and_configured_dry_run(tmp_path, monkeypatch, capsys):
    import json

    from vlm_bench import engine

    data = tmp_path / "paired"
    (data / "images").mkdir(parents=True)
    (data / "text").mkdir()
    Image.new("RGB", (90, 25), "white").save(data / "images" / "line.png")
    (data / "text" / "line.txt").write_text("hello", encoding="utf-8")
    assert main(["dataset", "check", "--data", str(data), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["valid"]
    config = tmp_path / "experiment.toml"
    config.write_text(
        'version = 1\n[experiment]\nmodels = ["ollama:first", "openai:second"]\n'
        'data = "nonexistent"\nlayout = "paired"\nseed = 21\n',
        encoding="utf-8",
    )
    original_preview = engine.preview
    monkeypatch.setattr(
        engine,
        "preview",
        lambda *args, **kwargs: original_preview(*args, **kwargs, backend_factory=Backend),
    )
    Backend.calls = []
    assert main(["run", "--config", str(config), "--data", str(data), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["sample_ids"] == ["line"]
    assert [item["execution"] for item in report["models"]] == ["local", "cloud"]
    assert Backend.calls == []
    assert not (tmp_path / "runs").exists()


def test_prepared_cli_workflow_and_selection_conflicts(tmp_path, monkeypatch, capsys):
    import json

    from vlm_bench import engine

    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "text").mkdir()
    Image.new("RGB", (90, 25), "white").save(data / "images" / "line.png")
    (data / "text" / "line.txt").write_text("hello", encoding="utf-8")
    prepared = tmp_path / "benchmark"
    assert main(["prepare", "--data", str(data), "--output", str(prepared)]) == 0
    capsys.readouterr()
    assert main(["dataset", "check", "--prepared", str(prepared)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    assert main(["prepare", "--data", str(data), "--output", str(prepared)]) == 1
    capsys.readouterr()
    original_preview = engine.preview
    monkeypatch.setattr(
        engine, "preview", lambda *a, **k: original_preview(*a, **k, backend_factory=Backend)
    )
    assert main(["run", "--prepared", str(prepared), "--models", "ollama:first", "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["benchmark_fingerprint"]
    for flag, value in [
        ("--seed", "42"),
        ("--limit", "1"),
        ("--preprocess", "original"),
        ("--data", str(data)),
    ]:
        assert (
            main(
                [
                    "run",
                    "--prepared",
                    str(prepared),
                    "--models",
                    "ollama:first",
                    "--dry-run",
                    flag,
                    value,
                ]
            )
            == 1
        )
        assert "--prepared cannot" in capsys.readouterr().err


def test_separate_runs_compare_and_cli_token_limit_overrides_config(tmp_path, monkeypatch, capsys):
    import json

    from vlm_bench import engine
    from vlm_bench.snapshot import freeze

    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "text").mkdir()
    Image.new("RGB", (90, 25), "white").save(data / "images" / "line.png")
    (data / "text" / "line.txt").write_text("hello", encoding="utf-8")
    snapshot = tmp_path / "benchmark"
    freeze(data, snapshot)
    config = tmp_path / "experiment.toml"
    config.write_text(
        'version = 1\n[experiment]\nprepared = "benchmark"\nmodels = ["ollama:first"]\n[models."ollama:first"]\nnum_predict = 77\n',
        encoding="utf-8",
    )
    original_run = engine.run
    monkeypatch.setattr(
        engine, "run", lambda *a, **k: original_run(*a, **k, backend_factory=Backend)
    )
    Backend.calls = []
    runs = tmp_path / "runs"
    assert (
        main(
            [
                "run",
                "--config",
                str(config),
                "--output",
                str(runs),
                "--num-predict",
                "22",
                "--no-warmup",
            ]
        )
        == 0
    )
    first = next(runs.iterdir())
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["provider_controls"]["ollama:first"]["effective"]["num_predict"] == 22
    assert (
        main(
            [
                "run",
                "--prepared",
                str(snapshot),
                "--models",
                "trocr:second",
                "--output",
                str(runs),
                "--no-warmup",
            ]
        )
        == 0
    )
    second = next(d for d in runs.iterdir() if d != first)
    output = tmp_path / "comparison"
    assert main(["compare", "--runs", str(first), str(second), "--output", str(output)]) == 0
    report = json.loads((output / "comparison.json").read_text())
    assert len(report["tracks"]["prose"]["model_results"]) == 2
    assert all(r["cer"] == 0 for r in report["tracks"]["prose"]["model_results"])
    assert (output / "comparison.csv").exists()
    assert (output / "paired.csv").exists()
    assert main(["compare", "--runs", str(first), str(second), "--output", str(output)]) == 1
    capsys.readouterr()


def test_run_research_protocol_defaults_config_and_cli_override(tmp_path, monkeypatch, capsys):
    import json

    from vlm_bench import engine

    data = tmp_path / "data"
    config = tmp_path / "experiment.toml"
    config.write_text(
        'version = 1\n[experiment]\nmodels = ["ollama:first"]\n'
        f'data = "{data}"\nlayout = "paired"\n'
        'strict_research = true\nprotocol = "writer-disjoint"\n',
        encoding="utf-8",
    )
    captured = []

    def preview(_data, _models, **options):
        captured.append(options)
        return {"strict_research": options["strict_research"], "protocol": options["protocol"]}

    monkeypatch.setattr(engine, "preview", preview)
    assert main(["run", "--config", str(config), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "strict_research": True,
        "protocol": "writer-disjoint",
    }
    assert captured[-1]["strict_research"] is True
    assert captured[-1]["protocol"] == "writer-disjoint"

    assert (
        main(
            [
                "run",
                "--config",
                str(config),
                "--dry-run",
                "--no-strict-research",
                "--protocol",
                "document-disjoint",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert captured[-1]["strict_research"] is False
    assert captured[-1]["protocol"] == "document-disjoint"


def test_dataset_split_cli_passes_selected_protocol(tmp_path, monkeypatch, capsys):
    from vlm_bench import dataset

    captured = {}
    monkeypatch.setattr(
        dataset,
        "check_dataset",
        lambda *_args, **_kwargs: {"valid": True, "samples": [{"id": "one"}]},
    )

    def split(samples, **options):
        captured.update(options)
        return samples

    monkeypatch.setattr(dataset, "split_dataset", split)
    output = tmp_path / "metadata.jsonl"
    assert (
        main(
            [
                "dataset",
                "split",
                "--data",
                str(tmp_path),
                "--output",
                str(output),
                "--protocol",
                "writer-disjoint",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert captured == {"seed": 42, "protocol": "writer-disjoint"}


def test_dataset_check_prints_bounded_review_findings(tmp_path, monkeypatch, capsys):
    from vlm_bench import dataset

    monkeypatch.setattr(
        dataset,
        "check_dataset",
        lambda *_args, **_kwargs: {
            "valid": True,
            "layout": "paired",
            "counts": {"samples": 2},
            "issues": [],
            "excluded": [],
            "findings": [{"type": "near_duplicate_images", "ids": ["a", "b"]}],
        },
    )
    assert main(["dataset", "check", "--data", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert 'Finding: {"type": "near_duplicate_images", "ids": ["a", "b"]}' in output
