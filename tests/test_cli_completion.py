"""Public CLI coverage for the completed experiment workflow."""

import json
import shutil

from PIL import Image
from test_engine import Backend

from vlm_bench import cli, engine
from vlm_bench.experiments import initialize_experiment_plan
from vlm_bench.reporting import write_report
from vlm_bench.runner import read_records
from vlm_bench.scoring_policy import ScoringPolicy
from vlm_bench.snapshot import freeze
from vlm_bench.status import run_status
from vlm_bench.weighted_comparison import compare_weighted


def data(tmp_path):
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    (root / "text").mkdir()
    for index in range(3):
        Image.new("RGB", (20 + index, 10), "white").save(root / "images" / f"s{index}.png")
        (root / "text" / f"s{index}.txt").write_text("hello")
    (root / "metadata.jsonl").write_text(
        "".join(json.dumps({"id": f"s{i}", "source_document": f"doc-{i}"}) + "\n" for i in range(3))
    )
    prepared = tmp_path / "prepared"
    freeze(root, prepared)
    return prepared


def native(prepared, output, model="ollama:fake", *, latency=0.2):
    Backend.calls, Backend.failure, Backend.identity = [], None, "fixed"
    run = engine.run(
        None,
        [model],
        output,
        prepared=prepared,
        warmup=False,
        backend_factory=Backend,
        scoring_policy=ScoringPolicy().to_dict(),
        measurement_profile={"profile": "latency"},
        progress=lambda _: None,
    )
    rows = read_records(run / "results.jsonl")
    for row in rows:
        row["latency_seconds"] = latency
    (run / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return run


def test_version_json_and_structured_doctor_failure(tmp_path, monkeypatch, capsys):
    assert cli.main(["--json", "--version"]) == 0
    assert "vlm-bench" in json.loads(capsys.readouterr().out)["messages"][0]
    from vlm_bench import backends

    monkeypatch.setattr(
        backends,
        "create_backend",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
    )
    assert cli.main(["doctor", "--provider", "openai", "--json"]) == 1
    result = json.loads(capsys.readouterr().out)["result"]
    assert result["ready"] is False
    assert result["checks"][-1]["detail"] == "provider unavailable"


def test_empty_experiment_plan_status_resume_and_portability(tmp_path, monkeypatch, capsys):
    prepared = data(tmp_path)
    experiment = tmp_path / "experiment"
    initialize_experiment_plan(
        experiment,
        planned_passes=3,
        prepared=prepared,
        models=["ollama:fake"],
        run_kwargs={
            "scoring_policy": ScoringPolicy().to_dict(),
            "measurement_profile": {"profile": "latency"},
            "warmup": False,
        },
    )
    state = run_status(experiment)
    assert state["expected_result_count"] == state["missing_result_count"] == 9
    assert state["completed_passes"] == 0
    original = engine.run
    Backend.calls, Backend.failure, Backend.identity = [], None, "fixed"
    monkeypatch.setattr(engine, "run", lambda *a, **kw: original(*a, **kw, backend_factory=Backend))
    assert cli.main(["resume", "--run", str(experiment), "--json"]) == 0
    capsys.readouterr()
    assert len(Backend.calls) == 9
    moved = tmp_path / "moved"
    shutil.move(experiment, moved)
    assert run_status(moved)["completed_passes"] == 3
    write_report(moved)
    assert (moved / "report" / "report.html").is_file()
    assert cli.main(["status", "--run", str(moved), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["result_coverage"] == 1


def test_weighted_comparison_uses_latency_and_joint_document_intervals(tmp_path, capsys):
    prepared = data(tmp_path)
    slower = native(prepared, tmp_path / "slow", "ollama:a", latency=0.5)
    faster = native(prepared, tmp_path / "fast", "ollama:z", latency=0.1)
    output = tmp_path / "comparison"
    assert (
        cli.main(["compare", "--runs", str(slower), str(faster), "--output", str(output), "--json"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    track = payload["result"]["conditions"]["baseline"]["tracks"]["prose"]
    assert track["ranking"]["winner"].startswith("ollama:z@")
    assert track["uncertainty"]["unit"] == "source_document"
    assert track["uncertainty"]["paired_score_differences"][0]["ci95"] == [0, 0]
    assert any(value.endswith("report.xlsx") for value in payload["report_paths"])
    assert (output / "paired.csv").is_file()
    assert (output / "report.html").is_file()
    assert cli.main(["report", "--run", str(slower), "--formats", "html", "--json"]) == 0
    assert (
        json.loads(capsys.readouterr().out)["result"]["conditions"]["baseline"]["tracks"]["prose"][
            "models"
        ][0]["score"]
        == 100
    )


def test_new_import_freezes_policy_and_never_ranks_unverified_external_speed(tmp_path, capsys):
    prepared = data(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps({"sample_id": f"s{i}", "prediction": "hello", "latency_seconds": 0.01})
            + "\n"
            for i in range(3)
        )
    )
    output = tmp_path / "import"
    assert (
        cli.main(
            [
                "import",
                "--prepared",
                str(prepared),
                "--predictions",
                str(predictions),
                "--output",
                str(output),
                "--system",
                "fixture",
                "--provenance",
                "test",
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["scoring_policy"]["character_accuracy_weight"] == 0.7
    report = json.loads((output / "report" / "evaluation.json").read_text())
    track = report["conditions"]["baseline"]["tracks"]["prose"]
    assert track["models"][0]["score"] == 100
    assert track["ranking"]["winner"] is None


def test_comparison_rejects_policy_mismatch_until_explicit_rescoring(tmp_path):
    prepared = data(tmp_path)
    first = native(prepared, tmp_path / "first")
    second = engine.run(
        None,
        ["ollama:fake"],
        tmp_path / "second",
        prepared=prepared,
        warmup=False,
        backend_factory=Backend,
        scoring_policy=ScoringPolicy(
            character_accuracy_weight=0.6, exact_match_weight=0.4
        ).to_dict(),
        measurement_profile={"profile": "latency"},
    )
    import pytest

    with pytest.raises(ValueError, match="policies differ"):
        compare_weighted([first, second], tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    result = compare_weighted(
        [first, second], tmp_path / "rescored", policy=ScoringPolicy().to_dict()
    )
    assert result["conditions"]["baseline"]["policy"]["character_accuracy_weight"] == 0.7


def test_derived_comparison_renders_offline_after_sources_move(tmp_path, capsys):
    prepared = data(tmp_path)
    left = native(prepared, tmp_path / "left", "ollama:a")
    right = native(prepared, tmp_path / "right", "ollama:b")
    comparison = tmp_path / "comparison"
    compare_weighted([left, right], comparison)
    shutil.move(tmp_path / "left", tmp_path / "moved-left")
    shutil.move(tmp_path / "right", tmp_path / "moved-right")
    assert (
        cli.main(
            ["report", "--run", str(comparison), "--output", str(tmp_path / "rendered"), "--json"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["comparison_schema_version"] == 2
    assert list((tmp_path / "rendered" / "assets").iterdir())
    assert (tmp_path / "rendered" / "operations.csv").is_file()
    assert (
        cli.main(
            [
                "report",
                "--run",
                str(comparison),
                "--output",
                str(tmp_path / "rejected"),
                "--weighted",
            ]
        )
        == 1
    )
    assert "original run artifacts" in capsys.readouterr().err


def test_csv_only_reports_include_diagnostics_without_json(tmp_path):
    prepared = data(tmp_path)
    run = native(prepared, tmp_path / "runs")
    paths = write_report(run, tmp_path / "csv", formats=["csv"])
    assert all(path.suffix == ".csv" for path in paths)
    assert (tmp_path / "csv" / "passes.csv").is_file()
    assert (tmp_path / "csv" / "usage.csv").is_file()
    assert not (tmp_path / "csv" / "evaluation.json").exists()


def test_resume_accepts_suite_directory_without_remeasurement(tmp_path, monkeypatch, capsys):
    from vlm_bench.init import create_config

    prepared = data(tmp_path)
    cfg = create_config(tmp_path / "suite.toml", prepared=prepared, models=["ollama:fake"])
    original_run, original_preview = engine.run, engine.preview
    Backend.calls, Backend.failure, Backend.identity = [], None, "fixed"
    monkeypatch.setattr(
        engine, "run", lambda *a, **kw: original_run(*a, **kw, backend_factory=Backend)
    )
    monkeypatch.setattr(
        engine, "preview", lambda *a, **kw: original_preview(*a, **kw, backend_factory=Backend)
    )
    root = tmp_path / "suite"
    assert cli.main(["suite", "--config", str(cfg), "--output", str(root)]) == 0
    capsys.readouterr()
    calls = len(Backend.calls)
    assert cli.main(["resume", "--run", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    assert len(Backend.calls) == calls
    assert (
        cli.main(
            [
                "run",
                "--config",
                str(cfg),
                "--prepared",
                str(prepared),
                "--output",
                str(tmp_path / "ignored"),
            ]
        )
        == 1
    )
    assert "suite configuration" in capsys.readouterr().err
