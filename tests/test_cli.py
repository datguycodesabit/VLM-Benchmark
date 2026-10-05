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


def test_paired_check_and_configured_dry_run(tmp_path, monkeypatch, capsys):
    import json

    from vlm_bench import engine

    data = tmp_path / "paired"
    (data / "images").mkdir(parents=True)
    (data / "text").mkdir()
    Image.new("RGB", (90, 25), "white").save(data / "images" / "line.png")
    (data / "text" / "line.txt").write_text("hello", encoding="utf-8")
    assert main(["dataset", "check", "--data", str(data), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
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
