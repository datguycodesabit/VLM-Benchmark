import json
import shutil

import pytest
from PIL import Image

from vlm_bench.cli import main
from vlm_bench.snapshot import freeze, load


def _source(tmp_path, layout):
    data = tmp_path / "source"
    (data / "images").mkdir(parents=True)
    (data / "text").mkdir()
    for name, size in (("tiny", (10, 10)), ("boundary", (100, 100)), ("large", (200, 100))):
        Image.new("RGB", size, "white").save(data / "images" / f"{name}.png")
        (data / "text" / f"{name}.txt").write_text(name)
    if layout == "iam-words":
        (data / "words.txt").write_text(
            "\n".join(
                f"{name} ok 154 1 0 0 {width} {height} NN {name}"
                for name, width, height in (
                    ("tiny", 10, 10),
                    ("boundary", 100, 100),
                    ("large", 200, 100),
                )
            )
        )
    elif layout == "iam-lines":
        (data / "lines.txt").write_text(
            "\n".join(
                f"{name} ok 154 1 0 0 {width} {height} {name}"
                for name, width, height in (
                    ("tiny", 10, 10),
                    ("boundary", 100, 100),
                    ("large", 200, 100),
                )
            )
        )
    return data


@pytest.mark.parametrize("layout", ["paired", "iam-words", "iam-lines"])
def test_native_area_filter_before_enlargement_and_sampling(tmp_path, layout):
    source = _source(tmp_path, layout)
    output = tmp_path / "snapshot"
    snapshot = freeze(
        source, output, layout=layout, min_pixels=10000, preprocess="enhanced", limit=2
    )
    assert set(snapshot["sample_ids"]) == {"boundary", "large"}
    audit = json.loads((output / "image-size-filter.json").read_text())
    assert audit["candidate_count"] == 3
    assert audit["eligible_count"] == 2
    assert audit["excluded"] == [
        {
            "sample_id": "tiny",
            "width": 10,
            "height": 10,
            "pixels": 100,
            "reason": "below_min_pixels",
        }
    ]
    moved = tmp_path / "moved"
    output.rename(moved)
    shutil.rmtree(source)
    assert load(moved)["benchmark_fingerprint"] == snapshot["benchmark_fingerprint"]
    (moved / "image-size-filter.json").write_text("{}")
    with pytest.raises(ValueError, match="exclusion audit"):
        load(moved)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_threshold_rejected_without_publishing(tmp_path, value):
    output = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="min_pixels"):
        freeze(_source(tmp_path, "paired"), output, min_pixels=value)
    assert not output.exists()


def test_empty_or_insufficient_filtered_population_does_not_publish(tmp_path):
    source = _source(tmp_path, "paired")
    output = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="all 3 excluded"):
        freeze(source, output, min_pixels=30000)
    assert not output.exists()
    with pytest.raises(ValueError, match="only 2"):
        freeze(source, output, min_pixels=10000, limit=3)
    assert not output.exists()


def test_prepare_cli_reports_exclusions(tmp_path, capsys):
    output = tmp_path / "snapshot"
    assert (
        main(
            [
                "prepare",
                "--data",
                str(_source(tmp_path, "paired")),
                "--output",
                str(output),
                "--min-pixels",
                "10000",
            ]
        )
        == 0
    )
    assert "1 excluded; 2 eligible" in capsys.readouterr().out


def test_json_prepare_exposes_filter_and_audit_path(tmp_path, capsys):
    output = tmp_path / "snapshot"
    assert (
        main(
            [
                "--json",
                "prepare",
                "--data",
                str(_source(tmp_path, "paired")),
                "--output",
                str(output),
                "--min-pixels",
                "10000",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["image_size_filter"]["excluded_count"] == 1
    assert result["exclusion_audit"] == str(output / "image-size-filter.json")


def test_form_filter_measures_actual_native_crop(tmp_path):
    from test_dataset import _sample

    source = tmp_path / "source"
    _sample(source, "form")
    # Handwriting bounds 30x8 become a clipped 60x48 crop with context.
    snapshot = freeze(source, tmp_path / "snapshot", min_pixels=2880)
    assert snapshot["samples"][0]["crop_bbox"] == [0, 0, 60, 48]
    with pytest.raises(ValueError, match="all 1 excluded"):
        freeze(source, tmp_path / "too-small", min_pixels=2881)


def test_word_example_freezes_effective_32_token_limit(tmp_path):
    from pathlib import Path

    from test_engine import Backend

    from vlm_bench import engine
    from vlm_bench.config import load_config

    config = load_config(Path(__file__).resolve().parents[1] / "examples/iam-ollama-filtered.toml")
    output = tmp_path / "snapshot"
    freeze(_source(tmp_path, "paired"), output, min_pixels=10000)
    preview = engine.preview(
        prepared=output,
        models=config["experiment"]["models"],
        settings=config["models"],
        backend_factory=Backend,
    )
    assert preview["models"][0]["controls"]["effective"]["num_predict"] == 32
    assert preview["sample_count"] == 2
    assert preview["inference"] is False
