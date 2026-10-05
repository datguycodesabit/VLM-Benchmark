import pytest

from vlm_bench.config import load_config, validate_settings


def test_config_preserves_settings_costs_and_resolves_paths_from_toml_parent(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    path = project / "experiment.toml"
    path.write_text(
        'version=1\n[experiment]\ndata="data/handwriting"\n'
        'models=["trocr:test"]\npreprocessing=["original","enhanced"]\n'
        '[models."trocr:test"]\nnum_beams=2\n[costs.local]\nupfront_cost_usd=100\n'
    )

    config = load_config(path)

    assert config["experiment"]["data"] == (project / "data/handwriting").resolve()
    assert config["models"]["trocr:test"]["num_beams"] == 2
    assert config["experiment"]["preprocessing"] == ["original", "enhanced"]
    assert config["costs"]["local"]["upfront_cost_usd"] == 100


def test_prepared_path_is_resolved_and_allows_models_but_no_selection_options(tmp_path):
    path = tmp_path / "experiment.toml"
    path.write_text(
        'version=1\n[experiment]\nprepared="snapshots/test"\nmodels=["openai:test"]\nwarmup=false\n'
    )

    config = load_config(path)

    assert config["experiment"]["prepared"] == (tmp_path / "snapshots/test").resolve()


@pytest.mark.parametrize(
    "selection",
    [
        'data="data"',
        'layout="paired"',
        'preprocess="original"',
        'preprocessing=["original"]',
        "limit=3",
        "seed=42",
        'split="test"',
        'content_type="prose"',
    ],
)
def test_prepared_excludes_every_other_selection_setting(tmp_path, selection):
    path = tmp_path / "experiment.toml"
    path.write_text(f'version=1\n[experiment]\nprepared="prepared"\n{selection}\n')

    with pytest.raises(ValueError, match="freezes selection"):
        load_config(path)


@pytest.mark.parametrize(
    "text",
    [
        "version=2",
        "version=true",
        'version=1\n[models."openai:test"]\napi_key="secret"',
        "version=1\n[experiment]\nlimit=0",
        "version=1\n[experiment]\nlimit=true",
        "version=1\n[experiment]\nseed=2.5",
        "version=1\n[experiment]\nseed=-1",
        "version=1\n[experiment]\nseed=4294967296",
        'version=1\n[experiment]\nmodels=["foo", "ollama:foo"]',
        "version=1\n[experiment]\npreprocessing=[]",
        'version=1\n[experiment]\npreprocess="sharpened"',
        'version=1\n[experiment]\nwarmup="yes"',
        'version=1\n[models."trocr:test"]\nnum_beams=0',
        'version=1\n[models."trocr:test"]\nnum_beams=11',
        'version=1\n[models."trocr:test"]\nnum_predict=true',
        'version=1\n[models."trocr:test"]\ndevice="cuda"',
        'version=1\n[models."openai:test"]\nimage_detail="medium"',
        'version=1\n[models."openai:test"]\nreasoning_effort="fast"',
        "version=1\n[costs]\nvolumes=[0, 100]",
        "version=1\n[costs]\nvolumes=[100, 100]",
        'version=1\n[costs.local]\nupfront_cost_usd="100"',
        "version=1\n[costs.api]\ninput_per_million_usd=-1",
        "version=1\n[costs.api]\ncost_per_sample_usd=nan",
    ],
)
def test_rejects_invalid_config_types_ranges_and_values(tmp_path, text):
    path = tmp_path / "experiment.toml"
    path.write_text(text)

    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize(
    "text",
    [
        'version=1\n[models.foo]\nnum_predict=4\n[models."ollama:foo"]\nnum_predict=8',
    ],
)
def test_rejects_duplicate_model_settings_after_canonicalization(tmp_path, text):
    path = tmp_path / "experiment.toml"
    path.write_text(text)

    with pytest.raises(ValueError, match="Duplicate model settings"):
        load_config(path)


def test_validate_settings_checks_only_selected_models_and_returns_selected_keys():
    settings = {
        "trocr:unused": {"image_detail": "high"},
        "trocr:handwritten": {"device": "mps", "num_beams": 4, "num_predict": 250},
        "chatgpt:vision": {
            "num_predict": 1000,
            "image_detail": "high",
            "reasoning_effort": "xhigh",
        },
    }

    selected = validate_settings(["trocr:handwritten", "chatgpt:vision", "ollama:other"], settings)

    assert selected == {
        "trocr:handwritten": {"device": "mps", "num_beams": 4, "num_predict": 250},
        "chatgpt:vision": {
            "num_predict": 1000,
            "image_detail": "high",
            "reasoning_effort": "xhigh",
        },
    }


@pytest.mark.parametrize(
    "models,settings",
    [
        (["trocr:test"], {"trocr:test": {"reasoning_effort": "high"}}),
        (["openai:test"], {"openai:test": {"device": "cpu"}}),
        (["chatgpt:test"], {"chatgpt:test": {"num_beams": 2}}),
        (["ollama:test"], {"ollama:test": {"image_detail": "low"}}),
        (["openai:test"], {"openai:test": {"reasoning_effort": "invalid"}}),
    ],
)
def test_validate_settings_rejects_invalid_or_provider_inapplicable_options(models, settings):
    with pytest.raises(ValueError):
        validate_settings(models, settings)


def test_validate_settings_allows_chatgpt_num_predict_while_recording_unsupported_control():
    assert validate_settings(["chatgpt:model"], {"chatgpt:model": {"num_predict": 512}}) == {
        "chatgpt:model": {"num_predict": 512}
    }


def test_load_config_without_path_returns_empty_defaults():
    assert load_config(None) == {
        "version": 1,
        "experiment": {},
        "models": {},
        "costs": {},
    }
