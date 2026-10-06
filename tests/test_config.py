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


def test_execution_controls_are_validated_and_cache_path_is_resolved(tmp_path):
    path = tmp_path / "experiment.toml"
    path.write_text(
        "version=1\n[experiment]\ncache_dir='response-cache'\n"
        "max_retries=0\nconcurrency=3\nmax_requests=20\nmax_spend_usd=1.25\n"
    )

    experiment = load_config(path)["experiment"]

    assert experiment["cache_dir"] == (tmp_path / "response-cache").resolve()
    assert experiment["max_retries"] == 0
    assert experiment["concurrency"] == 3
    assert experiment["max_requests"] == 20
    assert experiment["max_spend_usd"] == 1.25


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
        "version=1\n[experiment]\nmax_retries=-1",
        "version=1\n[experiment]\nmax_retries=true",
        "version=1\n[experiment]\nconcurrency=0",
        "version=1\n[experiment]\nmax_requests=0",
        "version=1\n[experiment]\nmax_spend_usd=0",
        'version=1\n[experiment]\nmax_spend_usd="1.0"',
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


def test_external_selector_is_valid_configuration_for_import_only_system(tmp_path):
    path = tmp_path / "experiment.toml"
    path.write_text('version=1\n[experiment]\nmodels=["external:tesseract"]\n')

    config = load_config(path)

    assert config["experiment"]["models"] == ["external:tesseract"]
    assert validate_settings(["external:tesseract"], {"external:tesseract": {}}) == {
        "external:tesseract": {}
    }


def test_external_selector_rejects_inference_settings():
    with pytest.raises(ValueError, match="not supported for provider 'external'"):
        validate_settings(["external:tesseract"], {"external:tesseract": {"num_predict": 10}})


def test_config_v2_resolves_suite_conditions_and_merges_model_settings(tmp_path):
    path = tmp_path / "experiment.toml"
    path.write_text(
        "version=2\n"
        '[experiment]\nprepared="prepared/default"\nstrict_research=true\n'
        'protocol="writer-disjoint"\n'
        '[models."ollama:vision"]\ntemperature=0.2\nseed=42\n'
        "[suite]\nrepeats=3\n"
        '[[suite.conditions]]\nname="baseline"\nmodels=["ollama:vision"]\n'
        'warmup=false\nprose_prompt="Read the page carefully."\n'
        '[suite.conditions.settings."ollama:vision"]\ntemperature=0.6\n'
        '[[suite.conditions]]\nname="adapted"\nmodels=["vision"]\n'
        'prepared="prepared/adapted"\nrepeats=1\nmath_prompt="Read all symbols."\n'
        "formula_rendering=true\n[suite.conditions.settings.vision]\nseed=0\n"
    )

    config = load_config(path)

    suite = config["suite"]
    baseline, adapted = suite["conditions"]
    assert config["version"] == 2
    assert suite["version"] == 2 and suite["repeats"] == 3
    assert baseline["prepared"] == (tmp_path / "prepared/default").resolve()
    assert baseline["repeats"] == 3
    assert baseline["warmup"] is False and baseline["strict_research"] is True
    assert baseline["protocol"] == "writer-disjoint" and baseline["formula_rendering"] is False
    assert (
        baseline["prose_prompt"] == "Read the page carefully." and baseline["math_prompt"] is None
    )
    assert baseline["settings"] == {"ollama:vision": {"temperature": 0.6, "seed": 42}}
    assert adapted["prepared"] == (tmp_path / "prepared/adapted").resolve()
    assert adapted["repeats"] == 1 and adapted["warmup"] is True
    assert adapted["strict_research"] is True and adapted["protocol"] == "writer-disjoint"
    assert adapted["formula_rendering"] is True
    assert adapted["prose_prompt"] is None and adapted["math_prompt"] == "Read all symbols."
    assert adapted["settings"] == {"vision": {"temperature": 0.2, "seed": 0}}


@pytest.mark.parametrize(
    "text, message",
    [
        (
            'version=2\n[experiment]\ndata="raw"\n[suite]\n[[suite.conditions]]\n'
            'name="x"\nmodels=["ollama:x"]\nprepared="prepared"',
            "frozen prepared snapshots",
        ),
        (
            'version=2\n[suite]\n[[suite.conditions]]\nname="x"\nprepared="p"',
            "non-empty list of model selectors",
        ),
        (
            'version=2\n[suite]\n[[suite.conditions]]\nname="x"\n'
            'models=["ollama:x"]\npreprocess="enhanced"\nprepared="p"',
            "separate prepared snapshot",
        ),
        (
            'version=2\n[suite]\n[[suite.conditions]]\nname="x"\n'
            'models=["external:tesseract"]\nprepared="p"',
            "import-only",
        ),
        (
            'version=2\n[suite]\n[[suite.conditions]]\nname="x"\n'
            'models=["ollama:x"]\nprepared="p"\nrepeats=0',
            "positive integer",
        ),
        (
            'version=2\n[suite]\n[[suite.conditions]]\nname="x"\n'
            'models=["ollama:x"]\nprepared="p"\n'
            '[[suite.conditions]]\nname="x"\nmodels=["ollama:y"]\nprepared="q"',
            "names must be unique",
        ),
        (
            'version=2\n[suite]\n[[suite.conditions]]\nname="x"\nmodels=["ollama:x"]',
            "prepared is required",
        ),
    ],
)
def test_config_v2_rejects_invalid_suite_conditions(tmp_path, text, message):
    path = tmp_path / "experiment.toml"
    path.write_text(text)

    with pytest.raises(ValueError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": 2.1}, "temperature"),
        ({"temperature": float("nan")}, "temperature"),
        ({"temperature": True}, "temperature"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**32}, "seed"),
        ({"seed": True}, "seed"),
    ],
)
def test_ollama_sampling_overrides_are_validated(settings, message):
    with pytest.raises(ValueError, match=message):
        validate_settings(["ollama:vision"], {"ollama:vision": settings})


@pytest.mark.parametrize("provider", ["trocr", "openai", "chatgpt"])
def test_sampling_overrides_are_ollama_only(provider):
    with pytest.raises(ValueError, match="not supported"):
        validate_settings([f"{provider}:vision"], {f"{provider}:vision": {"temperature": 0.5}})


@pytest.mark.parametrize(
    "text",
    [
        'version=1\n[experiment]\nstrict_research="yes"',
        'version=1\n[experiment]\nprotocol="sample-random"',
        'version=1\n[experiment]\nformula_rendering="yes"',
    ],
)
def test_rejects_invalid_research_options(tmp_path, text):
    path = tmp_path / "experiment.toml"
    path.write_text(text)

    with pytest.raises(ValueError, match="strict_research|protocol|formula_rendering"):
        load_config(path)


def test_config_accepts_research_options(tmp_path):
    path = tmp_path / "experiment.toml"
    path.write_text(
        'version=1\n[experiment]\nstrict_research=true\nprotocol="writer-disjoint"\n'
        "formula_rendering=true\n"
    )

    config = load_config(path)

    assert config["experiment"]["strict_research"] is True
    assert config["experiment"]["protocol"] == "writer-disjoint"
    assert config["experiment"]["formula_rendering"] is True
