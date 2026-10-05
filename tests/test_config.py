import pytest

from vlm_bench.config import load_config


def test_config_preserves_per_model_settings_and_cost_assumptions(tmp_path):
    path = tmp_path / "experiment.toml"
    path.write_text(
        'version=1\n[experiment]\nmodels=["trocr:test"]\npreprocessing=["original","enhanced"]\n[models."trocr:test"]\nnum_beams=2\n[costs.local]\nupfront_cost_usd=100\n'
    )
    config = load_config(path)
    assert config["models"]["trocr:test"]["num_beams"] == 2
    assert config["experiment"]["preprocessing"] == ["original", "enhanced"]


@pytest.mark.parametrize(
    "text",
    [
        "version=2",
        'version=1\n[models."openai:test"]\napi_key="secret"',
        "version=1\n[costs]\namount=-1",
    ],
)
def test_rejects_invalid_version_credentials_and_costs(tmp_path, text):
    path = tmp_path / "experiment.toml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)
