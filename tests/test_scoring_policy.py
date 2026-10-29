import pytest

from vlm_bench.scoring_policy import ScoringPolicy


def test_default_policy_is_versioned_and_matches_product_weights():
    policy = ScoringPolicy()

    assert policy.to_dict() == {
        "version": 1,
        "character_accuracy_weight": 0.7,
        "exact_match_weight": 0.3,
        "accuracy_margin_points": 0.5,
        "missing_and_failed_as_empty": True,
        "pass_aggregation": "arithmetic_mean",
        "normalization": "whitespace-v1",
    }


def test_policy_accepts_config_aliases_and_rejects_invalid_penalty_or_weights():
    policy = ScoringPolicy.from_dict(
        {"scoring": {"char_weight": 0.6, "exact_weight": 0.4, "accuracy_margin": 1}}
    )

    assert policy.character_accuracy_weight == pytest.approx(0.6)
    assert policy.accuracy_margin_points == 1
    with pytest.raises(ValueError, match="must receive the empty-prediction penalty"):
        ScoringPolicy(missing_and_failed_as_empty=False)
    with pytest.raises(ValueError, match="sum to 1"):
        ScoringPolicy(character_accuracy_weight=0.6, exact_match_weight=0.3)
    with pytest.raises(ValueError, match="unknown scoring policy field"):
        ScoringPolicy.from_dict({"char_weight": 0.7, "exact_weight": 0.3, "typo": 1})


def test_policy_version_must_be_integer_and_supported():
    with pytest.raises(TypeError, match="version must be an integer"):
        ScoringPolicy(version=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported scoring policy version"):
        ScoringPolicy(version=2)
