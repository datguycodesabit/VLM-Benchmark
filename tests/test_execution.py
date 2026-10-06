from __future__ import annotations

import math

import pytest

from vlm_bench import execution


def test_cost_meter_aggregates_fixed_costs_across_models_and_reserves_before_scheduling():
    ledger = execution.SpendLedger(limit=1.0)
    costs = {"api": {"cost_per_sample_usd": 0.4}}
    first = execution.CostMeter("openai:first", costs, ledger, provider="openai")
    second = execution.CostMeter("chatgpt:second", costs, ledger, provider="chatgpt")

    assert first.applicable and second.applicable
    assert first.can_schedule()
    first_reservation = first.reserve()
    assert ledger.reserved_usd == pytest.approx(0.4)
    first.finish({"usage": {}}, reserved_usd=first_reservation)
    assert ledger.estimated_usd == pytest.approx(0.4)
    assert second.can_schedule()
    second_reservation = second.reserve()
    second.finish({"usage": {}}, reserved_usd=second_reservation)
    assert ledger.estimated_usd == pytest.approx(0.8)
    assert ledger.reserved_usd == 0
    assert not first.can_schedule()


def test_unknown_cloud_usage_marks_spend_unknown_and_blocks_scheduling():
    ledger = execution.SpendLedger(limit=2.0)
    meter = execution.CostMeter(
        "openai:metered",
        {
            "api": {
                "model": "openai:metered",
                "input_per_million_usd": 2.0,
                "output_per_million_usd": 8.0,
            }
        },
        ledger,
        provider="openai",
    )

    assert meter.applicable
    assert meter.finish({"message": {"content": "text"}}) is None
    assert ledger.estimated_usd == 0
    assert ledger.unknown_spend_count == 1
    assert not meter.can_schedule()


def test_unpriced_cloud_model_is_not_applicable_for_a_spend_limit():
    ledger = execution.SpendLedger(limit=1.0)
    meter = execution.CostMeter("chatgpt:unpriced", {}, ledger, provider="chatgpt")

    assert not meter.applicable
    assert meter.estimate({"usage": {"input_tokens": 1, "output_tokens": 1}}) is None


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {"input_tokens": 1000, "output_tokens": 100, "cached_input_tokens": 400},
            0.0024,
        ),
        (
            {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "input_tokens_details": {"cached_tokens": 400},
            },
            0.0024,
        ),
    ],
)
def test_cost_meter_uses_uncached_cached_and_output_rates(usage, expected):
    meter = execution.CostMeter(
        "openai:metered",
        {
            "api": {
                "model": "openai:metered",
                "input_per_million_usd": 2.0,
                "cached_input_per_million_usd": 0.5,
                "output_per_million_usd": 10.0,
            }
        },
        execution.SpendLedger(limit=None),
        provider="openai",
    )

    assert meter.estimate({"usage": usage}) == pytest.approx(expected)


def test_cost_meter_reads_provider_details_usage():
    meter = execution.CostMeter(
        "chatgpt:metered",
        {
            "api": {
                "input_per_million_usd": 1.0,
                "output_per_million_usd": 2.0,
            }
        },
        execution.SpendLedger(limit=None),
        provider="chatgpt",
    )

    assert meter.estimate(
        {"provider_details": {"usage": {"input_tokens": 500, "output_tokens": 100}}}
    ) == pytest.approx(0.0007)


def test_retry_after_is_honored_and_fallback_delay_is_bounded(monkeypatch):
    provider_delay = execution.RetryableProviderError(
        "limited", retry_after_seconds=2.5, category="rate_limited"
    )
    assert execution.retry_delay(provider_delay, 1) == 2.5

    default_delay = execution.RetryableProviderError("temporary")
    assert execution.retry_delay(default_delay, 2) == pytest.approx(0.2)
    assert execution.retry_delay(default_delay, 100) == 1.0

    sleeps = []
    monkeypatch.setattr(execution.time, "sleep", sleeps.append)
    assert execution.wait_retry(provider_delay, 1) == 2.5
    assert sleeps == [2.5]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_retries": -1}, "max_retries"),
        ({"max_retries": True}, "max_retries"),
        ({"concurrency": 0}, "concurrency"),
        ({"concurrency": False}, "concurrency"),
        ({"max_requests": 0}, "max_requests"),
        ({"max_requests": 1.5}, "max_requests"),
        ({"max_spend_usd": 0}, "max_spend_usd"),
        ({"max_spend_usd": math.inf}, "max_spend_usd"),
    ],
)
def test_execution_controls_reject_invalid_values(options, message):
    values = {
        "max_retries": 2,
        "concurrency": 1,
        "max_requests": None,
        "max_spend_usd": None,
    }
    values.update(options)
    with pytest.raises(ValueError, match=message):
        execution.validate_execution_controls(**values)


def test_retry_after_rejects_non_finite_negative_and_boolean_values():
    for value in (-0.01, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="retry_after_seconds"):
            execution.RetryableProviderError("temporary", retry_after_seconds=value)
