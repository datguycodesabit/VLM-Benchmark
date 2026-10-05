from __future__ import annotations

import json

import pytest

from vlm_bench.research import normalize_equation, research_reports, score_equation


def _record(model: str, sample_id: str, prediction: str, reference: str, **extra):
    return {
        "model": model,
        "sample_id": sample_id,
        "status": "success",
        "prediction": prediction,
        "reference": reference,
        **extra,
    }


def test_equation_score_uses_conservative_unicode_and_whitespace_normalization():
    assert normalize_equation("  x  +\n y ") == "x + y"
    assert normalize_equation("e\u0301 = 1") == "é = 1"
    assert score_equation("x  ^ 2", "x ^ 2")["normalized_exact_match"] is True
    assert score_equation("x^2", "x^3")["equation_cer"] > 0
    assert "not mathematical equivalence" in score_equation("x", "x")["normalization"]
    with pytest.raises(ValueError, match="equation reference"):
        score_equation("x", " \n")


def test_research_report_ranks_same_verified_test_set_and_bootstraps_by_document():
    samples = [
        {
            "id": "a1",
            "reference": "abc",
            "split": "test",
            "verified": True,
            "document_id": "exam-a",
        },
        {
            "id": "a2",
            "reference": "hello",
            "split": "test",
            "verified": True,
            "document_id": "exam-a",
        },
        {
            "id": "b1",
            "reference": "physics",
            "split": "test",
            "verified": True,
            "document_id": "exam-b",
        },
        {
            "id": "train1",
            "reference": "leak",
            "split": "train",
            "verified": True,
            "document_id": "exam-c",
        },
        {
            "id": "uncertain",
            "reference": "maybe",
            "split": "test",
            "verified": False,
            "document_id": "exam-d",
        },
        {"id": "unassigned", "reference": "maybe", "verified": True, "document_id": "exam-f"},
        {
            "id": "eq1",
            "reference": r"F = ma",
            "content_type": "equation",
            "split": "test",
            "verified": True,
            "document_id": "exam-e",
        },
    ]
    records = []
    for model, edits in (("ollama:local", False), ("openai:remote", True)):
        for sample_id, reference in (("a1", "abc"), ("a2", "hello"), ("b1", "physics")):
            prediction = reference if not edits else reference[:-1] + "?"
            records.append(_record(model, sample_id, prediction, reference, latency_seconds=0.2))
        records.append(_record(model, "eq1", r"F = ma" if not edits else r"F = mb", r"F = ma"))
    # Standard TrOCR should be visible as unsupported for equations.
    for sample_id, reference in (("a1", "abc"), ("a2", "hello"), ("b1", "physics")):
        records.append(
            _record("trocr:microsoft/trocr-base-handwritten", sample_id, reference, reference)
        )

    manifest = {
        "seed": 19,
        "samples": samples,
        "models": ["ollama:local", "openai:remote", "trocr:microsoft/trocr-base-handwritten"],
        "research": {"bootstrap_replicates": 250, "bootstrap_seed": 7},
    }
    report = research_reports(records, manifest)
    prose = report["tracks"]["prose"]
    equation = report["tracks"]["equation"]

    assert prose["eligible_sample_ids"] == ["a1", "a2", "b1"]
    assert {item["sample_id"]: item["reason"] for item in prose["excluded_samples"]} == {
        "train1": "not_test_split",
        "uncertain": "reference_unverified",
        "unassigned": "missing_split",
    }
    assert [row["model"] for row in prose["common_eligible_ranking"]] == [
        "ollama:local",
        "trocr:microsoft/trocr-base-handwritten",
        "openai:remote",
    ]
    paired = next(
        row
        for row in prose["paired_differences"]
        if row["left_model"] == "ollama:local" and row["right_model"] == "openai:remote"
    )
    assert paired["paired_sample_count"] == 3
    assert paired["corpus_cer_difference_right_minus_left"] > 0
    assert paired["bootstrap"]["groups"] == 2
    assert paired["bootstrap"]["ci95"] is not None
    assert paired == next(
        row
        for row in research_reports(records, manifest)["tracks"]["prose"]["paired_differences"]
        if row["left_model"] == "ollama:local" and row["right_model"] == "openai:remote"
    )

    trocr_equation = next(
        row for row in equation["model_results"] if row["model"].startswith("trocr:")
    )
    assert trocr_equation["status"] == "unsupported"
    assert trocr_equation["rank"] is None
    assert equation["eligible_sample_ids"] == ["eq1"]
    json.dumps(report, allow_nan=False)


def test_incomplete_and_not_run_models_are_visible_but_not_ranked():
    manifest = {
        "samples": [
            {"id": "one", "reference": "abc", "content_type": "prose", "verified": True},
            {"id": "two", "reference": "def", "content_type": "prose", "verified": True},
        ],
        "models": ["ollama:partial", "openai:not-started"],
    }
    report = research_reports([_record("ollama:partial", "one", "abc", "abc")], manifest)
    prose = report["tracks"]["prose"]
    by_name = {row["model"]: row for row in prose["model_results"]}
    assert by_name["ollama:partial"]["status"] == "incomplete"
    assert by_name["ollama:partial"]["missing_sample_ids"] == ["two"]
    assert by_name["ollama:partial"]["rank"] is None
    assert by_name["openai:not-started"]["status"] == "not_run"
    assert prose["common_eligible_ranking"] == []


def test_word_samples_are_reported_as_a_separate_diagnostic_track():
    manifest = {
        "samples": [
            {"id": "word-1", "reference": "physics", "sample_type": "word", "verified": True}
        ],
        "models": ["ollama:model"],
    }
    report = research_reports([_record("ollama:model", "word-1", "physics", "physics")], manifest)
    assert report["tracks"]["word"]["eligible_sample_ids"] == ["word-1"]
    assert report["tracks"]["word"]["model_results"][0]["exact_match_rate"] == 1
    assert report["tracks"]["prose"]["eligible_sample_ids"] == []


def test_report_reads_nested_metadata_and_excludes_unresolved_equation_references():
    manifest = {
        "samples": [
            {
                "id": "eq-unresolved",
                "reference": r"x^2",
                "content_type": None,
                "source_document": None,
                "metadata": {
                    "content_type": "equation",
                    "source_document": "exam-1",
                    "verification_status": "unresolved",
                },
            }
        ],
        "models": ["ollama:model"],
    }
    report = research_reports([_record("ollama:model", "eq-unresolved", r"x^2", r"x^2")], manifest)
    assert report["tracks"]["equation"]["eligible_sample_ids"] == []
    assert report["tracks"]["equation"]["excluded_samples"] == [
        {
            "sample_id": "eq-unresolved",
            "track": "equation",
            "reason": "reference_unverified",
        }
    ]


def test_provider_reported_unsupported_task_is_visible_at_model_level():
    manifest = {
        "samples": [
            {"id": "eq1", "reference": r"x^2", "content_type": "equation", "verified": True}
        ],
        "models": ["openai:model"],
    }
    row = _record("openai:model", "eq1", "", r"x^2", status="unsupported")
    report = research_reports([row], manifest)
    model = report["tracks"]["equation"]["model_results"][0]
    assert model["status"] == "unsupported"
    assert model["unsupported_reason"] == "provider_marked_task_unsupported"
    assert model["unsupported_sample_ids"] == ["eq1"]
    assert model["failed_sample_count"] == 0


def test_report_includes_inference_end_to_end_failure_and_cold_load_timings():
    model = "ollama:timed"
    manifest = {
        "samples": [
            {"id": "one", "reference": "a", "content_type": "prose", "verified": True},
            {"id": "two", "reference": "b", "content_type": "prose", "verified": True},
            {"id": "three", "reference": "c", "content_type": "prose", "verified": True},
        ],
        "models": [model],
        "warmups": [
            {
                "model": model,
                "sample_id": "one",
                "status": "success",
                "response": {"load_duration": 300_000_000},
            }
        ],
    }
    records = [
        _record(
            model,
            "one",
            "",
            "a",
            latency_seconds=2.0,
            inference_latency_seconds=1.5,
            load_duration_seconds=0.4,
        ),
        _record(
            model,
            "two",
            "b",
            "b",
            latency_seconds=1.0,
            inference_latency_seconds=0.9,
            load_duration_seconds=0.0,
        ),
        _record(model, "three", None, "c", status="error", latency_seconds=0.5),
    ]
    result = research_reports(records, manifest)["tracks"]["prose"]["model_results"][0]

    assert result["failure_rate"] == pytest.approx(1 / 3)
    assert result["empty_count"] == 1
    assert result["mean_latency_seconds"] == pytest.approx(3.5 / 3)
    assert result["median_latency_seconds"] == pytest.approx(1.0)
    assert result["p95_latency_seconds"] == pytest.approx(1.9)
    assert result["samples_per_minute"] == pytest.approx(120 / 3.5)
    assert result["mean_inference_latency_seconds"] == pytest.approx(1.2)
    assert result["median_inference_latency_seconds"] == pytest.approx(1.2)
    assert result["p95_inference_latency_seconds"] == pytest.approx(1.47)
    assert result["total_inference_latency_seconds"] == pytest.approx(2.4)
    assert result["load_duration_seconds"] == pytest.approx(0.4)
    assert result["cold_load_duration_seconds"] == pytest.approx(0.3)
    assert result["cold_load_source"] == "warmup"
    assert result["warmup_count"] == 1


def test_cost_projection_does_not_guess_missing_values_and_calculates_break_even():
    unknown = research_reports([], {"models": [], "samples": []})["cost_scenarios"]
    assert unknown["break_even"] == {"status": "unknown_missing_cost_inputs", "samples": None}
    assert unknown["scenarios"][0]["local_total_cost_usd"] is None
    assert unknown["inputs"]["api_cost_per_sample_usd"] is None

    manifest = {
        "samples": [],
        "models": [],
        "research": {
            "costs": {
                "volumes": [100, 1_000],
                "local": {
                    "model": "ollama:qwen",
                    "upfront_cost_usd": 1_000,
                    "operating_cost_per_sample_usd": 0.01,
                },
                "api": {"model": "openai:gpt", "cost_per_sample_usd": 0.11},
            }
        },
    }
    costs = research_reports([], manifest)["cost_scenarios"]
    assert costs["scenarios"][0]["local_total_cost_usd"] == pytest.approx(1_001)
    assert costs["scenarios"][0]["api_total_cost_usd"] == pytest.approx(11)
    assert costs["break_even"] == {"status": "available", "samples": 10_000}


def test_api_cost_uses_only_exact_selector_usage_and_explicit_token_rates():
    manifest = {
        "samples": [],
        "models": [],
        "costs": {
            "volumes": [2],
            "local": {"upfront_cost_usd": 0, "operating_cost_per_sample_usd": 0.01},
            "api": {
                "model": "openai:model-a",
                "input_per_million_usd": 10,
                "cached_input_per_million_usd": 2,
                "output_per_million_usd": 20,
            },
        },
    }
    records = [
        {
            **_record("openai:model-a", "one", "a", "a"),
            "usage": {
                "input_tokens": 1_000,
                "output_tokens": 500,
                "input_tokens_details": {"cached_tokens": 200},
            },
        },
        {
            **_record("openai:model-b", "two", "b", "b"),
            "usage": {"input_tokens": 10_000, "output_tokens": 10_000},
        },
    ]
    costs = research_reports(records, manifest)["cost_scenarios"]
    estimate = costs["api_usage_estimate"]
    assert estimate["usage_record_count"] == 1
    assert estimate["input_tokens"] == 1_000
    assert estimate["cached_input_tokens"] == 200
    assert estimate["estimated_observed_cost_usd"] == pytest.approx(0.0184)
    assert costs["scenarios"][0]["api_total_cost_usd"] == pytest.approx(0.0368)


def test_api_cost_stays_unknown_when_usage_is_absent():
    manifest = {
        "samples": [],
        "models": [],
        "costs": {
            "api": {
                "model": "openai:model-a",
                "input_per_million_usd": 1,
                "output_per_million_usd": 1,
            }
        },
    }
    costs = research_reports([], manifest)["cost_scenarios"]
    assert costs["api_usage_estimate"]["source"] == "unknown_no_usage_records"
    assert costs["scenarios"][0]["api_total_cost_usd"] is None


def test_api_cost_stays_unknown_if_a_request_or_cached_rate_is_unpriced():
    manifest = {
        "samples": [],
        "models": [],
        "costs": {
            "api": {
                "model": "openai:model-a",
                "input_per_million_usd": 1,
                "output_per_million_usd": 1,
            }
        },
    }
    records = [
        {
            **_record("openai:model-a", "one", "a", "a"),
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 50},
            },
        },
        _record("openai:model-a", "two", "b", "b", status="error"),
    ]
    costs = research_reports(records, manifest)["cost_scenarios"]
    assert costs["api_usage_estimate"]["unpriced_attempt_count"] == 2
    assert costs["api_usage_estimate"]["source"] == "unknown_incomplete_usage_or_cached_rate"
    assert costs["scenarios"][0]["api_total_cost_usd"] is None


def test_research_report_rejects_duplicate_model_sample_and_bad_bootstrap_settings():
    row = _record("ollama:model", "one", "x", "x")
    with pytest.raises(ValueError, match="duplicate result"):
        research_reports([row, row], {"samples": [{"id": "one", "reference": "x"}]})
    with pytest.raises(ValueError, match="bootstrap_replicates"):
        research_reports([], {"research": {"bootstrap_replicates": 0}})
