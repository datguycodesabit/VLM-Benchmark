from __future__ import annotations

import pytest

from vlm_bench.metrics import score, summarize


def test_score_collapses_whitespace_but_keeps_case_and_punctuation():
    result = score("Hello,\nworld!", "Hello, world!")

    assert result["cer"] == 0
    assert result["wer"] == 0
    assert result["raw_cer"] == pytest.approx(1 / 13)
    assert result["exact_match"] is True
    assert result["reference_chars"] == 13
    assert result["reference_words"] == 2

    changed_case = score("hello, world!", "Hello, world!")
    assert changed_case["cer"] == pytest.approx(1 / 13)
    assert changed_case["exact_match"] is False


@pytest.mark.parametrize(
    ("prediction", "reference", "substitutions", "deletions", "insertions"),
    [
        ("cut", "cat", 1, 0, 0),
        ("ct", "cat", 0, 1, 0),
        ("cats", "cat", 0, 0, 1),
        ("", "a b", 0, 3, 0),
    ],
)
def test_score_reports_character_edit_operations(
    prediction, reference, substitutions, deletions, insertions
):
    result = score(prediction, reference)

    assert result["char_substitutions"] == substitutions
    assert result["char_deletions"] == deletions
    assert result["char_insertions"] == insertions
    assert result["char_edits"] == substitutions + deletions + insertions


def test_score_uses_word_edits_and_allows_rates_above_one():
    result = score("a b c", "x")

    assert result["cer"] == 5
    assert result["wer"] == 3
    assert result["word_edits"] == 3


def test_score_counts_mixed_edits_and_unicode_code_points():
    mixed = score("sitting", "kitten")
    assert mixed["char_edits"] == 3
    assert mixed["char_substitutions"] == 2
    assert mixed["char_deletions"] == 0
    assert mixed["char_insertions"] == 1

    unicode_result = score("café🙂", "café😃")
    assert unicode_result["reference_chars"] == 5
    assert unicode_result["char_substitutions"] == 1
    assert unicode_result["cer"] == pytest.approx(0.2)


def test_score_rejects_empty_normalized_reference():
    with pytest.raises(ValueError, match="reference must contain"):
        score("anything", " \n\t ")


def test_summary_uses_corpus_weighting_and_reports_latency_and_warmups():
    records = [
        {
            "model": "weighted",
            "sample_id": "short",
            "status": "success",
            "prediction": "b",
            "reference": "a",
            "metrics": score("b", "a"),
            "latency_seconds": 1.0,
            "load_duration_seconds": 0.2,
            "truncated": False,
        },
        {
            "model": "weighted",
            "sample_id": "long",
            "status": "success",
            "prediction": "aaaaaaa",
            "reference": "aaaaaaa",
            "metrics": score("aaaaaaa", "aaaaaaa"),
            "latency_seconds": 3.0,
            "load_duration_seconds": None,
            "truncated": True,
        },
    ]
    warmups = [
        {
            "model": "weighted",
            "sample_id": "short",
            "status": "success",
            "response": {"load_duration": 2_500_000_000},
        },
        {"model": "weighted", "sample_id": "short", "status": "error", "error": "warmup failed"},
    ]

    result = summarize(records, warmups=warmups)[0]

    assert result["cer"] == pytest.approx(1 / 8)
    assert result["mean_sample_cer"] == pytest.approx(0.5)
    assert result["wer"] == pytest.approx(0.5)
    assert result["exact_match_rate"] == pytest.approx(0.5)
    assert result["complete"] is True
    assert result["success_count"] == 2
    assert result["failed_count"] == 0
    assert result["failure_rate"] == 0
    assert result["empty_count"] == 0
    assert result["truncated_count"] == 1
    assert result["mean_latency_seconds"] == pytest.approx(2)
    assert result["median_latency_seconds"] == pytest.approx(2)
    assert result["p95_latency_seconds"] == pytest.approx(2.9)
    assert result["total_latency_seconds"] == pytest.approx(4)
    assert result["samples_per_minute"] == pytest.approx(30)
    assert result["load_duration_seconds"] == pytest.approx(0.2)
    assert result["load_event_count"] == 1
    assert result["warmup_count"] == 2
    assert result["warmup_load_duration_seconds"] == pytest.approx(2.5)


def test_summary_scores_cache_hits_but_excludes_them_from_latency_metrics():
    records = [
        {
            "model": "cached-model",
            "sample_id": "cached",
            "status": "success",
            "prediction": "wrong",
            "reference": "right",
            "cache_hit": True,
            "latency_seconds": 99.0,
            "load_duration_seconds": 50.0,
        },
        {
            "model": "cached-model",
            "sample_id": "measured",
            "status": "success",
            "prediction": "right",
            "reference": "right",
            "latency_seconds": 2.0,
            "load_duration_seconds": 0.25,
        },
    ]

    result = summarize(records)[0]

    assert result["cer"] == pytest.approx(score("wrong", "right")["cer"] / 2)
    assert result["success_count"] == 2
    assert result["scored_sample_count"] == 2
    assert result["complete"] is True
    assert result["cache_hit_count"] == 1
    assert result["measured_latency_sample_count"] == 1
    assert result["mean_latency_seconds"] == pytest.approx(2.0)
    assert result["total_latency_seconds"] == pytest.approx(2.0)
    assert result["samples_per_minute"] == pytest.approx(30.0)
    assert result["load_duration_seconds"] == pytest.approx(0.25)
    assert result["load_event_count"] == 1


def test_summary_ranks_only_complete_models_and_checks_expected_coverage():
    records = [
        {
            "model": "best",
            "sample_id": sample_id,
            "status": "success",
            "prediction": reference,
            "reference": reference,
            "metrics": score(reference, reference),
        }
        for sample_id, reference in [("one", "abc"), ("two", "d")]
    ]
    records.extend(
        [
            {
                "model": "partial",
                "sample_id": "one",
                "status": "success",
                "prediction": "b",
                "reference": "a",
                "metrics": score("b", "a"),
            },
            {
                "model": "failed",
                "sample_id": "one",
                "status": "error",
                "prediction": None,
                "reference": "a",
                "metrics": None,
                "error": "request timed out",
            },
        ]
    )

    rows = summarize(
        records,
        expected_samples=["one", "two"],
        models=["best", "partial", "failed", "not-started"],
    )

    assert [row["model"] for row in rows] == ["best", "failed", "not-started", "partial"]
    assert rows[0]["rank"] == 1 and rows[0]["complete"] is True
    assert all(row["rank"] is None for row in rows[1:])
    partial = next(row for row in rows if row["model"] == "partial")
    assert partial["sample_coverage_complete"] is False
    assert partial["missing_sample_count"] == 1
    failed = next(row for row in rows if row["model"] == "failed")
    assert failed["failed_count"] == 1
    assert failed["failure_rate"] == 1
    assert failed["complete"] is False
    not_started = next(row for row in rows if row["model"] == "not-started")
    assert not_started["sample_count"] == 0
    assert not_started["complete"] is False


def test_summary_does_not_rank_a_manifest_marked_interrupted():
    record = {
        "model": "model",
        "sample_id": "one",
        "status": "success",
        "prediction": "a",
        "reference": "a",
        "metrics": score("a", "a"),
    }

    row = summarize([record], expected_samples=["one"], models=["model"], force_incomplete=True)[0]

    assert row["complete"] is False
    assert row["rank"] is None
