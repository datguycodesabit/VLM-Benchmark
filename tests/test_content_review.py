"""Content scoring and human-reviewed eligibility must not hide recognition failures."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_cli_completion import data, native

from vlm_bench import cli
from vlm_bench.report_model import evaluate_run
from vlm_bench.sample_review import reference_hash, validate_review
from vlm_bench.scoring_policy import CONTENT_NORMALIZATION_VERSION, ScoringPolicy, normalize_prose


def sample(sid, reference, **kw):
    return {"id": sid, "reference": reference, "verified": True, "hashes": {"crop": "b" * 64}, **kw}


def record(sid, prediction, **kw):
    return {"model": "m", "sample_id": sid, "prediction": prediction, "status": "success", **kw}


def content_policy(**kw):
    return ScoringPolicy(normalization=CONTENT_NORMALIZATION_VERSION, **kw)


def reviewed(samples):
    return {
        "schema_version": 1,
        "benchmark_fingerprint": "a" * 64,
        "reviewer": "Test human reviewer",
        "reviewed_at": "2026-11-06T15:00:00-06:00",
        "samples": [
            {
                "sample_id": s["id"],
                "crop_sha256": s["hashes"]["crop"],
                "reference_sha256": reference_hash(s["reference"]),
                "decision": "keep",
                "reason_code": "",
                "reason": "",
            }
            for s in samples
        ],
    }


def test_content_ignores_punctuation_and_all_spacing_but_keeps_case_digits_and_symbols():
    assert normalize_prose("A, b!\nC", CONTENT_NORMALIZATION_VERSION) == "AbC"
    assert normalize_prose("e\u0301 — 2 + $", CONTENT_NORMALIZATION_VERSION) == "é2+$"
    samples = [
        sample("ok", "A, b!"),
        sample("case", "Ab"),
        sample("symbol", "a+b"),
        sample("equation", "x^2", content_type="equation"),
    ]
    rows = [
        record("ok", "A b"),
        record("case", "ab"),
        record("symbol", "ab"),
        record("equation", "x2"),
    ]
    result = evaluate_run(samples, rows, policy=content_policy())
    prose = result["tracks"]["prose"]["models"][0]
    assert prose["exact_match_rate"] == pytest.approx(1 / 3)
    assert prose["character_edits"] == 2
    assert prose["reference_chars"] == 7
    assert prose["wer"] is None
    assert result["tracks"]["prose"]["wer_unavailable_reason"] == "word_boundaries_ignored"
    assert result["tracks"]["equation"]["models"][0]["exact_match_rate"] == 0
    assert result["tracks"]["equation"]["models"][0]["wer"] is not None


def test_content_keeps_failure_penalties_and_explains_empty_normalized_references():
    result = evaluate_run(
        [sample("a", "a b!"), sample("punctuation", "---")],
        [record("a", None, status="error")],
        policy=content_policy(),
        planned_passes=2,
    )
    row = result["tracks"]["prose"]["models"][0]
    assert row["score"] == 0 and row["character_edits"] == 4
    assert row["failed_sample_count"] == row["missing_sample_count"] == 1
    assert result["tracks"]["prose"]["excluded_samples"] == [
        {
            "sample_id": "punctuation",
            "track": "prose",
            "reason": "empty_reference_after_normalization",
        }
    ]


def test_completed_review_is_shared_across_models_and_passes_and_changes_only_evaluation():
    samples = [sample("good", "a b!"), sample("broken", "wrong")]
    review = reviewed(samples)
    review["samples"][1].update(
        decision="exclude",
        reason_code="severe_fragmentation",
        reason="Missing lower portions of multiple letters",
    )
    rows = [
        {**record(s["id"], "ab"), "model": m, "pass": p}
        for s in samples
        for m in ["m", "n"]
        for p in [1, 2]
    ]
    manifest = {"benchmark_fingerprint": "a" * 64}
    original = evaluate_run(samples, rows, planned_passes=2, manifest=manifest)
    selected = evaluate_run(
        samples,
        rows,
        planned_passes=2,
        manifest=manifest,
        policy=content_policy(sample_review=review),
    )
    assert selected["raw_evidence_id"] == original["raw_evidence_id"]
    assert selected["evaluation_id"] != original["evaluation_id"]
    assert selected["provenance"]["sample_review_id"]
    for row in selected["tracks"]["prose"]["models"]:
        assert row["eligible_sample_ids"] == ["good"] and row["score"] == 100
    assert selected["policy"]["sample_review"]["reviewer"] == "Test human reviewer"
    changed = copy.deepcopy(review)
    changed["samples"][0]["reason"] = "Unexpected reason on a retained sample"
    with pytest.raises(ValueError):
        validate_review(changed)


@pytest.mark.parametrize(
    "mutation",
    ["pending", "reviewer", "reason", "duplicate", "unknown", "fingerprint", "crop", "reference"],
)
def test_unfinished_or_stale_reviews_cannot_change_scores(mutation):
    samples = [sample("a", "abc")]
    review = reviewed(samples)
    if mutation == "pending":
        review["samples"][0]["decision"] = "pending"
    elif mutation == "reviewer":
        review["reviewer"] = ""
    elif mutation == "reason":
        review["samples"][0]["decision"] = "exclude"
    elif mutation == "duplicate":
        review["samples"].append(copy.deepcopy(review["samples"][0]))
    elif mutation == "unknown":
        review["samples"][0]["sample_id"] = "other"
    elif mutation == "fingerprint":
        review["benchmark_fingerprint"] = "c" * 64
    elif mutation == "crop":
        review["samples"][0]["crop_sha256"] = "c" * 64
    elif mutation == "reference":
        review["samples"][0]["reference_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        evaluate_run(
            samples,
            [record("a", "abc")],
            policy=content_policy(sample_review=review),
            manifest={"benchmark_fingerprint": "a" * 64},
        )


def test_cli_review_report_compare_roundtrip_preserves_original_evidence(tmp_path, capsys):
    prepared = data(tmp_path)
    gallery = tmp_path / "review"
    assert cli.main(["review", "--prepared", str(prepared), "--output", str(gallery)]) == 0
    page = (gallery / "review.html").read_text()
    assert "prediction" not in page.lower().replace("predictions are deliberately absent", "")
    review_path = gallery / "review.json"
    review = json.loads(review_path.read_text())
    assert all(r["decision"] == "pending" for r in review["samples"])
    with pytest.raises(ValueError):
        validate_review(review)
    review.update(reviewer="Test human", reviewed_at="2026-11-06T15:00:00-06:00")
    for r in review["samples"]:
        r["decision"] = "keep"
    review["samples"][0].update(
        decision="exclude",
        reason_code="severe_fragmentation",
        reason="Test exclusion of cropped ink",
    )
    review_path.write_text(json.dumps(review))
    config = tmp_path / "scoring.toml"
    config.write_text(f'version=3\n[scoring]\nnormalization="{CONTENT_NORMALIZATION_VERSION}"\n')
    a = native(prepared, tmp_path / "a", "ollama:a")
    b = native(prepared, tmp_path / "b", "ollama:b")
    originals = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for root in [a, b]
        for p in root.glob("*.json*")
    }
    args = ["--scoring-config", str(config), "--review", str(review_path)]
    assert (
        cli.main(
            [
                "report",
                "--run",
                str(a),
                "--output",
                str(tmp_path / "report"),
                "--formats",
                "json",
                "html",
                "csv",
                *args,
            ]
        )
        == 0
    )
    derived = json.loads((tmp_path / "report" / "evaluation.json").read_text())
    row = derived["conditions"]["baseline"]["tracks"]["prose"]["models"][0]
    assert row["eligible_sample_count"] == 2 and row["score"] == 100 and row["wer"] is None
    assert (tmp_path / "report" / "exclusions.csv").is_file()
    assert (
        cli.main(
            ["compare", "--runs", str(a), str(b), "--output", str(tmp_path / "compare"), *args]
        )
        == 0
    )
    comparison = json.loads((tmp_path / "compare" / "comparison.json").read_text())
    for r in comparison["conditions"]["baseline"]["tracks"]["prose"]["models"]:
        assert r["eligible_sample_count"] == 2 and r["wer"] is None
    for name, digest in originals.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
