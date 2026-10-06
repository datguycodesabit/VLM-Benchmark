from __future__ import annotations

import pytest

from vlm_bench.eligibility import validate_research


def _sample(sample_id: str, **metadata):
    return {"id": sample_id, "reference": "reference", "metadata": metadata}


def test_strict_research_accepts_explicit_verified_test_samples():
    report = validate_research(
        [
            _sample(
                "line-1",
                verification_status="verified",
                split="test",
                source_document="exam-a",
                writer_id="writer-1",
            )
        ],
        strict_research=True,
    )

    assert report["valid"] is True
    assert report["eligible_sample_count"] == 1
    assert report["split_counts"] == {"test": 1}
    assert report["audit_scope"] == "provided_samples_only"


def test_strict_research_reports_all_missing_requirements():
    with pytest.raises(ValueError) as error:
        validate_research([{"id": "line-1", "reference": "reference"}], strict_research=True)

    message = str(error.value)
    assert "verification status" in message
    assert "split label" in message
    assert "source_document" in message


def test_explicit_unverified_status_wins_over_conflicting_verified_boolean():
    sample = {
        "id": "line-1",
        "reference": "reference",
        "verified": True,
        "verification_status": "unverified",
        "split": "test",
        "source_document": "exam-a",
    }

    with pytest.raises(ValueError, match="reference is not verified"):
        validate_research([sample], strict_research=True)


def test_document_overlap_is_reported_and_rejected_under_either_protocol():
    samples = [
        {
            "id": "line-train",
            "reference": "a",
            "verified": True,
            "split": "train",
            "source_document": "exam-a",
            "writer_id": "writer-1",
        },
        {
            "id": "line-test",
            "reference": "b",
            "verified": True,
            "split": "test",
            "source_document": "exam-a",
            "writer_id": "writer-2",
        },
    ]

    report = validate_research(samples)
    assert any(item["type"] == "document_split_overlap" for item in report["findings"])
    with pytest.raises(ValueError, match="source_document 'exam-a' crosses splits"):
        validate_research(samples, strict_research=True, protocol="writer-disjoint")


def test_writer_disjoint_requires_writer_ids_and_rejects_writer_overlap():
    missing_writer = _sample("line-1", verified=True, split="test", source_document="exam-a")
    with pytest.raises(ValueError, match="no writer_id"):
        validate_research([missing_writer], strict_research=True, protocol="writer-disjoint")

    samples = [
        _sample(
            "line-train",
            verified=True,
            split="train",
            source_document="exam-a",
            writer_id="writer-1",
        ),
        _sample(
            "line-test",
            verified=True,
            split="test",
            source_document="exam-b",
            writer_id="writer-1",
        ),
    ]
    report = validate_research(samples)
    assert any(item["type"] == "writer_split_overlap" for item in report["findings"])
    with pytest.raises(ValueError, match="writer_id 'writer-1' crosses splits"):
        validate_research(samples, strict_research=True, protocol="writer-disjoint")


def test_eligible_sample_count_excludes_missing_reference_or_writer():
    samples = [
        {
            "id": "missing-reference",
            "verified": True,
            "split": "test",
            "source_document": "exam-a",
            "writer_id": "writer-1",
        },
        {
            "id": "missing-writer",
            "reference": "reference",
            "verified": True,
            "split": "test",
            "source_document": "exam-b",
        },
    ]

    assert validate_research(samples)["eligible_sample_count"] == 1
    assert validate_research(samples, protocol="writer-disjoint")["eligible_sample_count"] == 0


def test_test_split_aliases_are_canonicalized_for_leakage_audit():
    samples = [
        _sample(
            "line-1",
            verified=True,
            split="evaluation",
            source_document="exam-a",
        ),
        _sample(
            "line-2",
            verified=True,
            split="test",
            source_document="exam-a",
        ),
    ]

    report = validate_research(samples, strict_research=True)

    assert not any("split_overlap" in item["type"] for item in report["findings"])
