"""Research-run eligibility checks and split-leakage audits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

_PROTOCOLS = {"document-disjoint", "writer-disjoint"}
_TEST_SPLITS = {"test", "testing", "eval", "evaluation", "heldout", "held-out"}
_SPLIT_ALIASES = {
    "testing": "test",
    "eval": "test",
    "evaluation": "test",
    "heldout": "test",
    "held-out": "test",
    "training": "train",
    "valid": "validation",
    "dev": "validation",
}
_VERIFIED_VALUES = {"true", "yes", "verified", "approved"}


def _sample_metadata(sample: Mapping[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata")
    result = dict(metadata) if isinstance(metadata, Mapping) else {}
    for key in (
        "source_document",
        "document_id",
        "writer_id",
        "split",
        "verified",
        "verification_status",
        "reference",
    ):
        value = sample.get(key)
        if value not in (None, ""):
            result[key] = value
    return result


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _verified_state(sample: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool | None:
    """Return True/False when verification is explicit, else None."""
    states: list[bool] = []
    explicit = False
    for key in ("verified", "verification_status"):
        value = metadata.get(key)
        if value not in (None, ""):
            explicit = True
        if isinstance(value, bool):
            states.append(value)
            continue
        normalized = _text(value)
        if normalized:
            lowered = normalized.lower().replace("_", " ").replace("-", " ")
            states.append(lowered in _VERIFIED_VALUES)
    if not explicit:
        return None
    return all(states) if states else False


def _split_value(metadata: Mapping[str, Any]) -> str | None:
    value = _text(metadata.get("split"))
    if not value:
        return None
    normalized = value.lower().replace("_", "-")
    return _SPLIT_ALIASES.get(normalized, normalized)


def _split_overlaps(
    samples: Sequence[Mapping[str, Any]], field: str, finding_type: str
) -> list[dict[str, Any]]:
    by_group: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for sample in samples:
        metadata = _sample_metadata(sample)
        group = _text(metadata.get(field))
        split = _split_value(metadata)
        sample_id = _text(sample.get("id")) or "<unknown>"
        if group and split:
            by_group[group][split].add(sample_id)
    findings = []
    for group, splits in sorted(by_group.items()):
        if len(splits) < 2:
            continue
        ids = sorted(sample_id for values in splits.values() for sample_id in values)
        findings.append(
            {
                "type": finding_type,
                field: group,
                "splits": sorted(splits),
                "sample_ids": ids,
            }
        )
    return findings


def validate_research(
    samples: Sequence[Mapping[str, Any]],
    strict_research: bool = False,
    protocol: str = "document-disjoint",
) -> dict[str, Any]:
    """Audit sample metadata and optionally enforce research-run requirements.

    Metadata may be stored at the sample's top level or in its ``metadata``
    object. Strict runs require an explicit verified reference, a test split,
    a source document, and no leakage under the selected split protocol.
    Non-strict calls return the same findings without rejecting the samples.
    """
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise TypeError("samples must be a sequence of sample mappings")
    if any(not isinstance(sample, Mapping) for sample in samples):
        raise TypeError("every sample must be a mapping")
    if not isinstance(strict_research, bool):
        raise TypeError("strict_research must be a boolean")
    if protocol not in _PROTOCOLS:
        raise ValueError("protocol must be 'document-disjoint' or 'writer-disjoint'")

    findings: list[dict[str, Any]] = []
    split_counts: dict[str, int] = defaultdict(int)
    documents: set[str] = set()
    writers: set[str] = set()
    eligible_count = 0

    for sample in samples:
        metadata = _sample_metadata(sample)
        sample_id = _text(sample.get("id")) or "<unknown>"
        split = _split_value(metadata)
        document = _text(metadata.get("source_document"))
        writer = _text(metadata.get("writer_id"))
        verification = _verified_state(sample, metadata)

        if split:
            split_counts[split] += 1
        if document:
            documents.add(document)
        if writer:
            writers.add(writer)

        if verification is None:
            findings.append({"type": "missing_verification_status", "sample_id": sample_id})
        elif not verification:
            findings.append({"type": "reference_not_verified", "sample_id": sample_id})
        if split is None:
            findings.append({"type": "missing_test_split", "sample_id": sample_id})
        elif split not in _TEST_SPLITS:
            findings.append({"type": "not_test_split", "sample_id": sample_id, "split": split})
        if not document:
            findings.append({"type": "missing_source_document", "sample_id": sample_id})
        if protocol == "writer-disjoint" and not writer:
            findings.append({"type": "missing_writer_id", "sample_id": sample_id})
        if not _text(metadata.get("reference")):
            findings.append({"type": "missing_reference", "sample_id": sample_id})
        if (
            verification is True
            and split in _TEST_SPLITS
            and document
            and _text(metadata.get("reference"))
            and (protocol != "writer-disjoint" or writer)
        ):
            eligible_count += 1

    document_overlaps = _split_overlaps(samples, "source_document", "document_split_overlap")
    writer_overlaps = _split_overlaps(samples, "writer_id", "writer_split_overlap")
    findings.extend(document_overlaps)
    findings.extend(writer_overlaps)

    blocking_types = {
        "missing_verification_status",
        "reference_not_verified",
        "missing_test_split",
        "not_test_split",
        "missing_source_document",
        "missing_reference",
        "document_split_overlap",
    }
    if protocol == "writer-disjoint":
        blocking_types.update({"writer_split_overlap", "missing_writer_id"})
    blocking_findings = [finding for finding in findings if finding["type"] in blocking_types]

    report = {
        "strict_research": strict_research,
        "protocol": protocol,
        "sample_count": len(samples),
        "split_counts": dict(sorted(split_counts.items())),
        "source_document_count": len(documents),
        "writer_count": len(writers),
        "eligible_sample_count": eligible_count,
        "findings": findings,
        "valid": not blocking_findings,
        "audit_scope": "provided_samples_only",
    }
    if strict_research and blocking_findings:
        details = "; ".join(_describe_finding(finding) for finding in blocking_findings[:12])
        omitted = len(blocking_findings) - 12
        if omitted > 0:
            details += f"; and {omitted} additional finding(s)"
        raise ValueError(f"Strict research validation failed: {details}")
    return report


def _describe_finding(finding: Mapping[str, Any]) -> str:
    finding_type = finding.get("type")
    sample_id = finding.get("sample_id")
    if finding_type == "missing_verification_status":
        return f"sample {sample_id!r} has no explicit verification status"
    if finding_type == "reference_not_verified":
        return f"sample {sample_id!r} reference is not verified"
    if finding_type == "missing_test_split":
        return f"sample {sample_id!r} has no split label"
    if finding_type == "not_test_split":
        return f"sample {sample_id!r} is assigned to split {finding.get('split')!r}, not test"
    if finding_type == "missing_source_document":
        return f"sample {sample_id!r} has no source_document"
    if finding_type == "missing_writer_id":
        return f"sample {sample_id!r} has no writer_id for writer-disjoint evaluation"
    if finding_type == "missing_reference":
        return f"sample {sample_id!r} has no reference"
    if finding_type in {"document_split_overlap", "writer_split_overlap"}:
        field = "source_document" if finding_type == "document_split_overlap" else "writer_id"
        return (
            f"{field} {finding.get(field)!r} crosses splits "
            f"{finding.get('splits')!r} for samples {finding.get('sample_ids')!r}"
        )
    return str(finding_type)
