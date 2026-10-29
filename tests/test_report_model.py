import json
from pathlib import Path

import pytest

from vlm_bench import report_model
from vlm_bench.runner import _digest
from vlm_bench.suite import _MUTABLE_FIELDS


def _sample(sample_id, reference, **fields):
    return {"id": sample_id, "reference": reference, "verified": True, **fields}


def _record(model, sample_id, prediction, *, pass_index=1, status="success", **fields):
    return {
        "model": model,
        "sample_id": sample_id,
        "pass": pass_index,
        "status": status,
        "prediction": prediction,
        **fields,
    }


def test_weighted_score_averages_passes_and_penalizes_failed_and_missing_roster_cells():
    samples = [_sample("line", "abcd")]
    records = [
        _record("model-a", "line", None, status="error", latency_seconds=0.2),
        _record("model-b", "line", "abcd", latency_seconds=0.1),
        _record("model-b", "line", "abce", pass_index=2, latency_seconds=0.3),
    ]

    result = report_model.evaluate_run(
        samples,
        records,
        models=["model-a", "model-b"],
        planned_passes=2,
        run_status="complete",
    )
    prose = result["tracks"]["prose"]
    model_a = next(row for row in prose["models"] if row["model"] == "model-a")
    model_b = next(row for row in prose["models"] if row["model"] == "model-b")

    assert model_a["score"] == 0
    assert model_a["failed_sample_count"] == 1
    assert model_a["missing_sample_count"] == 1
    assert model_b["per_pass"][0]["score"] == 100
    assert model_b["per_pass"][1]["score"] == pytest.approx(52.5)
    assert model_b["score"] == pytest.approx(76.25)
    assert model_b["score_mean"] == model_b["score"]
    assert model_b["score_min"] == pytest.approx(52.5)
    assert model_b["score_max"] == 100
    assert model_b["score_stddev"] == pytest.approx(33.58757210636101)
    assert model_b["completed_pass_count"] == 2
    assert model_b["measured_p95_latency_seconds"] == pytest.approx(0.29)
    assert prose["ranking"]["winner"] is None
    assert prose["ranking"]["winner_reason"] == "incomplete_evaluation"


def test_prose_includes_word_samples_equations_rank_separately_and_equivalence_is_diagnostic():
    samples = [
        _sample("line", "read me", content_type="prose", document_id="doc-1"),
        _sample("word", "hello", sample_type="word", document_id="doc-2"),
        _sample(
            "equation",
            "x^2",
            content_type="equation",
            document_id="doc-3",
            variable_domains={"x": [0, 1]},
        ),
    ]
    records = []
    for model in ("model-a", "model-b"):
        records.extend(
            [
                _record(model, "line", "read me", latency_seconds=0.2),
                _record(model, "word", "hello", latency_seconds=0.2),
                _record(model, "equation", "x^2", latency_seconds=0.2),
            ]
        )
    records[2]["prediction"] = "x^2 + 1"

    result = report_model.evaluate_run(samples, records, planned_passes=1)

    assert result["tracks"]["prose"]["eligible_sample_ids"] == ["line", "word"]
    assert result["tracks"]["equation"]["eligible_sample_ids"] == ["equation"]
    assert result["tracks"]["prose"]["ranking"]["winner"] == "model-a"
    model_a_equation = result["tracks"]["equation"]["models"][0]
    assert model_a_equation["equivalence"]["outcomes"]["different"] == 1
    assert model_a_equation["equivalence"]["influence_on_primary_score"] is False
    assert result["tracks"]["equation"]["ranking"]["winner"] == "model-b"
    assert result["tracks"]["prose"]["uncertainty"]["unit"] == "source_document"
    json.dumps(result, allow_nan=False)


def test_missing_top_band_latency_withholds_winner_and_capability_exclusions_are_explicit():
    samples = [_sample("p", "abc"), _sample("e", "x^2", content_type="equation")]
    records = [
        _record("plain", "p", "abc", latency_seconds=None),
        _record("plain", "e", "x^2", latency_seconds=0.1),
        _record("full", "p", "abc", latency_seconds=0.2),
        _record("full", "e", "x^2", latency_seconds=0.2),
    ]
    manifest = {
        "models": ["plain", "full"],
        "model_capabilities": {"plain": {"supports_equations": False}},
    }

    result = report_model.evaluate_run(samples, records, manifest=manifest)

    assert result["tracks"]["prose"]["ranking"]["winner"] is None
    assert result["tracks"]["prose"]["ranking"]["winner_reason"] == "missing_top_band_latency"
    plain_equation = next(
        row for row in result["tracks"]["equation"]["models"] if row["model"] == "plain"
    )
    assert plain_equation["status"] == "unsupported"
    assert plain_equation["score"] is None
    assert result["tracks"]["equation"]["ranking"]["winner"] == "full"


def test_throughput_or_concurrent_profile_does_not_supply_ranking_latency():
    result = report_model.evaluate_run(
        [_sample("p", "abc")],
        [_record("m", "p", "abc", latency_seconds=0.1)],
        manifest={
            "measurement_profile": {"profile": "throughput", "concurrency": 4},
            "execution_runs": [
                {
                    "pass": 1,
                    "run_id": "/tmp/run-a",
                    "invocations": [
                        {
                            "status": "complete",
                            "workload_wall_seconds": 2.0,
                            "successful_sample_count": 4,
                            "terminal_sample_count": 4,
                            "throughput_numerator": "successful_uncached_sample_results",
                            "throughput_denominator": "workload_wall_seconds",
                        }
                    ],
                }
            ],
        },
    )
    model = result["tracks"]["prose"]["models"][0]

    assert model["measured_median_latency_seconds"] == 0.1
    assert model["median_latency_seconds"] is None
    assert result["tracks"]["prose"]["ranking"]["winner_reason"] == "missing_top_band_latency"
    assert result["throughput"]["successful_samples_per_second"] == 2
    assert result["throughput"]["workload_wall_seconds"] == 2
    assert result["throughput"]["throughput_numerator"] == "successful_uncached_sample_results"
    assert result["throughput"]["throughput_denominator"] == "workload_wall_seconds"


def test_different_model_rosters_clear_all_comparable_placements():
    result = report_model.evaluate_run(
        [_sample("a", "abc"), _sample("b", "def")],
        [
            _record("model-a", "a", None, status="unsupported"),
            _record("model-a", "b", "def"),
            _record("model-b", "a", "abc"),
            _record("model-b", "b", "def"),
        ],
        models=["model-a", "model-b"],
    )
    track = result["tracks"]["prose"]

    assert track["roster_compatible"] is False
    assert track["ranking"]["status"] == "incompatible_rosters"
    assert track["ranking"]["bands"] == []
    assert all(row["rank"] is None for row in track["models"])


def test_report_ids_are_evidence_and_policy_specific_and_future_schema_is_rejected():
    samples = [_sample("p", "abc")]
    first = report_model.evaluate_run(samples, [_record("m", "p", "abc")])
    same = report_model.evaluate_run(samples, [_record("m", "p", "abc")])
    changed = report_model.evaluate_run(samples, [_record("m", "p", "abd")])

    assert first["evaluation_id"] == same["evaluation_id"]
    assert first["evaluation_id"] != changed["evaluation_id"]
    with pytest.raises(ValueError, match="unsupported evaluation schema version"):
        report_model.evaluate_run(samples, [], manifest={"evaluation_schema_version": 99})


def test_build_report_groups_experiment_passes_and_suite_conditions_independently(
    tmp_path, monkeypatch
):
    sample = _sample("p", "abc", document_id="doc-1")
    policy = {"version": 1}

    def fake_load(run_dir):
        name = Path(run_dir).name
        fingerprint = "fp-a" if name.startswith("a") else "fp-b"
        return (
            {
                "samples": [sample],
                "models": ["m"],
                "benchmark_fingerprint": fingerprint,
                "preprocess": "p",
                "prompt_hashes": {"prose": fingerprint},
                "scoring_policy": policy,
                "status": "complete",
            },
            [_record("m", "p", "abc", latency_seconds=0.1)],
        )

    monkeypatch.setattr(report_model, "_load_validated_run", fake_load)
    run_a = tmp_path / "a-run-1"
    run_b = tmp_path / "a-run-2"
    run_c = tmp_path / "b-run-1"
    for run in (run_a, run_b, run_c):
        run.mkdir()

    experiment = tmp_path / "experiment.json"
    experiment.write_text(
        json.dumps(
            {
                "evaluation_schema_version": 2,
                "condition": "default",
                "planned_passes": 2,
                "status": "complete",
                "benchmark_fingerprint": "fp-a",
                "scoring_policy": policy,
                "runs": [str(run_a), str(run_b)],
            }
        ),
        encoding="utf-8",
    )
    experiment_report = report_model.build_report(experiment)
    assert (
        experiment_report["conditions"]["default"]["tracks"]["prose"]["models"][0]["score"] == 100
    )

    suite_manifest = {
        "schema_version": 1,
        "suite_id": "suite",
        "status": "complete",
        "conditions": [
            {"name": "prompt-a", "repeats": 1, "benchmark_fingerprint": "fp-a"},
            {"name": "prompt-b", "repeats": 1, "benchmark_fingerprint": "fp-b"},
        ],
        "runs": {
            "prompt-a\u00001": {"condition": "prompt-a", "repetition": 1, "run_dir": str(run_a)},
            "prompt-b\u00001": {"condition": "prompt-b", "repetition": 1, "run_dir": str(run_c)},
        },
        "config_sha256": "a" * 64,
    }
    suite_manifest["integrity"] = _digest(
        {key: value for key, value in suite_manifest.items() if key not in _MUTABLE_FIELDS}
    )
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite_manifest), encoding="utf-8")

    suite_report = report_model.build_report(suite_path)
    assert set(suite_report["conditions"]) == {"prompt-a", "prompt-b"}
    assert suite_report["conditions"]["prompt-a"]["tracks"]["prose"]["eligible_sample_count"] == 1
    assert suite_report["conditions"]["prompt-b"]["tracks"]["prose"]["eligible_sample_count"] == 1


def test_build_report_does_not_apply_default_weighted_policy_to_legacy_run(tmp_path, monkeypatch):
    run_dir = tmp_path / "legacy-run"
    run_dir.mkdir()
    monkeypatch.setattr(
        report_model,
        "_load_validated_run",
        lambda _path: (
            {
                "samples": [_sample("p", "abc")],
                "models": ["m"],
                "scoring_version": "1",
                "status": "complete",
            },
            [_record("m", "p", "abc")],
        ),
    )

    result = report_model.build_report(run_dir)

    assert result["policy"] is None
    assert result["conditions"]["baseline"]["status"] == "legacy_policy_unavailable"
    assert result["conditions"]["baseline"]["legacy_scoring_version"] == "1"


def test_direct_physical_pass_preserves_manifest_repetition_identity(tmp_path, monkeypatch):
    run_dir = tmp_path / "pass-2"
    run_dir.mkdir()
    physical_record = _record("m", "p", "abc")
    physical_record.pop("pass")
    monkeypatch.setattr(
        report_model,
        "_load_validated_run",
        lambda _path: (
            {
                "samples": [_sample("p", "abc")],
                "models": ["m"],
                "scoring_policy": {"version": 1},
                "evaluation_schema_version": 2,
                "planned_passes": 3,
                "repetition": 2,
                "status": "complete",
            },
            [physical_record],
        ),
    )

    result = report_model.build_report(run_dir)
    model = result["conditions"]["baseline"]["tracks"]["prose"]["models"][0]

    assert [row["pass"] for row in model["per_pass"]] == [1, 2, 3]
    assert model["per_pass"][0]["missing_count"] == 1
    assert model["per_pass"][1]["success_count"] == 1
    assert model["per_pass"][2]["missing_count"] == 1


def test_evaluation_identity_changes_with_uncertainty_policy():
    samples = [
        _sample("one", "abc", document_id="doc-1"),
        _sample("two", "abc", document_id="doc-2"),
    ]
    records = [_record("m", "one", "abc"), _record("m", "two", "abc")]
    first = report_model.evaluate_run(samples, records, bootstrap_replicates=10, bootstrap_seed=1)
    second = report_model.evaluate_run(samples, records, bootstrap_replicates=20, bootstrap_seed=1)
    third = report_model.evaluate_run(samples, records, bootstrap_replicates=10, bootstrap_seed=2)
    assert len({row["evaluation_id"] for row in (first, second, third)}) == 3


def test_imported_timings_without_measurement_profile_do_not_resolve_speed():
    samples = [_sample("one", "abc")]
    records = [_record("external:test", "one", "abc", latency_seconds=0.1)]
    result = report_model.evaluate_run(
        samples, records, manifest={"external_predictions": {"system": "test"}}
    )
    track = result["tracks"]["prose"]
    assert track["models"][0]["score"] == 100
    assert track["ranking"]["winner"] is None
    assert not track["speed_comparable"]
