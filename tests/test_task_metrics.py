from __future__ import annotations

import sys
import types
from contextlib import nullcontext

import pytest

from vlm_bench.task_metrics import (
    FORMULA_RENDERER_VERSION,
    TASK_METRICS_VERSION,
    aggregate_task_metrics,
    preflight_renderer,
    score_task,
)


def test_score_task_keeps_formula_renderer_disabled_without_optional_dependency():
    metrics = score_task("x^2", "x^2", {"content_type": "equation"})

    assert metrics["version"] == TASK_METRICS_VERSION
    assert metrics["scores"]["formula_render_similarity"] == {
        "status": "disabled",
        "value": None,
    }


def test_critical_expression_and_order_annotations_are_scored_independently():
    sample = {
        "metadata": {
            "annotations": {
                "critical_expressions": [r"x^2", r"= 0"],
                "reading_order": [[r"x^2", r"= 0"]],
            }
        }
    }

    correct = score_task(r"x^2 = 0", r"x^2 = 0", sample)
    wrong_order = score_task(r"= 0 then x^2", r"x^2 = 0", sample)

    assert correct["scores"]["critical_expression_accuracy"]["value"] == 1.0
    assert correct["scores"]["reading_order_accuracy"]["value"] == 1.0
    assert wrong_order["scores"]["critical_expression_accuracy"]["value"] == 1.0
    assert wrong_order["scores"]["reading_order_accuracy"]["value"] == 0.0


def test_annotation_coverage_and_invalid_reference_anchors_are_explicit():
    unannotated = score_task("answer", "answer", {})
    invalid_reference = score_task(
        "other", "x + y", {"annotations": {"critical_expressions": ["z"]}}
    )

    assert unannotated["scores"]["critical_expression_accuracy"]["status"] == "not_annotated"
    assert invalid_reference["scores"]["critical_expression_accuracy"]["status"] == "error"
    assert invalid_reference["errors"][0]["metric"] == "critical_expression_accuracy"


def test_critical_symbol_multiplicity_matches_reference():
    metrics = score_task(
        "a - b",
        "a - b - c",
        {"annotations": {"critical_expressions": ["-"]}},
    )

    critical = metrics["scores"]["critical_expression_accuracy"]
    assert critical["status"] == "scored"
    assert critical["value"] == 0.0
    assert critical["items"][0]["reference_occurrences"] == 2


def test_malformed_annotations_are_reported_per_metric():
    metrics = score_task("x", "x", {"metadata": {"annotations": ["bad"]}})

    assert metrics["scores"]["critical_expression_accuracy"]["status"] == "error"
    assert metrics["scores"]["reading_order_accuracy"]["status"] == "error"
    assert len(metrics["errors"]) == 2


def test_preflight_checks_exact_renderer_pin(monkeypatch: pytest.MonkeyPatch):
    class Parser:
        def parse(self, expression: str, *, dpi: int, prop: object) -> None:
            assert expression == "$x$"
            assert dpi == 96

    class MathTextParser:
        def __init__(self, output: str) -> None:
            assert output == "agg"

        def parse(self, *args: object, **kwargs: object) -> None:
            Parser().parse(*args, **kwargs)

    matplotlib = types.ModuleType("matplotlib")
    matplotlib.__version__ = FORMULA_RENDERER_VERSION
    matplotlib.__path__ = []
    matplotlib.rc_context = lambda _: nullcontext()
    font_manager = types.ModuleType("matplotlib.font_manager")
    font_manager.FontProperties = lambda **_: object()
    mathtext = types.ModuleType("matplotlib.mathtext")
    mathtext.MathTextParser = MathTextParser
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.font_manager", font_manager)
    monkeypatch.setitem(sys.modules, "matplotlib.mathtext", mathtext)

    assert preflight_renderer() == {
        "name": "matplotlib-mathtext",
        "version": FORMULA_RENDERER_VERSION,
        "task_metrics_version": TASK_METRICS_VERSION,
        "engine": "MathTextParser(agg), DejaVu Sans, 18 pt, 96 dpi",
    }

    matplotlib.__version__ = "3.10.2"
    with pytest.raises(RuntimeError, match="requires matplotlib==3.10.3"):
        preflight_renderer()


def test_research_aggregate_ignores_mismatched_versions_and_reports_coverage():
    samples = {
        "equation-1": {
            "id": "equation-1",
            "content_type": "equation",
            "metadata": {"annotations": {"critical_expressions": ["x^2"]}},
        },
        "equation-2": {"id": "equation-2", "content_type": "equation"},
    }
    records = [
        {
            "sample_id": "equation-1",
            "model": "model-a",
            "metrics": {
                "task_metrics": {
                    "version": TASK_METRICS_VERSION,
                    "scores": {
                        "formula_render_similarity": {"status": "disabled", "value": None},
                        "critical_expression_accuracy": {
                            "status": "scored",
                            "value": 1.0,
                            "passed": 1,
                            "total": 1,
                        },
                        "reading_order_accuracy": {"status": "not_annotated", "value": None},
                    },
                    "errors": [],
                }
            },
        },
        {
            "sample_id": "equation-2",
            "model": "model-a",
            "metrics": {"task_metrics": {"version": 999, "scores": {}}},
        },
    ]

    report = aggregate_task_metrics(records, samples, ["model-a"], formula_rendering_enabled=False)
    model = report["tracks"]["equation"]["models"][0]

    assert report["ranking"] is None
    assert model["version_mismatch_count"] == 1
    assert model["metrics"]["critical_expression_accuracy"]["coverage"] == 1.0
    assert model["metrics"]["critical_expression_accuracy"]["item_accuracy"] == 1.0
    assert model["metrics"]["formula_render_similarity"]["coverage"] is None


def test_pinned_renderer_smoke_if_formula_extra_is_installed():
    matplotlib = pytest.importorskip("matplotlib")
    if matplotlib.__version__ != FORMULA_RENDERER_VERSION:
        pytest.skip("optional formula-render extra is pinned separately")

    metrics = score_task(r"x^{2}", r"x^2", {"content_type": "equation"}, formula_rendering=True)

    formula = metrics["scores"]["formula_render_similarity"]
    assert formula["status"] == "scored", formula
    assert formula["value"] == pytest.approx(1.0)
