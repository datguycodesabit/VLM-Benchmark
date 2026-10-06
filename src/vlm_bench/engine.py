"""Provider-neutral execution over frozen benchmark inputs."""

import hashlib
import json
import math
import platform
import tempfile
import time
import uuid
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .metrics import score
from .runner import (
    PROMPT,
    _append,
    _atomic_json,
    _digest,
    _lock,
    _now,
    _response_timings,
    read_records,
)

MATH_PROMPT = "Transcribe only the visible handwritten equation as LaTeX. Preserve symbols and errors. Return only LaTeX without delimiters, explanations, or Markdown."


def _factory(provider, base_url, timeout, settings):
    from .backends import create_backend

    return create_backend(provider, base_url=base_url, timeout=timeout, settings=settings)


def _parse(selector):
    from .backends import parse_model

    return parse_model(selector)


def _identity(info):
    return {
        key: info.get(key)
        for key in ("digest", "revision", "checkpoint_sha256", "model", "device", "version")
    }


def _eligible(provider, sample):
    if provider != "trocr":
        return True
    return (sample.get("content_type") or "prose") in {"prose", "line", "prose-line", "word"} and (
        sample.get("sample_type") or "line"
    ) != "page"


SCORING_VERSION = "1"
TASK_METRICS_VERSION = 1
RESEARCH_PROTOCOL_VERSION = 1
PROMPT_VERSION = 1
_RESEARCH_PROTOCOLS = {"document-disjoint", "writer-disjoint"}


def _validate_research_options(strict_research, protocol):
    if not isinstance(strict_research, bool):
        raise ValueError("strict_research must be a boolean")
    if not isinstance(protocol, str) or protocol not in _RESEARCH_PROTOCOLS:
        raise ValueError("protocol must be document-disjoint or writer-disjoint")


def _research_summary(samples, strict_research, protocol):
    from .eligibility import validate_research

    return validate_research(samples, strict_research=strict_research, protocol=protocol)


def _validate_formula_rendering(value):
    if not isinstance(value, bool):
        raise ValueError("formula_rendering must be a boolean")


def _resolve_prompts(prose_prompt, math_prompt):
    prompts = {
        "prose": PROMPT if prose_prompt is None else prose_prompt,
        "math": MATH_PROMPT if math_prompt is None else math_prompt,
    }
    for name, prompt in prompts.items():
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{name}_prompt must be a non-empty string")
    return prompts["prose"], prompts["math"]


def _prompt_hashes(prose_prompt, math_prompt):
    prose_hash = hashlib.sha256(prose_prompt.encode()).hexdigest()
    math_hash = hashlib.sha256(math_prompt.encode()).hexdigest()
    return {"prose": prose_hash, "math": math_hash, "equation": math_hash}


def _validate_suite_metadata(suite_condition, repetition, repeated_measurement):
    if suite_condition is not None and (
        not isinstance(suite_condition, str)
        or not suite_condition.strip()
        or suite_condition != suite_condition.strip()
    ):
        raise ValueError("suite_condition must be a non-empty trimmed string")
    if repetition is not None and (
        isinstance(repetition, bool) or not isinstance(repetition, int) or repetition <= 0
    ):
        raise ValueError("repetition must be a positive integer")
    if not isinstance(repeated_measurement, bool):
        raise ValueError("repeated_measurement must be a boolean")
    if repeated_measurement and (suite_condition is None or repetition is None):
        raise ValueError("repeated_measurement requires suite_condition and repetition")


def _validate_manifest_prompts(manifest):
    prompt_version = manifest.get("prompt_version")
    if prompt_version is None:
        if manifest.get("prompt") != PROMPT or manifest.get("math_prompt") != MATH_PROMPT:
            raise ValueError("Prompt changed; create a new run")
        return PROMPT, MATH_PROMPT
    if (
        isinstance(prompt_version, bool)
        or not isinstance(prompt_version, int)
        or prompt_version != PROMPT_VERSION
    ):
        raise ValueError("Prompt format changed; use a compatible benchmark version")
    prose_prompt, math_prompt = manifest.get("prompt"), manifest.get("math_prompt")
    if (
        not isinstance(prose_prompt, str)
        or not prose_prompt.strip()
        or not isinstance(math_prompt, str)
        or not math_prompt.strip()
    ):
        raise ValueError("Run manifest must contain non-empty saved prompts")
    saved_hashes = manifest.get("prompt_hashes")
    if not isinstance(saved_hashes, dict) or saved_hashes != _prompt_hashes(
        prose_prompt, math_prompt
    ):
        raise ValueError("Saved prompt hashes do not match the run prompts")
    return prose_prompt, math_prompt


def _sample_prompt(manifest, sample):
    if sample.get("content_type") in {"equation", "math"}:
        return manifest.get("math_prompt", MATH_PROMPT)
    return manifest.get("prompt", PROMPT)


def _preflight_formula_renderer(enabled):
    if not enabled:
        return None
    from .task_metrics import preflight_renderer

    renderer = preflight_renderer()
    if (
        not isinstance(renderer, dict)
        or renderer.get("task_metrics_version") != TASK_METRICS_VERSION
    ):
        raise RuntimeError(
            "Formula renderer preflight returned an unsupported task-metrics version"
        )
    return renderer


_REQUEST_BUDGET_SCOPE = "generation_attempts_including_warmups_and_retries_excluding_metadata"


def _execution_controls(
    models,
    parsed,
    *,
    costs,
    max_retries,
    concurrency,
    max_requests,
    max_spend_usd,
    cache_dir,
):
    from .execution import CostMeter, SpendLedger, validate_execution_controls

    validate_execution_controls(
        max_retries=max_retries,
        concurrency=concurrency,
        max_requests=max_requests,
        max_spend_usd=max_spend_usd,
    )
    if concurrency > 1 and any(provider not in {"openai", "chatgpt"} for provider, _ in parsed):
        raise ValueError("concurrency greater than 1 is supported only for cloud providers")
    costs = costs if isinstance(costs, dict) else {}
    if max_spend_usd is not None:
        ledger = SpendLedger(max_spend_usd)
        for selector, (provider, _) in zip(models, parsed):
            meter = CostMeter(selector, costs, ledger, provider=provider)
            if not meter.applicable:
                raise ValueError(
                    f"max_spend_usd requires an applicable fixed request price or both input "
                    f"and output token prices for {selector}"
                )
    resolved_cache_dir = (
        str(Path(cache_dir).expanduser().resolve()) if cache_dir is not None else None
    )
    return {
        "max_retries": max_retries,
        "concurrency": concurrency,
        "max_requests": max_requests,
        "max_spend_usd": max_spend_usd,
        "cache_dir": resolved_cache_dir,
        "request_budget_scope": _REQUEST_BUDGET_SCOPE,
        "spend_limit_kind": "estimated_soft_limit" if max_spend_usd is not None else None,
    }


def _new_execution_state(directory, controls):
    path = directory / "execution.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError(f"Unsupported execution summary: {path}")
        invocations = state.get("invocations")
        if not isinstance(invocations, list):
            raise ValueError(f"Execution summary has no invocation history: {path}")
    else:
        state = {"schema_version": 1, "invocations": []}
    if state["invocations"] and state["invocations"][-1].get("status") == "running":
        state["invocations"][-1].update(status="interrupted", finished_at=_now())
    invocation = {
        "started_at": _now(),
        "finished_at": None,
        "status": "running",
        "generation_requests": 0,
        "estimated_spend_usd": 0.0,
        "unknown_spend_count": 0,
        "cache_hits": 0,
        "limit_reason": None,
        "execution_controls": dict(controls),
    }
    state["invocations"].append(invocation)
    return state, invocation


def _save_execution_state(directory, state, invocation):
    invocations = state["invocations"]
    state.update(
        generation_requests_current=invocation["generation_requests"],
        generation_requests_cumulative=sum(
            item.get("generation_requests", 0) for item in invocations
        ),
        estimated_spend_usd_current=invocation["estimated_spend_usd"],
        estimated_spend_usd_cumulative=sum(
            item.get("estimated_spend_usd", 0.0) for item in invocations
        ),
        unknown_spend_count_current=invocation["unknown_spend_count"],
        unknown_spend_count_cumulative=sum(
            item.get("unknown_spend_count", 0) for item in invocations
        ),
        cache_hits_current=invocation["cache_hits"],
        cache_hits_cumulative=sum(item.get("cache_hits", 0) for item in invocations),
        limit_reason=invocation.get("limit_reason"),
        execution_controls=invocation["execution_controls"],
    )
    _atomic_json(directory / "execution.json", state)


def _audit_source(data_dir, layout, protocol):
    """Audit all source samples before strict mode freezes only the test split."""
    from .dataset import check_dataset

    source = Path(data_dir) if data_dir is not None else Path("data")
    report = check_dataset(source, layout=layout or "auto")
    audit = _research_summary(report["samples"], False, protocol)
    if protocol == "writer-disjoint":
        missing_writer_ids = sorted(
            finding.get("sample_id", "<unknown>")
            for finding in audit.get("findings", [])
            if finding.get("type") == "missing_writer_id"
        )
        if missing_writer_ids:
            raise ValueError(
                "Strict research source audit found samples missing writer_id: "
                + ", ".join(missing_writer_ids)
            )
    leakage_types = ["document_split_overlap"]
    if protocol == "writer-disjoint":
        leakage_types.append("writer_split_overlap")
    overlaps = [
        finding for finding in audit.get("findings", []) if finding.get("type") in leakage_types
    ]
    if overlaps:
        identifiers = sorted(
            {sample_id for finding in overlaps for sample_id in finding.get("sample_ids", [])}
        )
        raise ValueError(
            f"Strict research source audit found {protocol} split leakage in samples: "
            + ", ".join(identifiers)
        )
    if not report["valid"]:
        raise ValueError(
            "Strict research source dataset audit failed; resolve dataset issues first"
        )
    audit["dataset_findings"] = report.get("findings", [])
    audit["duplicate_content"] = report.get("duplicate_content", {"images": [], "references": []})
    return audit


def _validate(models, settings, *, timeout, num_predict, seed):
    from .config import validate_settings

    if not isinstance(models, (list, tuple)) or not models:
        raise ValueError("Select at least one model")
    parsed = [_parse(m) for m in models]
    if len(set(parsed)) != len(parsed):
        raise ValueError("Duplicate model selectors after provider resolution")
    external = [model for provider, model in parsed if provider == "external"]
    if external:
        systems = ", ".join(external)
        raise ValueError(
            "External predictions cannot be run or previewed as inference; "
            f"use 'vlm-bench import' for external system(s): {systems}"
        )
    normalized_settings = validate_settings(models, settings if settings is not None else {})
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("Timeout must be a positive finite number")
    if isinstance(num_predict, bool) or not isinstance(num_predict, int) or num_predict <= 0:
        raise ValueError("Token limit must be a positive integer")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError("Seed must be an integer from 0 to 4294967295")
    return parsed, normalized_settings


def _validate_url(base_url):
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Ollama base URL must be an HTTP or HTTPS address")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Base URLs must not contain credentials, query parameters, or fragments")


def _controls(provider, options, settings, info):
    requested = dict(options, **settings)
    supported = {
        "ollama": {"num_predict", "temperature", "seed"},
        "trocr": {"num_predict", "num_beams", "device", "revision"},
        "openai": {"num_predict", "image_detail", "reasoning_effort"},
        "chatgpt": {"image_detail", "reasoning_effort"},
    }[provider]
    effective = {k: v for k, v in requested.items() if k in supported}
    if provider == "trocr":
        effective.update(
            device=info.get("device"),
            revision=info.get("revision"),
            num_beams=settings.get("num_beams", 1),
        )
        effective["num_predict"] = min(effective["num_predict"], 4096)
        effective.update(do_sample=False, processor_use_fast=False)
    elif provider in {"chatgpt", "openai"}:
        effective.setdefault("image_detail", "auto")
        if provider == "openai":
            effective["max_output_tokens"] = effective.pop("num_predict")
    return {
        "requested": requested,
        "effective": effective,
        "unsupported": sorted(set(requested) - supported),
    }


def _environment():
    dependencies = {}
    for name in (
        "Pillow",
        "httpx",
        "openpyxl",
        "torch",
        "transformers",
        "huggingface-hub",
        "keyring",
        "PyJWT",
    ):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "dependencies": dependencies,
    }


@contextmanager
def _inputs(data_dir, prepared, selection):
    from .snapshot import freeze, load

    if prepared is not None:
        if data_dir is not None or any(v is not None for v in selection.values()):
            raise ValueError(
                "--prepared cannot be combined with source data, selection, or preprocessing options"
            )
        path = Path(prepared).resolve()
        yield path, load(path)
        return
    if data_dir is None:
        data_dir = Path("data")
    options = {k: v for k, v in selection.items() if v is not None}
    with tempfile.TemporaryDirectory(prefix="vlm-inputs-") as folder:
        path = Path(folder) / "benchmark"
        freeze(Path(data_dir), path, **options)
        yield path, load(path)


def preview(
    data_dir=None,
    models=None,
    *,
    prepared=None,
    layout=None,
    preprocess=None,
    split=None,
    content_type=None,
    limit=None,
    seed=None,
    base_url="http://localhost:11434",
    timeout=300,
    num_predict=4096,
    settings=None,
    costs=None,
    max_retries=2,
    concurrency=1,
    max_requests=None,
    max_spend_usd=None,
    cache_dir=None,
    strict_research=False,
    protocol="document-disjoint",
    formula_rendering=False,
    prose_prompt=None,
    math_prompt=None,
    suite_condition=None,
    repetition=None,
    repeated_measurement=False,
    backend_factory=_factory,
):
    _validate_research_options(strict_research, protocol)
    _validate_formula_rendering(formula_rendering)
    _validate_suite_metadata(suite_condition, repetition, repeated_measurement)
    prose_prompt, math_prompt = _resolve_prompts(prose_prompt, math_prompt)
    parsed, settings = _validate(
        models, settings, timeout=timeout, num_predict=num_predict, seed=seed
    )
    execution_controls = _execution_controls(
        models,
        parsed,
        costs=costs,
        max_retries=max_retries,
        concurrency=concurrency,
        max_requests=max_requests,
        max_spend_usd=max_spend_usd,
        cache_dir=cache_dir,
    )
    _validate_url(base_url)
    if strict_research and prepared is None and split not in {None, "test"}:
        raise ValueError("Strict research runs require split='test'")
    source_audit = (
        _audit_source(data_dir, layout, protocol) if strict_research and prepared is None else None
    )
    selected_split = "test" if strict_research and prepared is None and split is None else split
    selection = dict(
        layout=layout,
        preprocess=preprocess,
        split=selected_split,
        content_type=content_type,
        limit=limit,
        seed=seed,
    )
    with _inputs(data_dir, prepared, selection) as (_, snapshot):
        samples = snapshot["samples"]
        research = _research_summary(samples, strict_research, protocol)
        formula_renderer = _preflight_formula_renderer(formula_rendering)
        choices = []
        options = {"num_predict": num_predict, "temperature": 0, "seed": snapshot.get("seed", 42)}
        for selector, (provider, name) in zip(models, parsed):
            configured = (settings or {}).get(selector, {})
            with backend_factory(provider, base_url, timeout, configured) as backend:
                info = backend.validate_model(name)
            choices.append(
                {
                    "selector": selector,
                    "provider": provider,
                    "execution": "cloud" if provider in {"openai", "chatgpt"} else "local",
                    "model_info": info,
                    "controls": _controls(provider, options, configured, info),
                    "eligible_samples": sum(_eligible(provider, s) for s in samples),
                    "unsupported_sample_ids": [
                        s["id"] for s in samples if not _eligible(provider, s)
                    ],
                }
            )
        return {
            "sample_count": len(samples),
            "sample_ids": [s["id"] for s in samples],
            "benchmark_fingerprint": snapshot["benchmark_fingerprint"],
            "verification_status_counts": dict(
                Counter(
                    s.get("verification_status")
                    or (s.get("metadata") or {}).get("verification_status")
                    or "unknown"
                    for s in samples
                )
            ),
            "layout": snapshot["layout"],
            "preprocess": snapshot["preprocess"],
            "scoring_version": SCORING_VERSION,
            "research_protocol_version": RESEARCH_PROTOCOL_VERSION,
            "strict_research": strict_research,
            "protocol": protocol,
            "formula_rendering": formula_rendering,
            "task_metrics_version": TASK_METRICS_VERSION,
            "formula_renderer": formula_renderer,
            "prompt_version": PROMPT_VERSION,
            "prompt": prose_prompt,
            "math_prompt": math_prompt,
            "prompt_hashes": _prompt_hashes(prose_prompt, math_prompt),
            "suite_condition": suite_condition,
            "repetition": repetition,
            "repeated_measurement": repeated_measurement,
            "execution_controls": execution_controls,
            "cache_enabled": cache_dir is not None,
            "research": research,
            "source_audit": source_audit
            if source_audit is not None
            else {
                "status": "unavailable",
                "reason": "Prepared snapshots contain only the frozen sample selection"
                if strict_research and prepared is not None
                else "Strict research source audit was not requested",
            },
            "validation_scope": "source-dataset-and-selected-test-samples"
            if source_audit is not None
            else "prepared-snapshot-only"
            if strict_research and prepared is not None
            else "selected-samples-only",
            "models": choices,
            "inference": False,
            "determinism_note": "The sampling seed freezes selection; it does not guarantee identical model predictions.",
        }


def run(
    data_dir=None,
    models=None,
    output_dir=Path("runs"),
    *,
    prepared=None,
    limit=None,
    seed=None,
    layout=None,
    preprocess=None,
    split=None,
    content_type=None,
    base_url="http://localhost:11434",
    timeout=300,
    num_predict=4096,
    warmup=True,
    settings=None,
    costs=None,
    max_retries=2,
    concurrency=1,
    max_requests=None,
    max_spend_usd=None,
    cache_dir=None,
    strict_research=False,
    protocol="document-disjoint",
    formula_rendering=False,
    prose_prompt=None,
    math_prompt=None,
    suite_condition=None,
    repetition=None,
    repeated_measurement=False,
    backend_factory=_factory,
    progress=print,
):
    from .snapshot import copy_inputs

    _validate_research_options(strict_research, protocol)
    _validate_formula_rendering(formula_rendering)
    _validate_suite_metadata(suite_condition, repetition, repeated_measurement)
    prose_prompt, math_prompt = _resolve_prompts(prose_prompt, math_prompt)
    parsed, settings = _validate(
        models, settings, timeout=timeout, num_predict=num_predict, seed=seed
    )
    execution_controls = _execution_controls(
        models,
        parsed,
        costs=costs,
        max_retries=max_retries,
        concurrency=concurrency,
        max_requests=max_requests,
        max_spend_usd=max_spend_usd,
        cache_dir=cache_dir,
    )
    _validate_url(base_url)
    if not isinstance(warmup, bool):
        raise ValueError("Warmup must be a boolean")
    if strict_research and prepared is None and split not in {None, "test"}:
        raise ValueError("Strict research runs require split='test'")
    source_audit = (
        _audit_source(data_dir, layout, protocol) if strict_research and prepared is None else None
    )
    selected_split = "test" if strict_research and prepared is None and split is None else split
    selection = dict(
        layout=layout,
        preprocess=preprocess,
        split=selected_split,
        content_type=content_type,
        limit=limit,
        seed=seed,
    )
    with _inputs(data_dir, prepared, selection) as (snapshot_path, snapshot):
        research = _research_summary(snapshot["samples"], strict_research, protocol)
        formula_renderer = _preflight_formula_renderer(formula_rendering)
        info = {}
        for selector, (provider, name) in zip(models, parsed):
            with backend_factory(
                provider, base_url, timeout, (settings or {}).get(selector, {})
            ) as backend:
                info[selector] = backend.validate_model(name)
        directory = Path(output_dir).resolve() / (
            datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
        directory.mkdir(parents=True)
        with _lock(directory):
            samples, copied = copy_inputs(snapshot_path, directory)
            options = {"num_predict": num_predict, "temperature": 0, "seed": copied.get("seed", 42)}
            manifest = {
                "schema_version": 2,
                "research_protocol_version": RESEARCH_PROTOCOL_VERSION,
                "benchmark_version": __version__,
                "created_at": _now(),
                "benchmark_fingerprint": copied["benchmark_fingerprint"],
                "scoring_version": SCORING_VERSION,
                "models": list(models),
                "model_info": info,
                "samples": samples,
                "data_dir": str(Path(data_dir).resolve()) if data_dir is not None else None,
                "prepared": str(Path(prepared).resolve()) if prepared is not None else None,
                "layout": copied["layout"],
                "preprocess": copied["preprocess"],
                "split": copied.get("split"),
                "strict_research": strict_research,
                "protocol": protocol,
                "formula_rendering": formula_rendering,
                "task_metrics_version": TASK_METRICS_VERSION,
                "formula_renderer": formula_renderer,
                "prompt_version": PROMPT_VERSION,
                "suite_condition": suite_condition,
                "repetition": repetition,
                "repeated_measurement": repeated_measurement,
                "research": research,
                "source_audit": source_audit
                if source_audit is not None
                else {
                    "status": "unavailable",
                    "reason": "Prepared snapshots contain only the frozen sample selection"
                    if strict_research and prepared is not None
                    else "Strict research source audit was not requested",
                },
                "validation_scope": "source-dataset-and-selected-test-samples"
                if source_audit is not None
                else "prepared-snapshot-only"
                if strict_research and prepared is not None
                else "selected-samples-only",
                "content_type": copied.get("content_type"),
                "limit": copied.get("limit"),
                "seed": copied.get("seed", 42),
                "base_url": base_url,
                "timeout": timeout,
                "prompt": prose_prompt,
                "math_prompt": math_prompt,
                "prompt_hashes": _prompt_hashes(prose_prompt, math_prompt),
                "options": options,
                "settings": settings or {},
                "costs": costs or {},
                "execution_controls": execution_controls,
                "max_retries": max_retries,
                "concurrency": concurrency,
                "max_requests": max_requests,
                "max_spend_usd": max_spend_usd,
                "cache_dir": execution_controls["cache_dir"],
                "warmup": warmup,
                "provider_controls": {
                    selector: _controls(
                        provider, options, (settings or {}).get(selector, {}), info[selector]
                    )
                    for selector, (provider, _) in zip(models, parsed)
                },
                "host": _environment(),
            }
            manifest["integrity"] = _digest(manifest)
            _atomic_json(directory / "manifest.json", manifest)
            progress(f"Run directory: {directory}")
            return _execute(directory, manifest, backend_factory, progress)


def resume(
    directory,
    *,
    retry_failed=False,
    max_requests=None,
    max_spend_usd=None,
    backend_factory=_factory,
    progress=print,
):
    directory = directory.resolve()
    with _lock(directory):
        manifest = json.loads((directory / "manifest.json").read_text())
        immutable = {
            k: v for k, v in manifest.items() if k not in {"integrity", "status", "updated_at"}
        }
        if _digest(immutable) != manifest["integrity"]:
            raise ValueError("Run manifest changed; create a new run")
        if any(_parse(selector)[0] == "external" for selector in manifest.get("models", [])):
            raise ValueError(
                "Imported external-prediction runs cannot be resumed; "
                "use 'vlm-bench import' to create an imported run"
            )
        if manifest.get("scoring_version", SCORING_VERSION) != SCORING_VERSION:
            raise ValueError("Scoring rules changed; create a new run instead of mixing scores")
        _validate_formula_rendering(manifest.get("formula_rendering", False))
        if manifest.get("task_metrics_version", TASK_METRICS_VERSION) != TASK_METRICS_VERSION:
            raise ValueError(
                "Task scoring rules changed; create a new run instead of mixing scores"
            )
        if manifest.get("benchmark_fingerprint"):
            from .snapshot import fingerprint

            if (
                fingerprint(manifest["samples"], manifest["preprocess"])
                != manifest["benchmark_fingerprint"]
            ):
                raise ValueError("Frozen benchmark fingerprint does not match the run manifest")
        _validate_manifest_prompts(manifest)
        # Resume consumes saved inputs, not a newly sampled live dataset.
        for sample in manifest["samples"]:
            path = Path(sample["crop_path"])
            if (
                not path.exists()
                or hashlib.sha256(path.read_bytes()).hexdigest() != sample["hashes"]["crop"]
            ):
                raise ValueError(f"Saved image changed: {sample['id']}")
        formula_renderer = _preflight_formula_renderer(manifest.get("formula_rendering", False))
        saved_renderer = manifest.get("formula_renderer")
        if saved_renderer is not None and saved_renderer != formula_renderer:
            raise ValueError("Formula renderer changed; create a new run")
        saved_controls = manifest.get("execution_controls") or {}
        if not isinstance(saved_controls, dict):
            raise ValueError("Run manifest execution_controls must be a JSON object")
        saved_max_requests = saved_controls.get("max_requests", manifest.get("max_requests"))
        saved_max_spend = saved_controls.get("max_spend_usd", manifest.get("max_spend_usd"))
        effective_max_requests = max_requests if max_requests is not None else saved_max_requests
        effective_max_spend = max_spend_usd if max_spend_usd is not None else saved_max_spend
        parsed = [_parse(selector) for selector in manifest["models"]]
        invocation_controls = _execution_controls(
            manifest["models"],
            parsed,
            costs=manifest.get("costs", {}),
            max_retries=saved_controls.get("max_retries", manifest.get("max_retries", 2)),
            concurrency=saved_controls.get("concurrency", manifest.get("concurrency", 1)),
            max_requests=effective_max_requests,
            max_spend_usd=effective_max_spend,
            cache_dir=saved_controls.get("cache_dir", manifest.get("cache_dir")),
        )
        if retry_failed:
            rows = read_records(directory / "results.jsonl", repair=True)
            failures = [r for r in rows if r["status"] not in {"success", "unsupported"}]
            if failures:
                for row in failures:
                    _append(directory / "attempts.jsonl", row)
                temporary = directory / "results.retry.tmp"
                temporary.write_text(
                    "".join(
                        json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r not in failures
                    )
                )
                temporary.replace(directory / "results.jsonl")
        return _execute(
            directory,
            manifest,
            backend_factory,
            progress,
            verify=True,
            invocation_controls=invocation_controls,
        )


def _execute(
    directory,
    manifest,
    backend_factory,
    progress,
    verify=False,
    invocation_controls=None,
):
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    from .backends import BackendPaused
    from .execution import CostMeter, RetryableProviderError, SpendLedger, retry_delay

    rows_path = directory / "results.jsonl"
    attempts_path = directory / "attempts.jsonl"
    existing = read_records(rows_path, repair=True)
    warmup_history = read_records(directory / "warmups.jsonl", repair=True)
    warmed_models = {row.get("model") for row in warmup_history if row.get("status") == "success"}
    expected = {
        (model, sample["id"]) for model in manifest["models"] for sample in manifest["samples"]
    }
    completed = {(row["model"], row["sample_id"]) for row in existing}
    if len(completed) != len(existing) or not completed.issubset(expected):
        raise ValueError("Duplicate or unknown result pairs")
    rows_path.touch(exist_ok=True)
    controls = dict(
        invocation_controls
        or manifest.get("execution_controls")
        or {
            "max_retries": manifest.get("max_retries", 2),
            "concurrency": manifest.get("concurrency", 1),
            "max_requests": manifest.get("max_requests"),
            "max_spend_usd": manifest.get("max_spend_usd"),
            "cache_dir": manifest.get("cache_dir"),
            "request_budget_scope": _REQUEST_BUDGET_SCOPE,
        }
    )
    # Schema-v2 runs created before managed execution keep their original
    # controls and receive the conservative defaults for the new runtime.
    controls.setdefault("max_retries", 2)
    controls.setdefault("concurrency", 1)
    controls.setdefault("max_requests", None)
    controls.setdefault("max_spend_usd", None)
    controls.setdefault("cache_dir", None)
    controls.setdefault("request_budget_scope", _REQUEST_BUDGET_SCOPE)
    controls.setdefault(
        "spend_limit_kind",
        "estimated_soft_limit" if controls["max_spend_usd"] is not None else None,
    )
    state, invocation = _new_execution_state(directory, controls)
    ledger = SpendLedger(controls["max_spend_usd"])
    cache = None
    if controls["cache_dir"] is not None:
        from .cache import ResponseCache

        cache = ResponseCache(controls["cache_dir"])

    def persist_runtime():
        invocation["estimated_spend_usd"] = ledger.estimated_usd
        invocation["unknown_spend_count"] = ledger.unknown_spend_count
        _save_execution_state(directory, state, invocation)

    def log_start(selector, sample, attempt, *, warmup=False, reserved_usd=0.0):
        invocation["generation_requests"] += 1
        event = {
            "event_kind": "request_started",
            "model": selector,
            "sample_id": sample["id"],
            "attempt": attempt,
            "warmup": warmup,
            "timestamp": _now(),
            "request_budget_scope": _REQUEST_BUDGET_SCOPE,
        }
        _append(attempts_path, event)
        persist_runtime()
        return reserved_usd

    def log_finish(
        selector,
        sample,
        attempt,
        status,
        elapsed,
        *,
        error=None,
        category=None,
        retry_after=None,
        estimate=None,
        warmup=False,
    ):
        event = {
            "event_kind": "request_finished",
            "model": selector,
            "sample_id": sample["id"],
            "attempt": attempt,
            "warmup": warmup,
            "status": status,
            "latency_seconds": elapsed,
            "estimated_spend_usd": estimate,
            "timestamp": _now(),
        }
        if error is not None:
            event["error"] = str(error)
        if category is not None:
            event["error_category"] = category
        if retry_after is not None:
            event["retry_after_seconds"] = retry_after
        _append(attempts_path, event)
        persist_runtime()

    def blocked_reason(meter):
        if controls["max_requests"] is not None and (
            invocation["generation_requests"] >= controls["max_requests"]
        ):
            return "max_requests"
        if controls["max_spend_usd"] is not None:
            if ledger.unknown_spend_count:
                return "usage_unavailable"
            if not meter.can_schedule():
                return "max_spend_usd"
        return None

    def write_result(row, state_item=None):
        _append(rows_path, row)
        completed.add((row["model"], row["sample_id"]))
        if state_item is not None:
            state_item["_result_written"] = True
        if row.get("cache_hit") is True:
            invocation["cache_hits"] += 1
            persist_runtime()
        completed_this_invocation[0] += 1
        elapsed_invocation = max(0.001, time.perf_counter() - invocation_started)
        finished = len(completed)
        remaining = max(0, len(expected) - finished)
        eta = elapsed_invocation / max(1, completed_this_invocation[0]) * remaining
        progress(
            f"[{finished}/{len(expected)}] {row['model']} {row['sample_id']}: {row['status']} "
            f"(elapsed {elapsed_invocation:.1f}s, ETA {eta:.1f}s)"
        )

    def base_row(selector, provider, sample):
        return {
            "model": selector,
            "provider": provider,
            "sample_id": sample["id"],
            "reference": sample["reference"],
            "reference_source": sample["reference_source"],
            "content_type": sample.get("content_type") or "prose",
            "source_document": sample.get("source_document"),
            "prediction": None,
            "metrics": None,
            "error": None,
            "timestamp": _now(),
            "status": "error",
            "truncated": False,
        }

    def score_response(
        row,
        raw,
        sample,
        provider,
        name,
        elapsed,
        *,
        cache_hit=False,
        cache_lookup_seconds=None,
        cache_bypass_reason=None,
    ):
        prediction = raw.get("message", {}).get("content")
        if not isinstance(prediction, str):
            raise ValueError("Backend response must contain message.content text")
        row.update(
            raw_response=raw,
            prediction=prediction,
            usage=None if cache_hit else raw.get("usage"),
            usage_source="cache" if cache_hit else "provider",
            actual_model=raw.get("model", name),
            actual_device=(raw.get("provider_details") or {}).get("device"),
            cache_hit=cache_hit,
            cache_lookup_seconds=cache_lookup_seconds,
            cache_bypass_reason=cache_bypass_reason,
        )
        row.update(
            _response_timings(
                raw, float("inf") if provider == "chatgpt" else manifest["options"]["num_predict"]
            )
        )
        math_task = sample.get("content_type") in {"equation", "math"}
        if math_task:
            from .research import score_equation

            row["metrics"] = score_equation(prediction, row["reference"])
        else:
            row["metrics"] = score(prediction, row["reference"])
        from .task_metrics import score_task

        row["metrics"]["task_metrics"] = score_task(
            prediction,
            row["reference"],
            sample,
            formula_rendering=manifest.get("formula_rendering", False),
        )
        row["status"] = "success"
        if cache_hit:
            row["latency_seconds"] = None
            row["inference_latency_seconds"] = None
            row["usage"] = None
        else:
            row["latency_seconds"] = elapsed
            load = row.get("load_duration_seconds")
            if isinstance(load, (int, float)) and not isinstance(load, bool):
                row["inference_latency_seconds"] = max(0, elapsed - load)
            elif provider in {"openai", "chatgpt"}:
                row["inference_latency_seconds"] = elapsed
            else:
                row["inference_latency_seconds"] = None
        return row

    def invoke(backend, name, sample, prompt, options):
        method = getattr(backend, "transcribe_once", None)
        if callable(method):
            return method(name, Path(sample["crop_path"]), prompt, options)
        return backend.transcribe(name, Path(sample["crop_path"]), prompt, options)

    def invoke_timed(backend, name, sample, prompt, options):
        started = time.perf_counter()
        try:
            return (
                invoke(backend, name, sample, prompt, options),
                None,
                time.perf_counter() - started,
            )
        except BaseException as exc:
            return None, exc, time.perf_counter() - started

    manifest["status"] = "running"
    _atomic_json(directory / "manifest.json", manifest)
    invocation_started = time.perf_counter()
    completed_this_invocation = [0]
    paused = False
    paused_providers = set()
    try:
        for selector in manifest["models"]:
            provider, name = _parse(selector)
            if provider in paused_providers:
                continue
            pending = [
                sample
                for sample in manifest["samples"]
                if (selector, sample["id"]) not in completed
            ]
            if not pending:
                continue
            settings = dict(manifest.get("settings", {}).get(selector, {}))
            info = manifest["model_info"][selector]
            selector_paused = False
            if provider == "trocr" and info.get("revision"):
                settings["revision"] = info["revision"]
            meter = CostMeter(selector, manifest.get("costs", {}), ledger, provider=provider)
            with ExitStack() as stack:
                backend = stack.enter_context(
                    backend_factory(provider, manifest["base_url"], manifest["timeout"], settings)
                )
                stack.callback(backend.unload, name)
                current = backend.validate_model(name)
                if _identity(current) != _identity(info):
                    raise ValueError(f"Model identity changed: {selector}; create a new run")
                options = dict(manifest["options"], **settings)
                eligible = [sample for sample in pending if _eligible(provider, sample)]

                if (
                    eligible
                    and manifest.get("warmup")
                    and provider in {"ollama", "trocr"}
                    and selector not in warmed_models
                ):
                    warmup_sample = eligible[0]
                    warmup_result = {
                        "model": selector,
                        "sample_id": warmup_sample["id"],
                        "status": "skipped",
                    }
                    started_at = time.perf_counter()
                    for attempt in range(1, controls["max_retries"] + 2):
                        reason = blocked_reason(meter)
                        if reason:
                            invocation["limit_reason"] = reason
                            warmup_result["limit_reason"] = reason
                            break
                        reserved = meter.reserve()
                        log_start(
                            selector, warmup_sample, attempt, warmup=True, reserved_usd=reserved
                        )
                        try:
                            raw = invoke(
                                backend,
                                name,
                                warmup_sample,
                                _sample_prompt(manifest, warmup_sample),
                                options,
                            )
                        except RetryableProviderError as exc:
                            estimate = meter.finish(
                                None,
                                retryable_rejection=exc.category
                                in {"connection_error", "rate_limited"},
                                reserved_usd=reserved,
                            )
                            delay = exc.retry_after_seconds
                            log_finish(
                                selector,
                                warmup_sample,
                                attempt,
                                "retryable_error",
                                time.perf_counter() - started_at,
                                error=exc,
                                category=exc.category,
                                retry_after=delay,
                                estimate=estimate,
                                warmup=True,
                            )
                            warmup_result.update(status="error", error=str(exc))
                            if attempt <= controls["max_retries"]:
                                delay = retry_delay(exc, attempt)
                                if delay:
                                    time.sleep(delay)
                                continue
                            break
                        except BackendPaused as exc:
                            meter.finish(None, reserved_usd=reserved)
                            log_finish(
                                selector,
                                warmup_sample,
                                attempt,
                                "paused",
                                time.perf_counter() - started_at,
                                error=exc,
                                warmup=True,
                            )
                            paused = True
                            selector_paused = True
                            paused_providers.add(provider)
                            break
                        except Exception as exc:
                            estimate = meter.finish(None, reserved_usd=reserved)
                            log_finish(
                                selector,
                                warmup_sample,
                                attempt,
                                "error",
                                time.perf_counter() - started_at,
                                error=exc,
                                estimate=estimate,
                                warmup=True,
                            )
                            warmup_result.update(status="error", error=str(exc))
                            break
                        else:
                            estimate = meter.finish(raw, reserved_usd=reserved)
                            log_finish(
                                selector,
                                warmup_sample,
                                attempt,
                                "success",
                                time.perf_counter() - started_at,
                                estimate=estimate,
                                warmup=True,
                            )
                            warmup_result.update(status="success", response=raw)
                            break
                    warmup_result["latency_seconds"] = time.perf_counter() - started_at
                    _append(directory / "warmups.jsonl", warmup_result)
                    if warmup_result["status"] == "success":
                        warmed_models.add(selector)
                    if selector_paused or invocation.get("limit_reason"):
                        continue

                queue = []
                cache_info = manifest.get("provider_controls", {}).get(selector, {})
                effective_controls = cache_info.get("effective", options)
                for sample in pending:
                    if not _eligible(provider, sample):
                        row = base_row(selector, provider, sample)
                        row.update(
                            status="unsupported", error="TrOCR requires a single prose-line image"
                        )
                        write_result(row)
                        continue
                    prompt = _sample_prompt(manifest, sample)
                    bypass_reason = None
                    lookup_seconds = None
                    if cache is not None:
                        from .cache import cache_key

                        key, bypass_reason = cache_key(
                            sample,
                            selector,
                            info,
                            prompt,
                            effective_controls,
                            repeated_measurement=manifest.get("repeated_measurement", False),
                        )
                        if key is not None:
                            lookup_started = time.perf_counter()
                            cached = cache.lookup(key)
                            lookup_seconds = time.perf_counter() - lookup_started
                            if cached is not None:
                                row = base_row(selector, provider, sample)
                                row = score_response(
                                    row,
                                    cached,
                                    sample,
                                    provider,
                                    name,
                                    0.0,
                                    cache_hit=True,
                                    cache_lookup_seconds=lookup_seconds,
                                )
                                row["cache_key"] = key
                                write_result(row)
                                continue
                    queue.append(
                        {
                            "sample": sample,
                            "attempt": 1,
                            "cache_key": key if cache is not None else None,
                            "cache_bypass_reason": bypass_reason,
                            "cache_lookup_seconds": lookup_seconds,
                        }
                    )

                if not queue or selector_paused or invocation.get("limit_reason"):
                    continue

                def finish_state(state_item, raw, error, elapsed, *, retry_allowed=True):
                    sample = state_item["sample"]
                    attempt = state_item["attempt"]
                    if state_item.get("_result_written"):
                        return "finished"
                    already_finalized = state_item.get("_attempt_finalized", False)
                    if already_finalized:
                        if (
                            isinstance(error, RetryableProviderError)
                            and attempt <= controls["max_retries"]
                        ):
                            return "retry"
                    elif isinstance(error, RetryableProviderError):
                        estimate = meter.finish(
                            None,
                            retryable_rejection=error.category
                            in {"connection_error", "rate_limited"},
                            reserved_usd=state_item["reserved"],
                        )
                        log_finish(
                            selector,
                            sample,
                            attempt,
                            "retryable_error",
                            elapsed,
                            error=error,
                            category=error.category,
                            retry_after=error.retry_after_seconds,
                            estimate=estimate,
                        )
                        state_item["_attempt_finalized"] = True
                        state_item["_raw_response"] = raw
                        state_item["_error"] = error
                        state_item["_elapsed"] = elapsed
                        if retry_allowed and attempt <= controls["max_retries"]:
                            delay = retry_delay(error, attempt)
                            state_item["_attempt_finalized"] = True
                            state_item["_raw_response"] = raw
                            state_item["_error"] = error
                            state_item["_elapsed"] = elapsed
                            if delay:
                                time.sleep(delay)
                            state_item["attempt"] += 1
                            state_item["_attempt_finalized"] = False
                            return "retry"
                    elif isinstance(error, BackendPaused):
                        meter.finish(None, reserved_usd=state_item["reserved"])
                        log_finish(selector, sample, attempt, "paused", elapsed, error=error)
                        state_item["_attempt_finalized"] = True
                        state_item["_raw_response"] = raw
                        state_item["_error"] = error
                        state_item["_elapsed"] = elapsed
                        paused_providers.add(provider)
                        return "paused"
                    elif error is not None:
                        estimate = meter.finish(None, reserved_usd=state_item["reserved"])
                        log_finish(
                            selector,
                            sample,
                            attempt,
                            "error",
                            elapsed,
                            error=error,
                            estimate=estimate,
                        )
                        state_item["_attempt_finalized"] = True
                        state_item["_raw_response"] = raw
                        state_item["_error"] = error
                        state_item["_elapsed"] = elapsed
                    else:
                        estimate = meter.finish(raw, reserved_usd=state_item["reserved"])
                        log_finish(selector, sample, attempt, "success", elapsed, estimate=estimate)
                        state_item["_attempt_finalized"] = True
                        state_item["_raw_response"] = raw
                        state_item["_error"] = error
                        state_item["_elapsed"] = elapsed

                    if error is None:
                        row = base_row(selector, provider, sample)
                        cache_write_error = None
                        try:
                            row = score_response(
                                row,
                                raw,
                                sample,
                                provider,
                                name,
                                elapsed,
                                cache_bypass_reason=state_item["cache_bypass_reason"],
                                cache_lookup_seconds=state_item["cache_lookup_seconds"],
                            )
                            if row.get("truncated"):
                                row["cache_bypass_reason"] = (
                                    "truncated_response"
                                    if cache is not None and state_item["cache_key"] is not None
                                    else state_item["cache_bypass_reason"]
                                )
                        except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
                            row.update(status="error", error=str(exc))
                            row["latency_seconds"] = elapsed
                        if (
                            row["status"] == "success"
                            and not row.get("truncated")
                            and cache is not None
                            and state_item["cache_key"] is not None
                        ):
                            try:
                                cache.store(state_item["cache_key"], raw)
                            except (OSError, ValueError) as exc:
                                row["cache_error"] = str(exc)
                                cache_write_error = exc
                            else:
                                row["cache_key"] = state_item["cache_key"]
                        write_result(row, state_item)
                        if cache_write_error is not None:
                            raise RuntimeError(
                                "Prediction was saved, but writing its cache entry failed: "
                                f"{cache_write_error}"
                            ) from cache_write_error
                    elif not isinstance(error, BackendPaused):
                        row = base_row(selector, provider, sample)
                        row.update(error=str(error), latency_seconds=elapsed)
                        write_result(row, state_item)
                    if controls["max_spend_usd"] is not None and ledger.unknown_spend_count:
                        invocation["limit_reason"] = "usage_unavailable"
                    state_item["_attempt_finalized"] = True
                    state_item["_raw_response"] = raw
                    state_item["_error"] = error
                    state_item["_elapsed"] = elapsed
                    return "paused" if isinstance(error, BackendPaused) else "finished"

                if controls["concurrency"] == 1:
                    while queue and not selector_paused and not invocation.get("limit_reason"):
                        state_item = queue.pop(0)
                        sample = state_item["sample"]
                        reason = blocked_reason(meter)
                        if reason:
                            invocation["limit_reason"] = reason
                            break
                        state_item["reserved"] = meter.reserve()
                        log_start(
                            selector,
                            sample,
                            state_item["attempt"],
                            reserved_usd=state_item["reserved"],
                        )
                        started = time.perf_counter()
                        try:
                            raw = invoke(
                                backend, name, sample, _sample_prompt(manifest, sample), options
                            )
                            error = None
                        except (RetryableProviderError, BackendPaused) as exc:
                            raw, error = None, exc
                        except Exception as exc:
                            raw, error = None, exc
                        action = finish_state(state_item, raw, error, time.perf_counter() - started)
                        if action == "retry":
                            queue.append(state_item)
                        elif action == "paused":
                            paused = True
                            selector_paused = True
                else:
                    futures = {}
                    executor = ThreadPoolExecutor(max_workers=controls["concurrency"])

                    def schedule_one(state_item):
                        sample = state_item["sample"]
                        state_item["reserved"] = meter.reserve()
                        log_start(
                            selector,
                            sample,
                            state_item["attempt"],
                            reserved_usd=state_item["reserved"],
                        )
                        future = executor.submit(
                            invoke_timed,
                            backend,
                            name,
                            sample,
                            _sample_prompt(manifest, sample),
                            options,
                        )
                        futures[future] = state_item

                    try:
                        while queue or futures:
                            while (
                                queue
                                and len(futures) < controls["concurrency"]
                                and not selector_paused
                                and not invocation.get("limit_reason")
                            ):
                                reason = blocked_reason(meter)
                                if reason:
                                    invocation["limit_reason"] = reason
                                    break
                                schedule_one(queue.pop(0))
                            if not futures:
                                break
                            done, _ = wait(futures, return_when=FIRST_COMPLETED)
                            for future in done:
                                state_item = futures[future]
                                raw, error, elapsed = future.result()
                                state_item["_raw_response"] = raw
                                state_item["_error"] = error
                                state_item["_elapsed"] = elapsed
                                action = finish_state(
                                    state_item,
                                    raw,
                                    error,
                                    state_item["_elapsed"],
                                    retry_allowed=not invocation.get("limit_reason"),
                                )
                                futures.pop(future)
                                if action == "retry":
                                    queue.append(state_item)
                                elif action == "paused":
                                    paused = True
                                    selector_paused = True
                                if isinstance(error, KeyboardInterrupt):
                                    raise error
                    except BaseException as scheduler_error:
                        drain_errors = []
                        for future, state_item in list(futures.items()):
                            if future.cancel():
                                meter.finish(
                                    None,
                                    retryable_rejection=True,
                                    reserved_usd=state_item["reserved"],
                                )
                                invocation["generation_requests"] -= 1
                                _append(
                                    attempts_path,
                                    {
                                        "event_kind": "request_finished",
                                        "model": selector,
                                        "sample_id": state_item["sample"]["id"],
                                        "attempt": state_item["attempt"],
                                        "status": "cancelled_before_start",
                                        "timestamp": _now(),
                                    },
                                )
                                futures.pop(future)
                        for future, state_item in list(futures.items()):
                            try:
                                raw, error, elapsed = future.result()
                            except BaseException as exc:
                                raw, error, elapsed = None, exc, 0.0
                            state_item["_raw_response"] = raw
                            state_item["_error"] = error
                            state_item["_elapsed"] = elapsed
                            try:
                                finish_state(
                                    state_item,
                                    raw,
                                    error,
                                    state_item["_elapsed"],
                                    retry_allowed=False,
                                )
                            except BaseException as exc:
                                # Preserve other completed responses; the
                                # original failure remains the primary error.
                                drain_errors.append(exc)
                            futures.pop(future)
                        executor.shutdown(wait=True, cancel_futures=True)
                        persist_runtime()
                        for drain_error in drain_errors:
                            scheduler_error.add_note(
                                "A concurrently completed request also failed during drain: "
                                f"{type(drain_error).__name__}: {drain_error}"
                            )
                        raise scheduler_error
                    finally:
                        executor.shutdown(wait=True, cancel_futures=True)

                if invocation.get("limit_reason"):
                    break
    except BaseException:
        manifest.update(status="interrupted", updated_at=_now())
        invocation.update(status="interrupted", finished_at=_now())
        _atomic_json(directory / "manifest.json", manifest)
        persist_runtime()
        raise

    rows = read_records(rows_path)
    final_status = (
        "paused"
        if paused or invocation.get("limit_reason")
        else "complete"
        if all(row["status"] in {"success", "unsupported"} for row in rows)
        else "complete_with_errors"
    )
    manifest.update(status=final_status, updated_at=_now())
    invocation.update(status=final_status, finished_at=_now())
    _atomic_json(directory / "manifest.json", manifest)
    persist_runtime()
    return directory
