"""Generate editable, validated experiment configurations without touching credentials."""

from __future__ import annotations

import json
import os
from pathlib import Path


def create_config(output, *, prepared, models, repeats=3):
    """Create a configuration exclusively; preserve any existing user configuration."""
    from .config import load_config

    output = Path(output).expanduser().resolve()
    if not models:
        raise ValueError("Select at least one model")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared = Path(prepared).expanduser().resolve()
    relative = os.path.relpath(prepared, output.parent)
    text = (
        "# VLM Benchmark: edit model selectors and review the frozen references.\n"
        "version = 3\n\n"
        "[experiment]\n"
        f"prepared = {json.dumps(relative)}\n"
        "warmup = true\n"
        "strict_research = false\n"
        'protocol = "document-disjoint"\n\n'
        "[suite]\n"
        f"repeats = {repeats}\n\n"
        "[[suite.conditions]]\n"
        'name = "baseline"\n'
        f"models = {json.dumps(list(models))}\n"
    )
    # Validation runs on an adjacent temporary file so relative paths round-trip.
    import tempfile

    descriptor, name = tempfile.mkstemp(prefix=".vlm-init-", suffix=".toml", dir=output.parent)
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        load_config(staged)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    finally:
        staged.unlink(missing_ok=True)
    return output
