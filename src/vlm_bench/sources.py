"""Read-only discovery of portable run, experiment, and suite members."""

from __future__ import annotations

import json
from pathlib import Path


def _object(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read benchmark source: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark source must be an object: {path}")
    return value


def member_path(root, value):
    """Resolve members within the source, including moved legacy absolute paths."""
    if not isinstance(value, str) or not value:
        raise ValueError("Saved run path must be a non-empty string")
    root = Path(root).resolve()
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        path = (root / raw).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Saved run path escapes the benchmark source")
    else:
        path = raw.resolve()
        if not path.is_relative_to(root):
            matches = [
                item.parent.resolve()
                for item in root.rglob("manifest.json")
                if item.parent.name == raw.name and item.parent.resolve().is_relative_to(root)
            ]
            if len(matches) != 1:
                raise ValueError(f"Saved run cannot be resolved inside the benchmark source: {raw}")
            path = matches[0]
    if not (path / "manifest.json").is_file():
        raise ValueError(f"Saved run manifest is missing: {path}")
    return path


def source_plan(source):
    """Return marker metadata and ordered condition/pass/run entries without inference."""
    root = Path(source).expanduser().resolve()
    if root.is_file():
        root = root.parent
    if (root / "suite.json").is_file():
        from .suite import _read_manifest

        manifest = _read_manifest(root / "suite.json")
        kind = "suite"
        conditions = manifest.get("conditions", [])
        entries = list(manifest.get("runs", {}).values())
        planned = {item["name"]: item["repeats"] for item in conditions}
    elif (root / "experiment.json").is_file():
        from .experiments import _state_for

        _, manifest = _state_for(root)
        kind = "experiment"
        planned = {"baseline": manifest["planned_passes"]}
        entries = [
            {"condition": "baseline", "repetition": index, "run_dir": value}
            for index, value in enumerate(manifest["runs"], 1)
        ]
    else:
        manifest = _object(root / "manifest.json")
        kind = "run"
        planned = {"baseline": manifest.get("planned_passes", 1)}
        entries = [
            {
                "condition": "baseline",
                "repetition": manifest.get("repetition") or 1,
                "run_dir": str(root),
            }
        ]
    for name, count in planned.items():
        if type(count) is not int or count < 1:
            raise ValueError(f"Invalid planned pass count for condition {name!r}")
    resolved, seen, paths = [], set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Saved run entries must be objects")
        condition, repetition = entry.get("condition"), entry.get("repetition")
        if (
            condition not in planned
            or type(repetition) is not int
            or not 1 <= repetition <= planned[condition]
        ):
            raise ValueError("Saved condition or pass identity does not match the source plan")
        key = (condition, repetition)
        if key in seen:
            raise ValueError("Source contains duplicate condition/pass identities")
        seen.add(key)
        if not entry.get("run_dir"):
            continue
        path = member_path(root, entry["run_dir"])
        if path in paths:
            raise ValueError("Source contains duplicate run paths")
        paths.add(path)
        resolved.append({"condition": condition, "repetition": repetition, "path": path})
    return {"root": root, "kind": kind, "manifest": manifest, "planned": planned, "runs": resolved}
