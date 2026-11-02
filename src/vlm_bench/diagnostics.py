"""Structured provider readiness checks that never perform inference."""

from __future__ import annotations

import importlib.util
import os
import platform
from importlib.metadata import PackageNotFoundError, version


def diagnose(provider, *, models=(), base_url="http://localhost:11434", device="auto"):
    from .backends import create_backend, parse_model

    dependencies = {}
    for name in ("torch", "transformers", "keyring", "jwt"):
        dependencies[name] = importlib.util.find_spec(name) is not None
    checks = []
    result = {
        "provider": provider,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dependencies": dependencies,
        "openai_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "checks": checks,
    }
    result["dependency_versions"] = {}
    for name in ("torch", "transformers", "httpx", "Pillow", "openpyxl"):
        try:
            result["dependency_versions"][name] = version(name)
        except PackageNotFoundError:
            result["dependency_versions"][name] = None
    if provider in {"trocr", "python"}:
        try:
            from .devices import resolve_device

            resolution = resolve_device(device)
            result["runtime"] = resolution.to_dict()
        except (RuntimeError, ValueError, ImportError) as exc:
            checks.append({"name": "device", "ok": False, "detail": str(exc)})
            result["ready"] = False
            return result
    try:
        settings = {"device": device} if provider in {"trocr", "python"} else {}
        with create_backend(provider, base_url=base_url, timeout=10, settings=settings) as backend:
            if models:
                for selector in models:
                    selected_provider, model = parse_model(selector)
                    # Bare local checkpoint names are accepted for doctor --provider trocr.
                    if ":" in selector and selected_provider != provider:
                        raise ValueError(f"Model {selector!r} does not use provider {provider!r}")
                    name = model if selector.startswith(provider + ":") else selector
                    identity = backend.validate_model(name)
                    checks.append({"name": selector, "ok": True, "identity": identity})
            elif provider == "python":
                checks.append(
                    {
                        "name": "python",
                        "ok": True,
                        "detail": "Runtime available; supply --models python:module:Factory to validate an adapter",
                    }
                )
            else:
                available = backend.list_models()
                checks.append(
                    {"name": "provider", "ok": True, "available_model_count": len(available)}
                )
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        checks.append({"name": "provider", "ok": False, "detail": str(exc)})
    result["ready"] = bool(checks) and all(check["ok"] for check in checks)
    return result
