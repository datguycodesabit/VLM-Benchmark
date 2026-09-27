"""Small, local-only HTTP client for the Ollama API."""

from __future__ import annotations

import base64
import ipaddress
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx


class OllamaClient:
    """Call a loopback Ollama server without pulling or forwarding models.

    ``transport`` is primarily useful for tests that use ``httpx.MockTransport``.
    The client disables environment proxy settings so loopback requests are not
    accidentally routed through a configured HTTP proxy.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 300,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Ollama request timeout must be a positive finite number")
        self.base_url = self._validate_base_url(base_url)
        self.timeout = timeout
        self._model_info_cache: dict[str, dict[str, Any]] = {}
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            trust_env=False,
        )

    @staticmethod
    def _validate_base_url(base_url: str) -> str:
        try:
            parsed = urlsplit(base_url)
            hostname = parsed.hostname
            # Accessing .port also validates malformed port syntax.
            _ = parsed.port
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Ollama base URL: {base_url!r}") from exc

        if parsed.scheme not in {"http", "https"} or not hostname:
            raise ValueError("Ollama base URL must use http(s) and name a loopback host")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Credentials are not allowed in the Ollama base URL")
        if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("Ollama base URL must be a host URL without a path or query")

        normalized_host = hostname.lower().rstrip(".")
        is_loopback = normalized_host == "localhost"
        if not is_loopback:
            try:
                is_loopback = ipaddress.ip_address(normalized_host).is_loopback
            except ValueError:
                is_loopback = False
        if not is_loopback:
            raise ValueError(
                "OllamaClient only permits loopback hosts (localhost, 127.0.0.1, or ::1)"
            )

        return base_url.rstrip("/")

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _describe_error(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and isinstance(body.get("error"), str):
            return body["error"]
        text = response.text.strip()
        return text[:500] if text else f"HTTP {response.status_code}"

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, json=payload)
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama request {method} {path} timed out after {self.timeout:g} seconds"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama request {method} {path} failed: {exc}") from exc

        if not response.is_success:
            raise RuntimeError(
                f"Ollama request {method} {path} failed "
                f"(HTTP {response.status_code}): {self._describe_error(response)}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Ollama request {method} {path} returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError(
                f"Ollama request {method} {path} returned an invalid response object"
            )
        if isinstance(data.get("error"), str):
            raise RuntimeError(f"Ollama request {method} {path} failed: {data['error']}")
        return data

    def version(self) -> str:
        data = self._request_json("GET", "/api/version")
        version = data.get("version")
        if not isinstance(version, str) or not version:
            raise RuntimeError("Ollama /api/version response did not contain a version")
        return version

    def list_models(self) -> list[dict[str, Any]]:
        data = self._request_json("GET", "/api/tags")
        models = data.get("models")
        if not isinstance(models, list) or any(not isinstance(item, dict) for item in models):
            raise RuntimeError("Ollama /api/tags response did not contain a model list")
        return models

    def model_info(self, model: str) -> dict[str, Any]:
        if not model:
            raise ValueError("Model name cannot be empty")
        info = self._request_json("POST", "/api/show", payload={"model": model, "verbose": True})
        self._model_info_cache[model] = info
        return info

    @staticmethod
    def _model_names_match(requested: str, installed: str) -> bool:
        if requested == installed:
            return True
        # Ollama treats an omitted tag as ``:latest``.
        if ":" not in requested and installed == f"{requested}:latest":
            return True
        if ":" not in installed and requested == f"{installed}:latest":
            return True
        return False

    @staticmethod
    def _is_cloud_model(model: str) -> bool:
        model_lower = model.lower()
        tag = model_lower.rsplit(":", 1)[-1]
        return tag == "cloud" or tag.endswith("-cloud") or model_lower.endswith("-cloud")

    @staticmethod
    def _reject_remote_metadata(model: str, metadata: dict[str, Any]) -> None:
        if metadata.get("remote_model") or metadata.get("remote_host"):
            raise RuntimeError(
                f"Model {model!r} is backed by a remote Ollama model; "
                "the benchmark only accepts local models"
            )

    def validate_model(self, model: str) -> dict[str, Any]:
        """Require an installed, digest-addressable, local vision model."""
        if not model:
            raise ValueError("Model name cannot be empty")
        if self._is_cloud_model(model):
            raise RuntimeError(
                f"Model {model!r} appears to be an Ollama cloud model; "
                "the benchmark only accepts local models"
            )

        entries = self.list_models()
        entry = next(
            (
                candidate
                for candidate in entries
                if self._model_names_match(
                    model,
                    str(candidate.get("name") or candidate.get("model") or ""),
                )
            ),
            None,
        )
        if entry is None:
            raise RuntimeError(f"Model {model!r} is not installed in the local Ollama instance")

        self._reject_remote_metadata(model, entry)
        digest = entry.get("digest")
        if not isinstance(digest, str) or not digest.strip():
            raise RuntimeError(f"Ollama did not provide a digest for installed model {model!r}")

        info = self.model_info(model)
        self._reject_remote_metadata(model, info)
        capabilities = info.get("capabilities")
        if not isinstance(capabilities, list) or not any(
            isinstance(capability, str) and capability.lower() == "vision"
            for capability in capabilities
        ):
            raise RuntimeError(f"Model {model!r} is not advertised as vision-capable by Ollama")

        # Keep /api/tags identity fields and digest available alongside the
        # richer /api/show fields (capabilities, details, model_info, etc.).
        return {**entry, **info, "digest": digest}

    def transcribe(
        self,
        model: str,
        image_path: Path,
        prompt: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Submit one image and return Ollama's complete chat response object."""
        if not model:
            raise ValueError("Model name cannot be empty")
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")
        if not isinstance(options, dict):
            raise TypeError("Ollama generation options must be a dictionary")
        try:
            image = Path(image_path).read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"Cannot read image for Ollama request: {image_path}: {exc}"
            ) from exc

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image).decode("ascii")],
                }
            ],
            "stream": False,
            "options": dict(options),
            "keep_alive": "10m",
        }
        thinking = self._model_info_cache.get(model, {}).get("thinking")
        values = thinking.get("values") if isinstance(thinking, dict) else None
        if isinstance(values, list) and any(value is False for value in values):
            payload["think"] = False
        data = self._request_json("POST", "/api/chat", payload=payload)
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("Ollama /api/chat response did not contain message.content text")
        if data.get("done") is not True:
            raise RuntimeError("Ollama /api/chat response was incomplete (done was not true)")
        return data

    def unload(self, model: str) -> None:
        if not model:
            raise ValueError("Model name cannot be empty")
        self._request_json(
            "POST",
            "/api/generate",
            payload={"model": model, "keep_alive": 0, "stream": False},
        )
