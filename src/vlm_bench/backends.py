"""Provider adapters used by the benchmark runner.

Adapters intentionally return a small Ollama-shaped response so older result
consumers can continue reading ``message.content`` and completion metadata.
Cloud credentials are never included in identities, responses, or exceptions.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import json
import mimetypes
import os
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from . import auth
from .execution import RetryableProviderError
from .ollama import OllamaClient

DEFAULT_TROCR_MODEL = "microsoft/trocr-base-handwritten"
OPENAI_API_BASE = "https://api.openai.com/v1"
_PROVIDERS = {"ollama", "trocr", "chatgpt", "openai"}


class BackendPaused(RuntimeError):
    """A provider asked the benchmark to stop this model while saving progress."""

    def __init__(self, provider: str, model: str, reason: str) -> None:
        self.provider = provider
        self.model = model
        self.reason = reason
        super().__init__(f"{provider} model {model!r} paused: {reason}")


class _RetryableStatus(Exception):
    def __init__(self, delay: float) -> None:
        self.delay = delay


class Backend:
    """Small common interface for OCR and vision-language backends."""

    provider = "unknown"

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        pass

    def list_models(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def validate_model(self, model: str) -> dict[str, Any]:
        raise NotImplementedError

    def load(self, model: str) -> None:
        """Load or warm one model. Stateless providers may keep the no-op."""

    def transcribe(
        self,
        model: str,
        image_path: Path,
        prompt: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def transcribe_once(
        self,
        model: str,
        image_path: Path,
        prompt: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform one generation request; managed retries belong to engine."""
        return self.transcribe(model, image_path, prompt, options)

    def unload(self, model: str) -> None:
        """Release a model after its benchmark run when possible."""


def parse_model(selector: str) -> tuple[str, str]:
    """Return ``(provider, model_id)``; bare IDs remain Ollama IDs.

    ``external:<system>`` is a selector for imported predictions only. It is
    intentionally not a backend provider and cannot be passed to
    :func:`create_backend`.
    """
    if not isinstance(selector, str) or not selector.strip():
        raise ValueError("Model selector cannot be empty")
    selector = selector.strip()
    provider, separator, model = selector.partition(":")
    provider = provider.lower()
    if separator and provider in _PROVIDERS | {"external"}:
        model = model.strip()
        if not model:
            raise ValueError(f"Model selector {selector!r} has no model ID")
        return provider, model
    return "ollama", selector


class OllamaBackend(Backend):
    provider = "ollama"

    def __init__(self, base_url: str, timeout: float, settings: dict[str, Any]) -> None:
        client_type = settings.get("client_factory", OllamaClient)
        kwargs: dict[str, Any] = {"base_url": base_url, "timeout": timeout}
        if settings.get("transport") is not None:
            kwargs["transport"] = settings["transport"]
        self.client = client_type(**kwargs)
        self._version: str | None = None

    def close(self) -> None:
        self.client.close()

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": item.get("name") or item.get("model"),
                "provider": self.provider,
                "digest": item.get("digest"),
                "size": item.get("size"),
            }
            for item in self.client.list_models()
        ]

    def validate_model(self, model: str) -> dict[str, Any]:
        metadata = self.client.validate_model(model)
        if self._version is None:
            self._version = self.client.version()
        # Ollama metadata is local model metadata. Retain useful architecture
        # fields, while excluding any incidental non-JSON values.
        return {
            "provider": self.provider,
            "model": model,
            "digest": metadata.get("digest"),
            "capabilities": metadata.get("capabilities", []),
            "details": metadata.get("details", {}),
            "model_info": metadata.get("model_info", {}),
            "size": metadata.get("size"),
            "server_version": self._version,
        }

    def transcribe(self, model, image_path, prompt, options):
        clean_options = {
            key: value
            for key, value in options.items()
            if key not in {"image_detail", "reasoning_effort", "max_output_tokens"}
        }
        return self.client.transcribe(model, image_path, prompt, clean_options)

    def unload(self, model: str) -> None:
        self.client.unload(model)


class TrOCRBackend(Backend):
    """Lazy Transformers adapter for single handwritten prose lines."""

    provider = "trocr"

    def __init__(self, timeout: float, settings: dict[str, Any]) -> None:
        self.timeout = timeout
        self.settings = settings
        self._models: dict[str, tuple[Any, Any, Any]] = {}
        self._identities: dict[str, dict[str, Any]] = {}
        self._load_durations: dict[str, int] = {}

    @staticmethod
    def _extra_error() -> RuntimeError:
        return RuntimeError(
            "TrOCR support requires the optional dependencies; install with `uv sync --extra trocr`"
        )

    @staticmethod
    def _local_hash(path: Path) -> str:
        digest = hashlib.sha256()
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if not files:
            raise RuntimeError(f"Local TrOCR checkpoint is empty: {path}")
        for item in files:
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with item.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def _resolved_model(self, model: str) -> tuple[str, str | None, str | None]:
        path = Path(model).expanduser()
        if path.exists():
            if not path.is_dir():
                raise RuntimeError("A local TrOCR model must be a checkpoint directory")
            resolved = str(path.resolve())
            return resolved, None, self._local_hash(path)
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise self._extra_error() from exc
        revision = self.settings.get("revision")
        try:
            info = HfApi().model_info(model, revision=revision)
        except Exception as exc:
            # Do not expose request headers or remote response bodies.
            raise RuntimeError(
                f"Could not resolve TrOCR model {model!r} from Hugging Face ({type(exc).__name__})"
            ) from None
        commit = getattr(info, "sha", None)
        if not isinstance(commit, str) or not commit:
            raise RuntimeError(f"Hugging Face did not return an immutable revision for {model!r}")
        return model, commit, None

    def list_models(self) -> list[dict[str, Any]]:
        model = str(self.settings.get("model", DEFAULT_TROCR_MODEL))
        return [{"id": model, "provider": self.provider, "display_name": model}]

    def validate_model(self, model: str) -> dict[str, Any]:
        if not model:
            raise ValueError("TrOCR model ID cannot be empty")
        repo_or_path, revision, local_hash = self._resolved_model(model)
        try:
            import torch
        except ImportError as exc:
            raise self._extra_error() from exc
        configured_device = self.settings.get("device", "auto")
        if configured_device not in {"auto", "cpu", "mps"}:
            raise ValueError("TrOCR device must be auto, cpu, or mps")
        mps_available = bool(
            getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
        )
        preferred_device = (
            configured_device if configured_device != "auto" else "mps" if mps_available else "cpu"
        )
        if preferred_device == "mps" and not mps_available:
            preferred_device = "cpu"
        identity = {
            "provider": self.provider,
            "model": repo_or_path,
            "revision": revision,
            "checkpoint_sha256": local_hash,
            "task": "handwritten-prose-line-ocr",
            "device": preferred_device,
        }
        if model == DEFAULT_TROCR_MODEL:
            identity["pretrained_data_note"] = "The published checkpoint is IAM-trained."
        self._identities[model] = identity
        return dict(identity)

    def load(self, model: str) -> None:
        if model in self._models:
            return
        identity = self._identities.get(model) or self.validate_model(model)
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise self._extra_error() from exc

        location = identity["model"]
        revision = identity.get("revision")
        processor = None
        loaded_model = None
        requested_device = identity["device"]
        load_started = time.perf_counter()
        try:
            processor = TrOCRProcessor.from_pretrained(location, revision=revision, use_fast=False)
            loaded_model = VisionEncoderDecoderModel.from_pretrained(location, revision=revision)
            try:
                loaded_model.to(requested_device)
                selected_device = requested_device
            except Exception:
                if requested_device != "mps":
                    raise
                del loaded_model
                loaded_model = VisionEncoderDecoderModel.from_pretrained(
                    location, revision=revision
                )
                loaded_model.to("cpu")
                selected_device = "cpu"
            loaded_model.eval()
        except Exception as exc:
            if processor is not None:
                del processor
            if loaded_model is not None:
                del loaded_model
            raise RuntimeError(
                f"Could not load TrOCR model {model!r} ({type(exc).__name__})"
            ) from None
        self._models[model] = (processor, loaded_model, torch)
        identity["device"] = selected_device
        self._load_durations[model] = int((time.perf_counter() - load_started) * 1_000_000_000)
        config = getattr(loaded_model, "config", None)
        resolved = getattr(config, "_commit_hash", None)
        if resolved:
            identity["revision"] = resolved

    def transcribe(self, model, image_path, prompt, options):
        self.load(model)
        processor, loaded_model, torch = self._models[model]
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                pixels = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
            device = self._identities[model]["device"]
            pixels = pixels.to(device)
            max_new_tokens = options.get("num_predict", 128)
            if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
                max_new_tokens = 128
            max_new_tokens = max(1, min(max_new_tokens, 4096))
            num_beams = self.settings.get("num_beams", 1)
            if (
                isinstance(num_beams, bool)
                or not isinstance(num_beams, int)
                or not 1 <= num_beams <= 10
            ):
                raise ValueError("TrOCR num_beams must be an integer from 1 to 10")
            started = time.perf_counter()
            with torch.inference_mode():
                generated = loaded_model.generate(
                    pixels,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=num_beams,
                )
            generation_seconds = time.perf_counter() - started
            token_ids = generated[0]
            token_count = int(token_ids.shape[-1])
            start_id = getattr(loaded_model.config, "decoder_start_token_id", None)
            if start_id is not None and token_count and int(token_ids[0]) == int(start_id):
                token_count -= 1
            text = processor.batch_decode(generated, skip_special_tokens=True)[0]
            eos_id = getattr(loaded_model.config, "eos_token_id", None)
            ended = bool(eos_id is not None and int(generated[0][-1]) == int(eos_id))
            return {
                "model": model,
                "message": {"role": "assistant", "content": text},
                "done": True,
                "done_reason": "stop"
                if ended
                else "length"
                if token_count >= max_new_tokens
                else "stop",
                "eval_count": token_count,
                "load_duration": self._load_durations.pop(model, 0),
                "provider": self.provider,
                "provider_details": {
                    "device": device,
                    "revision": self._identities[model].get("revision"),
                    "checkpoint_sha256": self._identities[model].get("checkpoint_sha256"),
                    "generation_seconds": generation_seconds,
                    "num_beams": num_beams,
                    "processor_use_fast": False,
                    "prompt_supported": False,
                    "unsupported_options": sorted(
                        key
                        for key in options
                        if key not in {"num_predict", "num_beams", "device", "revision"}
                    ),
                },
            }
        except (RuntimeError, OSError, ValueError):
            raise
        except Exception as exc:
            raise RuntimeError(f"TrOCR inference failed ({type(exc).__name__})") from None

    def unload(self, model: str) -> None:
        loaded = self._models.pop(model, None)
        self._load_durations.pop(model, None)
        if loaded is None:
            return
        del loaded
        gc.collect()
        try:
            import torch

            if bool(getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()):
                torch.mps.empty_cache()
        except (ImportError, AttributeError, RuntimeError):
            pass


class _ResponsesBackend(Backend):
    """Common streaming Responses API implementation."""

    provider = "openai"

    def __init__(self, timeout: float, settings: dict[str, Any]) -> None:
        self.timeout = timeout
        self.settings = settings
        self.client_id = settings.get("client_id")
        transport = settings.get("transport")
        base_url = settings.get("api_base_url", OPENAI_API_BASE).rstrip("/")
        parsed = urlsplit(base_url)
        if transport is None and (parsed.scheme != "https" or parsed.netloc != "api.openai.com"):
            raise ValueError("Cloud API base URL must be https://api.openai.com/v1")
        self.client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            trust_env=transport is None,
        )

    def close(self) -> None:
        self.client.close()

    def _access_token(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        token = self._access_token()
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    @staticmethod
    def _safe_error(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = {}
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            code = error.get("code") or error.get("type")
            if isinstance(code, str) and code.replace("_", "").replace("-", "").isalnum():
                return code[:80]
        return f"HTTP {response.status_code}"

    def list_models(self) -> list[dict[str, Any]]:
        try:
            response = self.client.get("/models", headers=self._headers())
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"{self.provider} model listing failed ({type(exc).__name__})"
            ) from None
        if not response.is_success:
            raise RuntimeError(
                f"{self.provider} model listing failed: {self._safe_error(response)}"
            )
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(f"{self.provider} model listing returned invalid JSON") from None
        entries = data.get("models") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            entries = data.get("data") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            raise RuntimeError(f"{self.provider} model listing response has no model list")
        models = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("slug") or entry.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            if self.provider == "chatgpt" and entry.get("visibility") != "list":
                continue
            models.append(
                {
                    "id": model_id,
                    "provider": self.provider,
                    "display_name": entry.get("display_name") or model_id,
                    "input_modalities": entry.get("input_modalities") or entry.get("modalities"),
                    "capabilities": entry.get("capabilities"),
                }
            )
        return models

    def validate_model(self, model: str) -> dict[str, Any]:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model ID cannot be empty")
        available = self.list_models()
        selected = next((item for item in available if item["id"] == model), None)
        if selected is None:
            raise RuntimeError(f"Model {model!r} is not available to {self.provider}")
        modalities = selected.get("input_modalities")
        if (
            isinstance(modalities, list)
            and modalities
            and not any(str(item).lower() in {"image", "vision"} for item in modalities)
        ):
            raise RuntimeError(f"Model {model!r} is advertised as text-only by {self.provider}")
        return {
            "provider": self.provider,
            "model": model,
            "input_modalities": modalities,
            "capabilities": selected.get("capabilities"),
        }

    def transcribe(self, model, image_path, prompt, options):
        return self._transcribe(model, image_path, prompt, options, managed=False)

    def transcribe_once(self, model, image_path, prompt, options):
        return self._transcribe(model, image_path, prompt, options, managed=True)

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            delay = float(value)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                delay = (retry_at - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return None
        return max(0.0, delay)

    def _transcribe(self, model, image_path, prompt, options, *, managed):
        try:
            image = Path(image_path).read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"Cannot read image for {self.provider} request ({type(exc).__name__})"
            ) from None
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            mime_type = "image/png"
        image_url = f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"
        image_detail = self.settings.get("image_detail", options.get("image_detail", "auto"))
        if image_detail not in {"auto", "low", "high"}:
            raise ValueError("Image detail must be one of auto, low, or high")
        payload = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url, "detail": image_detail},
                    ],
                }
            ],
            "store": False,
            "stream": True,
        }
        max_output_tokens = options.get("max_output_tokens", options.get("num_predict"))
        if (
            self.provider == "openai"
            and isinstance(max_output_tokens, int)
            and not isinstance(max_output_tokens, bool)
        ):
            payload["max_output_tokens"] = max(1, max_output_tokens)
        reasoning_effort = self.settings.get("reasoning_effort", options.get("reasoning_effort"))
        if reasoning_effort is not None:
            if reasoning_effort not in {
                "none",
                "minimal",
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
                "ultra",
            }:
                raise ValueError("Unsupported reasoning effort value")
            payload["reasoning"] = {"effort": reasoning_effort}
        unsupported_options = sorted(
            key
            for key in options
            if key
            not in {
                "num_predict",
                "max_output_tokens",
                "image_detail",
                "reasoning_effort",
            }
        )
        if self.provider == "chatgpt":
            unsupported_options = sorted(
                set(unsupported_options) | {"num_predict", "max_output_tokens"} & set(options)
            )
        output_parts: list[str] = []
        completed_data: dict[str, Any] | None = None
        retry_count = 0
        # Only retry failures raised while establishing the connection. Once a
        # response begins, a dropped stream may represent a completed billable
        # request, so the adapter never replays it.
        max_attempts = 1 if managed else 3
        for attempt in range(max_attempts):
            output_parts.clear()
            completed_data = None
            try:
                with self.client.stream(
                    "POST", "/responses", headers=self._headers(), json=payload
                ) as response:
                    if not response.is_success:
                        error_code = self._safe_error(response)
                        if self.provider == "chatgpt" and error_code in {
                            "subscription_sharing_usage_limit_exceeded",
                            "subscription_sharing_usage_unavailable",
                        }:
                            raise BackendPaused(self.provider, model, error_code)
                        if response.status_code == 429:
                            delay = self._retry_after(response)
                            if managed:
                                raise RetryableProviderError(
                                    f"{self.provider} rate limit exceeded",
                                    retry_after_seconds=delay,
                                    category="rate_limited",
                                )
                            if attempt < 2:
                                raise _RetryableStatus(
                                    max(0.1, min(delay if delay is not None else 0.5, 3.0))
                                )
                        if response.status_code in {500, 502, 503, 504}:
                            if managed:
                                raise RetryableProviderError(
                                    f"{self.provider} temporary HTTP {response.status_code}",
                                    retry_after_seconds=self._retry_after(response),
                                    category=f"http_{response.status_code}",
                                )
                        raise RuntimeError(f"{self.provider} response failed: {error_code}")
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        encoded = line[5:].strip()
                        if not encoded or encoded == "[DONE]":
                            continue
                        try:
                            event = json.loads(encoded)
                        except ValueError:
                            raise RuntimeError(
                                f"{self.provider} returned malformed stream data"
                            ) from None
                        if not isinstance(event, dict):
                            continue
                        event_type = event.get("type")
                        if event_type == "response.output_text.delta":
                            delta = event.get("delta")
                            if isinstance(delta, str):
                                output_parts.append(delta)
                        elif event_type == "response.completed":
                            completed_data = (
                                event.get("response")
                                if isinstance(event.get("response"), dict)
                                else {}
                            )
                        elif event_type in {"response.failed", "error"}:
                            failure = (
                                event.get("response")
                                if isinstance(event.get("response"), dict)
                                else event
                            )
                            error = failure.get("error") if isinstance(failure, dict) else None
                            code = error.get("code") if isinstance(error, dict) else None
                            if self.provider == "chatgpt" and code in {
                                "subscription_sharing_usage_limit_exceeded",
                                "subscription_sharing_usage_unavailable",
                            }:
                                raise BackendPaused(self.provider, model, str(code))
                            safe_code = (
                                re.sub(r"[^A-Za-z0-9_-]", "_", code)[:80]
                                if isinstance(code, str)
                                else "request_failed"
                            )
                            raise RuntimeError(f"{self.provider} response failed: {safe_code}")
                        elif event_type == "response.incomplete":
                            raise RuntimeError(f"{self.provider} response was incomplete")
                break
            except _RetryableStatus as retry:
                retry_count += 1
                time.sleep(retry.delay)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if managed:
                    raise RetryableProviderError(
                        f"{self.provider} connection failed before response",
                        category="connection_error",
                    ) from None
                if attempt >= 2:
                    raise RuntimeError(
                        f"{self.provider} could not connect after {attempt + 1} attempts"
                    ) from None
                retry_count += 1
                time.sleep(0.1 * (attempt + 1))
            except httpx.TimeoutException:
                raise RuntimeError(
                    f"{self.provider} request timed out; partial output was discarded"
                ) from None
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"{self.provider} request failed ({type(exc).__name__}); partial output was discarded"
                ) from None
        if completed_data is None:
            raise RuntimeError(
                f"{self.provider} stream ended without response.completed; partial output was discarded"
            )
        if not output_parts:
            output_parts.extend(self._extract_output_text(completed_data))
        usage = completed_data.get("usage") if isinstance(completed_data, dict) else None
        result: dict[str, Any] = {
            "model": completed_data.get("model", model),
            "message": {"role": "assistant", "content": "".join(output_parts)},
            "done": True,
            "done_reason": "stop",
            "provider": self.provider,
            "usage": usage if isinstance(usage, dict) else None,
            "provider_details": {
                "response_id": completed_data.get("id"),
                "usage": usage if isinstance(usage, dict) else None,
                "image_detail": image_detail,
                "reasoning_effort": reasoning_effort,
                "unsupported_options": unsupported_options,
                "retry_count": retry_count,
            },
        }
        if isinstance(usage, dict):
            result["prompt_eval_count"] = usage.get("input_tokens")
            result["eval_count"] = usage.get("output_tokens")
        return result

    @staticmethod
    def _extract_output_text(response: dict[str, Any]) -> list[str]:
        result: list[str] = []
        output = response.get("output")
        if not isinstance(output, list):
            return result
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                    value = part.get("text")
                    if isinstance(value, str):
                        result.append(value)
        return result


class ChatGPTBackend(_ResponsesBackend):
    provider = "chatgpt"

    def _access_token(self) -> str:
        return auth.get_access_token(self.client_id)


class OpenAIBackend(_ResponsesBackend):
    provider = "openai"

    def _access_token(self) -> str:
        token = self.settings.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not isinstance(token, str) or not token:
            raise RuntimeError(
                "OpenAI API access requires OPENAI_API_KEY or an explicitly configured API key"
            )
        return token


def create_backend(
    provider: str,
    base_url: str = "http://localhost:11434",
    timeout: float = 300,
    settings: dict[str, Any] | None = None,
) -> Backend:
    """Create a provider backend. Optional heavyweight imports remain lazy."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("Backend timeout must be a positive number")
    settings = dict(settings or {})
    name = provider.lower()
    if name == "ollama":
        return OllamaBackend(base_url, timeout, settings)
    if name == "trocr":
        return TrOCRBackend(timeout, settings)
    if name == "chatgpt":
        return ChatGPTBackend(timeout, settings)
    if name == "openai":
        return OpenAIBackend(timeout, settings)
    raise ValueError(
        f"Unknown backend provider {provider!r}; choose one of {', '.join(sorted(_PROVIDERS))}"
    )
