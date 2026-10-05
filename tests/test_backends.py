import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from vlm_bench.backends import BackendPaused, TrOCRBackend, create_backend, parse_model


def _sse(*events):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


def _completed(text="written words", usage=None):
    return {
        "type": "response.completed",
        "response": {
            "id": "resp-test",
            "model": "test-model-version",
            "output": [{"content": [{"type": "output_text", "text": text}], "type": "message"}],
            "usage": usage or {"input_tokens": 20, "output_tokens": 3, "total_tokens": 23},
        },
    }


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("qwen2.5vl:3b", ("ollama", "qwen2.5vl:3b")),
        ("ollama:qwen2.5vl:3b", ("ollama", "qwen2.5vl:3b")),
        ("trocr:microsoft/trocr-base-handwritten", ("trocr", "microsoft/trocr-base-handwritten")),
        ("chatgpt:gpt-6.1-sol", ("chatgpt", "gpt-6.1-sol")),
        ("openai:gpt-6.1-sol", ("openai", "gpt-6.1-sol")),
    ],
)
def test_parse_provider_selector(selector, expected):
    assert parse_model(selector) == expected


@pytest.mark.parametrize("selector", ["", "   ", "trocr:", "chatgpt: "])
def test_parse_model_rejects_empty_selector(selector):
    with pytest.raises(ValueError):
        parse_model(selector)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown backend"):
        create_backend("made-up")


def test_ollama_adapter_keeps_legacy_response_and_sanitizes_identity():
    class FakeOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def close(self):
            pass

        def list_models(self):
            return [{"name": "vision:latest", "digest": "sha256:xyz"}]

        def validate_model(self, model):
            return {
                "digest": "sha256:xyz",
                "capabilities": ["vision"],
                "details": {"family": "llama"},
                "api_key": "should-not-leak",
            }

        def version(self):
            return "0.12.0"

        def transcribe(self, *args):
            return {"message": {"content": "recognized"}, "done": True}

        def unload(self, model):
            pass

    backend = create_backend("ollama", settings={"client_factory": FakeOllama})
    identity = backend.validate_model("vision:latest")
    result = backend.transcribe(
        "vision:latest", Path("image.png"), "read", {"temperature": 0, "reasoning_effort": "high"}
    )
    assert identity["server_version"] == "0.12.0"
    assert "api_key" not in identity
    assert result == {"message": {"content": "recognized"}, "done": True}
    backend.close()


def test_openai_streams_response_and_keeps_credentials_out_of_result(tmp_path: Path):
    image = tmp_path / "line.png"
    image.write_bytes(b"small png")
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": "test-model", "modalities": ["text", "image"]}]}
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(
                {"type": "response.output_text.delta", "delta": "written words"}, _completed()
            ),
        )

    with create_backend(
        "openai",
        settings={"api_key": "unit-test-secret", "transport": httpx.MockTransport(handler)},
    ) as backend:
        assert backend.validate_model("test-model")["model"] == "test-model"
        result = backend.transcribe(
            "test-model", image, "Transcribe exactly.", {"num_predict": 48, "temperature": 0}
        )

    assert result["message"]["content"] == "written words"
    assert result["done"] is True
    assert result["model"] == "test-model-version"
    assert result["usage"]["total_tokens"] == 23
    assert result["provider_details"]["unsupported_options"] == ["temperature"]
    assert result["provider_details"]["retry_count"] == 0
    assert "unit-test-secret" not in repr(result)
    payload = json.loads(requests[-1].content)
    assert payload["store"] is False and payload["stream"] is True
    assert payload["max_output_tokens"] == 48
    assert payload["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")


def test_chatgpt_omits_unsupported_max_output_tokens_and_reports_it(tmp_path, monkeypatch):
    image = tmp_path / "line.png"
    image.write_bytes(b"image")
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_completed()),
        )

    monkeypatch.setattr("vlm_bench.backends.auth.get_access_token", lambda client_id: "temporary")
    with create_backend(
        "chatgpt", settings={"client_id": "oaiapp_test", "transport": httpx.MockTransport(handler)}
    ) as backend:
        result = backend.transcribe(
            "gpt-6.1-sol",
            image,
            "Read this.",
            {"num_predict": 100, "temperature": 0, "reasoning_effort": "xhigh"},
        )

    assert "max_output_tokens" not in captured
    assert captured["reasoning"] == {"effort": "xhigh"}
    assert captured["store"] is False and captured["stream"] is True
    assert result["provider_details"]["unsupported_options"] == [
        "num_predict",
        "temperature",
    ]


def test_stream_without_terminal_completion_discards_partial_text(tmp_path, monkeypatch):
    image = tmp_path / "line.png"
    image.write_bytes(b"image")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse({"type": "response.output_text.delta", "delta": "partial"}),
        )

    monkeypatch.setattr("vlm_bench.backends.auth.get_access_token", lambda client_id: "temporary")
    with create_backend("chatgpt", settings={"transport": httpx.MockTransport(handler)}) as backend:
        with pytest.raises(RuntimeError, match="without response.completed"):
            backend.transcribe("model", image, "read", {})
    assert len(calls) == 1


def test_subscription_usage_limit_mid_stream_pauses_without_scoring(tmp_path, monkeypatch):
    image = tmp_path / "line.png"
    image.write_bytes(b"image")

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(
                {"type": "response.output_text.delta", "delta": "partial"},
                {
                    "type": "response.failed",
                    "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded"}},
                },
            ),
        )

    monkeypatch.setattr("vlm_bench.backends.auth.get_access_token", lambda client_id: "temporary")
    with create_backend("chatgpt", settings={"transport": httpx.MockTransport(handler)}) as backend:
        with pytest.raises(BackendPaused) as caught:
            backend.transcribe("model", image, "read", {})
    assert caught.value.provider == "chatgpt"
    assert caught.value.reason == "subscription_sharing_usage_limit_exceeded"


def test_connect_failure_retries_but_stream_failure_does_not_retry(tmp_path, monkeypatch):
    image = tmp_path / "line.png"
    image.write_bytes(b"image")
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("private details", request=request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_completed()),
        )

    monkeypatch.setattr("vlm_bench.backends.auth.get_access_token", lambda client_id: "temporary")
    with create_backend("chatgpt", settings={"transport": httpx.MockTransport(handler)}) as backend:
        result = backend.transcribe("model", image, "read", {})
    assert len(calls) == 2
    assert result["provider_details"]["retry_count"] == 1


def test_retry_after_on_rejected_rate_limit_is_bounded_and_counted(tmp_path, monkeypatch):
    image = tmp_path / "line.png"
    image.write_bytes(b"image")
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                json={"error": {"code": "rate_limit_exceeded", "message": "secret"}},
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_completed()),
        )

    monkeypatch.setattr("vlm_bench.backends.auth.get_access_token", lambda client_id: "temporary")
    with create_backend("chatgpt", settings={"transport": httpx.MockTransport(handler)}) as backend:
        result = backend.transcribe("model", image, "read", {})
    assert len(calls) == 2
    assert result["provider_details"]["retry_count"] == 1


def test_subscription_http_quota_limit_pauses_without_retry(tmp_path, monkeypatch):
    image = tmp_path / "line.png"
    image.write_bytes(b"image")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            429,
            json={"error": {"code": "subscription_sharing_usage_limit_exceeded"}},
        )

    monkeypatch.setattr("vlm_bench.backends.auth.get_access_token", lambda client_id: "temporary")
    with create_backend("chatgpt", settings={"transport": httpx.MockTransport(handler)}) as backend:
        with pytest.raises(BackendPaused, match="usage_limit_exceeded"):
            backend.transcribe("model", image, "read", {})
    assert len(calls) == 1


def test_chatgpt_model_catalog_is_account_specific_and_hides_unlisted_models(
    monkeypatch,
):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "slug": "vision-model",
                        "display_name": "Vision model",
                        "visibility": "list",
                        "input_modalities": ["text", "image"],
                    },
                    {"slug": "hidden-model", "display_name": "Hidden", "visibility": "hidden"},
                ]
            },
        )

    monkeypatch.setattr("vlm_bench.backends.auth.get_access_token", lambda client_id: "temporary")
    with create_backend("chatgpt", settings={"transport": httpx.MockTransport(handler)}) as backend:
        models = backend.list_models()
        identity = backend.validate_model("vision-model")
        with pytest.raises(RuntimeError, match="not available"):
            backend.validate_model("hidden-model")
    assert [model["id"] for model in models] == ["vision-model"]
    assert identity["input_modalities"] == ["text", "image"]


def test_known_text_only_model_is_rejected(tmp_path):
    def handler(request):
        return httpx.Response(
            200,
            json={"data": [{"id": "text-only", "modalities": ["text"]}]},
        )

    with create_backend(
        "openai", settings={"api_key": "x", "transport": httpx.MockTransport(handler)}
    ) as backend:
        with pytest.raises(RuntimeError, match="text-only"):
            backend.validate_model("text-only")


class FakeTensor:
    shape = (1, 3)

    def __init__(self, values=(0, 5, 2)):
        self.values = values

    def to(self, device):
        self.device = device
        return self

    def __getitem__(self, index):
        if isinstance(index, int) and index in (0, -1):
            return self.values[index]
        return self


def test_trocr_uses_local_checkpoint_hash_and_reports_load_duration(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"local weights")
    captured = {}

    class FakeTorch:
        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return False

        @staticmethod
        def inference_mode():
            from contextlib import nullcontext

            return nullcontext()

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, location, revision=None, use_fast=None):
            captured["processor"] = (location, revision)
            captured["processor_use_fast"] = use_fast
            return cls()

        def __call__(self, images, return_tensors):
            assert return_tensors == "pt"
            return SimpleNamespace(pixel_values=FakeTensor())

        def batch_decode(self, generated, skip_special_tokens):
            return ["written words"]

    class FakeModel:
        config = SimpleNamespace(
            decoder_start_token_id=0, eos_token_id=2, _commit_hash="resolved-commit"
        )

        @classmethod
        def from_pretrained(cls, location, revision=None):
            captured["model"] = (location, revision)
            return cls()

        def to(self, device):
            captured["device"] = device

        def eval(self):
            pass

        def generate(self, pixels, **kwargs):
            captured["generation"] = kwargs
            return [FakeTensor()]

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(TrOCRProcessor=FakeProcessor, VisionEncoderDecoderModel=FakeModel),
    )
    image = tmp_path / "sample.png"
    Image.new("RGB", (60, 20), "white").save(image)
    backend = TrOCRBackend(timeout=10, settings={"num_beams": 3})
    identity = backend.validate_model(str(checkpoint))
    backend.load(str(checkpoint))
    result = backend.transcribe(str(checkpoint), image, "ignored prompt", {"num_predict": 20})
    assert identity["checkpoint_sha256"]
    assert identity["device"] == "cpu"
    assert captured["generation"]["num_beams"] == 3
    assert captured["processor_use_fast"] is False
    assert result["message"]["content"] == "written words"
    assert result["provider_details"]["revision"] == "resolved-commit"
    assert result["load_duration"] > 0
    backend.unload(str(checkpoint))


def test_trocr_missing_optional_transformers_is_actionable(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")

    class FakeTorch:
        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return False

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "transformers", None)
    backend = TrOCRBackend(timeout=10, settings={})
    backend.validate_model(str(checkpoint))
    with pytest.raises(RuntimeError, match="uv sync --extra trocr"):
        backend.load(str(checkpoint))
