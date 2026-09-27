import base64
import json
from pathlib import Path

import httpx
import pytest

from vlm_bench.ollama import OllamaClient


def response(request: httpx.Request, body: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=body, request=request)


def test_local_only_base_url_validation() -> None:
    for url in (
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://[::1]:11434",
        "http://127.0.0.2:11434",
    ):
        client = OllamaClient(
            url, transport=httpx.MockTransport(lambda request: response(request, {}))
        )
        client.close()

    for url in (
        "https://ollama.com",
        "http://192.168.1.20:11434",
        "http://localhost:11434/api",
        "http://user:pass@localhost:11434",
    ):
        with pytest.raises(ValueError):
            OllamaClient(url)


def test_version_and_model_listing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return response(request, {"version": "0.12.0"})
        return response(request, {"models": [{"name": "llava:7b", "digest": "sha256:abc"}]})

    with OllamaClient(transport=httpx.MockTransport(handler)) as client:
        assert client.version() == "0.12.0"
        assert client.list_models() == [{"name": "llava:7b", "digest": "sha256:abc"}]


def test_transcribe_posts_single_image_and_returns_raw_response(tmp_path: Path) -> None:
    image_bytes = b"handwritten image bytes"
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(image_bytes)
    raw_response = {
        "model": "llava:7b",
        "message": {"role": "assistant", "content": "The handwritten text."},
        "done": True,
        "total_duration": 123,
    }
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return response(request, raw_response)

    options = {"temperature": 0, "seed": 42}
    with OllamaClient(transport=httpx.MockTransport(handler)) as client:
        result = client.transcribe("llava:7b", image_path, "Transcribe exactly.", options)

    assert result == raw_response
    assert captured["model"] == "llava:7b"
    assert captured["messages"] == [
        {
            "role": "user",
            "content": "Transcribe exactly.",
            "images": [base64.b64encode(image_bytes).decode("ascii")],
        }
    ]
    assert captured["stream"] is False
    assert captured["options"] == options
    assert captured["keep_alive"] == "10m"
    assert "think" not in captured


def test_transcribe_disables_thinking_only_when_model_metadata_allows_it(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"image")
    chat_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return response(
                request,
                {"thinking": {"values": [False, True], "default": True}},
            )
        chat_payload.update(json.loads(request.content))
        return response(
            request,
            {"message": {"content": "handwritten text"}, "done": True},
        )

    with OllamaClient(transport=httpx.MockTransport(handler)) as client:
        client.model_info("vision")
        client.transcribe("vision", image_path, "transcribe", {})

    assert chat_payload["think"] is False


def test_validate_model_combines_tags_and_show_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return response(
                request,
                {"models": [{"name": "llava:7b", "digest": "sha256:abc", "size": 100}]},
            )
        assert request.url.path == "/api/show"
        assert json.loads(request.content) == {"model": "llava:7b", "verbose": True}
        return response(
            request,
            {
                "capabilities": ["completion", "vision"],
                "details": {"family": "llava"},
                "model_info": {"general.architecture": "llama"},
            },
        )

    with OllamaClient(transport=httpx.MockTransport(handler)) as client:
        metadata = client.validate_model("llava:7b")

    assert metadata["digest"] == "sha256:abc"
    assert metadata["size"] == 100
    assert metadata["capabilities"] == ["completion", "vision"]
    assert metadata["details"] == {"family": "llava"}
    assert metadata["model_info"] == {"general.architecture": "llama"}


@pytest.mark.parametrize("model", ["gpt-oss:120b-cloud", "example:cloud"])
def test_validate_model_rejects_cloud_tags_without_request(model: str) -> None:
    def unexpected(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"cloud model must be rejected before making a request: {request.url}")

    with OllamaClient(transport=httpx.MockTransport(unexpected)) as client:
        with pytest.raises(RuntimeError, match="cloud"):
            client.validate_model(model)


@pytest.mark.parametrize(
    ("tag_metadata", "show_metadata", "message"),
    [
        (
            {"name": "remote:vision", "digest": "sha256:x", "remote_host": "https://ollama.com"},
            None,
            "remote",
        ),
        ({"name": "text:only", "digest": "sha256:x"}, {"capabilities": ["completion"]}, "vision"),
        ({"name": "missing-digest", "digest": ""}, None, "digest"),
    ],
)
def test_validate_model_rejects_remote_nonvision_and_missing_digest(
    tag_metadata: dict, show_metadata: dict | None, message: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return response(request, {"models": [tag_metadata]})
        if show_metadata is None:
            pytest.fail("model_info must not be fetched after an invalid tags entry")
        return response(request, show_metadata)

    model = tag_metadata["name"]
    with OllamaClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match=message):
            client.validate_model(model)


def test_validate_model_rejects_remote_metadata_from_show() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return response(request, {"models": [{"name": "vision", "digest": "sha256:x"}]})
        return response(
            request,
            {"remote_model": "upstream", "capabilities": ["vision"]},
        )

    with OllamaClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="remote"):
            client.validate_model("vision")


@pytest.mark.parametrize(
    "body",
    [
        {"message": {"content": "text"}, "done": False},
        {"message": {"content": None}, "done": True},
        {"done": True},
    ],
)
def test_transcribe_rejects_incomplete_or_malformed_response(tmp_path: Path, body: dict) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"image")
    with OllamaClient(
        transport=httpx.MockTransport(lambda request: response(request, body))
    ) as client:
        with pytest.raises(RuntimeError, match="/api/chat"):
            client.transcribe("vision", image_path, "transcribe", {})


def test_transcribe_reports_server_error_field(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"image")
    with OllamaClient(
        transport=httpx.MockTransport(
            lambda request: response(request, {"error": "model is unavailable"}, status=404)
        )
    ) as client:
        with pytest.raises(RuntimeError, match="model is unavailable"):
            client.transcribe("missing", image_path, "transcribe", {})


def test_client_wraps_timeout_and_malformed_json(tmp_path: Path) -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    with OllamaClient(timeout=2, transport=httpx.MockTransport(timeout)) as client:
        with pytest.raises(RuntimeError, match="timed out after 2 seconds"):
            client.version()

    with OllamaClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"not-json", request=request)
        )
    ) as client:
        with pytest.raises(RuntimeError, match="invalid JSON"):
            client.version()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_constructor_rejects_nonpositive_or_nonfinite_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        OllamaClient(timeout=timeout)


def test_unload_requests_immediate_model_release() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return response(request, {"model": "llava:7b", "done": True})

    with OllamaClient(transport=httpx.MockTransport(handler)) as client:
        assert client.unload("llava:7b") is None
    assert captured == {"model": "llava:7b", "keep_alive": 0, "stream": False}
