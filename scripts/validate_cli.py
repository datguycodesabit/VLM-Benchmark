"""Black-box CLI check against a local synthetic Ollama-compatible HTTP fixture.

This checks packaging, streaming, budgets, resume, measurements and offline reports.
It does not execute a model or measure handwriting recognition quality.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import platform
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageDraw


class Fixture(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    calls = 0

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, *_):
        pass

    def respond(self, value, *, ndjson=False):
        body = (
            "".join(json.dumps(item) + "\n" for item in value) if ndjson else json.dumps(value)
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson" if ndjson else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/api/version":
            self.respond({"version": "synthetic-cli-fixture-1"})
        elif self.path == "/api/tags":
            self.respond(
                {
                    "models": [
                        {"name": name, "model": name, "digest": "sha256:" + "a" * 64, "size": 0}
                        for name in ("fixture-a", "fixture-b")
                    ]
                }
            )
        else:
            self.send_error(404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path == "/api/show":
            self.respond(
                {
                    "capabilities": ["vision"],
                    "details": {"families": ["synthetic"]},
                    "model_info": {},
                }
            )
            return
        if self.path == "/api/generate" and payload.get("keep_alive") == 0:
            self.respond({"done": True, "response": ""})
            return
        if self.path != "/api/chat":
            self.send_error(404)
            return
        messages = payload.get("messages", [])
        images = [image for message in messages for image in message.get("images", [])]
        # Unload requests contain no images and are not generation requests.
        if not images:
            self.respond({"done": True, "message": {"content": ""}})
            return
        image = Image.open(io.BytesIO(base64.b64decode(images[0]))).convert("RGB")
        high, low, _ = image.getpixel((0, 0))
        index = high * 256 + low
        text = f"x+{index}" if index % 2 else f"Synthetic sample {index}"
        if payload["model"] == "fixture-b" and index % 17 == 0:
            text += "!"
        type(self).calls += 1
        terminal = {
            "model": payload["model"],
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "" if payload.get("stream") else text},
            "prompt_eval_count": 12,
            "eval_count": 5,
            "total_duration": 3000000,
            "load_duration": 0,
            "prompt_eval_duration": 1000000,
            "eval_duration": 2000000,
        }
        if payload.get("stream"):
            self.respond([{"message": {"content": text}, "done": False}, terminal], ndjson=True)
        else:
            self.respond(terminal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=1000)
    args = parser.parse_args()
    if not 2 <= args.images <= 65536:
        parser.error("--images must be between 2 and 65536")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = root / "data"
    (source / "images").mkdir(parents=True)
    (source / "text").mkdir()
    metadata = []
    for index in range(args.images):
        name = f"s{index:04d}"
        text = f"x+{index}" if index % 2 else f"Synthetic sample {index}"
        image = Image.new("RGB", (360, 45), "white")
        ImageDraw.Draw(image).text((8, 12), text, fill="black")
        image.putpixel((0, 0), (index // 256, index % 256, 0))
        image.save(source / "images" / f"{name}.png")
        (source / "text" / f"{name}.txt").write_text(text)
        metadata.append(
            {
                "id": name,
                "source_document": f"doc-{index // 10:03d}",
                "verified": True,
                "split": "test",
                "content_type": "equation" if index % 2 else "prose",
            }
        )
    (source / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metadata))
    cli = str(args.cli.resolve())

    def invoke(arguments, expected=0):
        result = subprocess.run(
            [cli, "--json", *arguments], capture_output=True, text=True, cwd=root
        )
        if result.returncode != expected:
            raise RuntimeError(f"CLI failed ({result.returncode}): {result.stderr[-2000:]}")
        payload = json.loads(result.stdout)
        assert payload["exit_code"] == expected
        return payload

    invoke(["prepare", "--data", str(source), "--output", str(root / "prepared")])
    with ThreadingHTTPServer(("127.0.0.1", 0), Fixture) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        config = root / "benchmark.toml"
        config.write_text(
            'version=3\n[experiment]\nprepared="prepared"\nmodels=["ollama:fixture-a","ollama:fixture-b"]\nwarmup=true\nmax_requests=7\n[models."ollama:fixture-a"]\nstreaming=true\n[models."ollama:fixture-b"]\nstreaming=true\n'
        )
        invoke(["doctor", "--provider", "ollama", "--base-url", base])
        paused = invoke(
            ["run", "--config", str(config), "--output", str(root / "runs"), "--base-url", base],
            expected=2,
        )
        experiment = Path(paused["experiment_directory"])
        invoke(["status", "--run", str(experiment)])
        print(
            "Initial request budget paused the experiment; resuming from saved evidence.",
            flush=True,
        )
        invoke(["resume", "--run", str(experiment), "--max-requests", str(args.images * 6 + 100)])
        status = invoke(["status", "--run", str(experiment)])["result"]
        assert (
            status["completed_passes"] == 3 and status["recorded_result_count"] == args.images * 6
        )
        server.shutdown()
        thread.join(timeout=5)
    report = invoke(["report", "--run", str(experiment), "--output", str(root / "offline-report")])
    evaluation = json.loads((root / "offline-report" / "evaluation.json").read_text())
    assert evaluation["conditions"]["baseline"]["finalizable"]
    assert any(row["metric"] == "time_to_first_output" for row in evaluation["timing_breakdowns"])
    assert any(row["metric"] == "tokens_per_second" for row in evaluation["timing_breakdowns"])
    assert all(row["output_tokens_observed"] is not None for row in evaluation["usage_summary"])
    result = {
        "kind": "synthetic-http-cli-validation",
        "images": args.images,
        "models": 2,
        "passes": 3,
        "terminal_results": status["recorded_result_count"],
        "generation_requests": Fixture.calls,
        "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "report_paths": report["report_paths"],
        "limitations": "Local synthetic HTTP fixture; no real Ollama/cloud model or GPU inference.",
    }
    (root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
