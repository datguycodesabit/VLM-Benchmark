"""Command-line interface; all inference is explicit and local."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import prepare_dataset
from .export import export_run
from .metrics import score, summarize
from .ollama import OllamaClient
from .runner import _atomic_json, _lock, read_records, resume_benchmark, run_benchmark


def parser():
    root = argparse.ArgumentParser(
        prog="vlm-bench", description="Benchmark local Ollama vision models on IAM handwriting."
    )
    sub = root.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare", help="Validate IAM forms, crop handwriting, and generate missing reference text"
    )
    prepare.add_argument("--data", type=Path, default=Path("data"))
    prepare.add_argument("--output", type=Path, help="Default: DATA/prepared")
    prepare.add_argument("--limit", type=int, help="Prepare a reproducible subset")
    prepare.add_argument("--seed", type=int, default=42)
    models = sub.add_parser(
        "models", help="List locally installed Ollama models and their capabilities"
    )
    models.add_argument("--base-url", default="http://localhost:11434")
    models.add_argument("--timeout", type=float, default=10)
    run = sub.add_parser("run", help="Evaluate every supplied form, or a reproducible subset")
    run.add_argument("--data", type=Path, default=Path("data"))
    run.add_argument("--models", nargs="+", required=True)
    run.add_argument("--output", type=Path, default=Path("runs"))
    run.add_argument("--limit", type=int)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--base-url", default="http://localhost:11434")
    run.add_argument("--timeout", type=float, default=300)
    run.add_argument("--num-predict", type=int, default=4096)
    run.add_argument("--no-warmup", action="store_true")
    resume = sub.add_parser(
        "resume", help="Continue unrecorded samples with unchanged data and models"
    )
    resume.add_argument("--run", type=Path, required=True)
    export = sub.add_parser("export", help="Regenerate exports without model inference")
    export.add_argument("--run", type=Path, required=True)
    export.add_argument(
        "--formats", nargs="+", choices=["xlsx", "csv", "jsonl"], default=["xlsx", "csv", "jsonl"]
    )
    rescore = sub.add_parser(
        "rescore", help="Recompute metrics from saved predictions and frozen references"
    )
    rescore.add_argument("--run", type=Path, required=True)
    return root


def _report(run_dir):
    rows = read_records(run_dir / "results.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"{'Model':<28} {'CER':>9} {'WER':>9} {'Exact':>9} {'p50 sec':>9}  Status")
    for summary in summarize(
        rows,
        expected_samples=[s["id"] for s in manifest["samples"]],
        models=manifest["models"],
        force_incomplete=manifest.get("status") in {"running", "interrupted"},
        warmups=read_records(run_dir / "warmups.jsonl"),
    ):

        def percentage(value):
            return "N/A" if value is None else f"{value:.2%}"

        latency = summary.get("median_latency_seconds")
        latency_text = "N/A" if latency is None else f"{latency:.2f}"
        status = "complete" if summary["complete"] else "incomplete"
        print(
            f"{summary['model']:<28} {percentage(summary['cer']):>9} "
            f"{percentage(summary['wer']):>9} {percentage(summary['exact_match_rate']):>9} "
            f"{latency_text:>9}  {status} ({summary['failed_count']} failed)"
        )
    for output in export_run(run_dir, ["xlsx", "csv", "jsonl"]):
        print(f"Saved: {output}")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            output = args.output or args.data / "prepared"
            samples = prepare_dataset(
                args.data, output, limit=args.limit, seed=args.seed, write_references=True
            )
            _atomic_json(output / "dataset.json", samples)
            print(f"Prepared {len(samples)} samples in {output.resolve()}")
            print("Review the references and saved crops before benchmarking.")
        elif args.command == "models":
            with OllamaClient(base_url=args.base_url, timeout=args.timeout) as client:
                for model in client.list_models():
                    name = model.get("name", model.get("model", ""))
                    try:
                        info = client.validate_model(name)
                        print(f"{name}\tvision/local\t{info['digest']}")
                    except (ValueError, RuntimeError) as exc:
                        print(f"{name}\tunavailable: {exc}")
        elif args.command == "run":
            run_dir = run_benchmark(
                args.data,
                args.models,
                args.output,
                args.limit,
                args.seed,
                args.base_url,
                args.timeout,
                args.num_predict,
                not args.no_warmup,
            )
            print(f"Run: {run_dir}")
            _report(run_dir)
            if any(r["status"] != "success" for r in read_records(run_dir / "results.jsonl")):
                return 2
        elif args.command == "resume":
            run_dir = resume_benchmark(args.run)
            _report(run_dir)
            if any(r["status"] != "success" for r in read_records(run_dir / "results.jsonl")):
                return 2
        elif args.command == "export":
            with _lock(args.run):
                for output in export_run(args.run, args.formats):
                    print(output)
        elif args.command == "rescore":
            with _lock(args.run):
                rows = read_records(args.run / "results.jsonl")
                for row in rows:
                    if row["status"] == "success":
                        row["metrics"] = score(row["prediction"], row["reference"])
                temporary = args.run / "results.rescored.tmp"
                temporary.write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8",
                )
                temporary.replace(args.run / "results.jsonl")
                _report(args.run)
    except KeyboardInterrupt:
        print(
            "Interrupted. Completed samples are saved; use vlm-bench resume --run RUN_DIRECTORY.",
            file=sys.stderr,
        )
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
