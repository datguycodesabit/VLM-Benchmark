"""Command-line interface with explicitly selected local and cloud engines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import prepare_dataset
from .export import export_run
from .metrics import score, summarize
from .runner import _atomic_json, _lock, read_records, resume_benchmark


def parser():
    root = argparse.ArgumentParser(
        prog="vlm-bench",
        description="Benchmark handwriting recognition across local and cloud models.",
    )
    sub = root.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare", help="Validate and freeze paired or IAM handwriting samples"
    )
    prepare.add_argument("--data", type=Path, default=Path("data"))
    prepare.add_argument("--output", type=Path, help="Default: DATA/prepared")
    prepare.add_argument("--limit", type=int, help="Prepare a reproducible subset")
    prepare.add_argument("--seed", type=int, default=42)
    for command in (prepare,):
        _dataset_options(command)
    models = sub.add_parser("models", help="List available models for the selected provider")
    models.add_argument("--base-url", default="http://localhost:11434")
    models.add_argument("--timeout", type=float, default=10)
    models.add_argument(
        "--provider", choices=["ollama", "trocr", "chatgpt", "openai"], default="ollama"
    )
    run = sub.add_parser("run", help="Evaluate a frozen dataset or reproducible subset")
    run.add_argument("--data", type=Path, default=Path("data"))
    run.add_argument("--models", nargs="+")
    run.add_argument("--output", type=Path, default=Path("runs"))
    run.add_argument("--limit", type=int)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--base-url", default="http://localhost:11434")
    run.add_argument("--timeout", type=float, default=300)
    run.add_argument("--num-predict", type=int, default=4096)
    run.add_argument("--no-warmup", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--config", type=Path)
    _dataset_options(run)
    resume = sub.add_parser(
        "resume", help="Continue unrecorded samples with unchanged data and models"
    )
    resume.add_argument("--run", type=Path, required=True)
    resume.add_argument("--retry-failed", action="store_true")
    auth = sub.add_parser("auth", help="Manage official ChatGPT subscription sign-in")
    auth_sub = auth.add_subparsers(dest="auth_command", required=True)
    for action in ("login", "status", "logout"):
        auth_sub.add_parser(action).add_argument(
            "--provider", choices=["chatgpt"], default="chatgpt"
        )
    doctor = sub.add_parser("doctor", help="Check dependencies, credentials and engines")
    doctor.add_argument(
        "--provider", choices=["ollama", "trocr", "chatgpt", "openai"], default="ollama"
    )
    doctor.add_argument("--base-url", default="http://localhost:11434")
    doctor.add_argument("--models", nargs="*", default=[])
    dataset = sub.add_parser(
        "dataset", help="Audit paired data or generate document-grouped splits"
    )
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    check = dataset_sub.add_parser("check")
    check.add_argument("--data", type=Path, default=Path("data"))
    check.add_argument(
        "--json", action="store_true", help="Print the complete audit, including sample records"
    )
    _dataset_options(check)
    split_cmd = dataset_sub.add_parser("split")
    split_cmd.add_argument("--data", type=Path, required=True)
    split_cmd.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New metadata JSONL; existing files are preserved",
    )
    split_cmd.add_argument("--seed", type=int, default=42)
    _dataset_options(split_cmd)
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


def _dataset_options(command):
    command.add_argument(
        "--layout", choices=["auto", "paired", "iam-words", "iam-lines", "iam-forms"], default=None
    )
    command.add_argument("--preprocess", choices=["original", "enhanced"], default=None)
    command.add_argument("--split", choices=["train", "validation", "test"])
    command.add_argument("--content-type", choices=["prose", "equation", "word"])


def _report(run_dir):
    rows = read_records(run_dir / "results.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") == 2:
        from .research import research_reports

        research = research_reports(
            rows, dict(manifest, warmups=read_records(run_dir / "warmups.jsonl"))
        )
        for track, report in research["tracks"].items():
            if not report["eligible_sample_count"]:
                continue
            print(f"{track}: {report['eligible_sample_count']} eligible samples")
            print(f"{'Model':<40} {'CER':>9} {'WER':>9} {'Exact':>9} Status")
            for row in report["model_results"]:
                values = [
                    "N/A" if row.get(key) is None else f"{row[key]:.2%}"
                    for key in ("cer", "wer", "exact_match_rate")
                ]
                print(
                    f"{row['model']:<40} {values[0]:>9} {values[1]:>9} {values[2]:>9} {row['status']}"
                )
        for output in export_run(run_dir, ["xlsx", "csv", "jsonl"]):
            print(f"Saved: {output}")
        return
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
                args.data,
                output,
                limit=args.limit,
                seed=args.seed,
                write_references=True,
                layout=args.layout or "auto",
                preprocess=args.preprocess or "original",
                split=args.split,
                content_type=args.content_type,
            )
            _atomic_json(output / "dataset.json", samples)
            print(f"Prepared {len(samples)} samples in {output.resolve()}")
            print("Review the references and saved crops before benchmarking.")
        elif args.command == "auth":
            from . import auth

            print(json.dumps(getattr(auth, args.auth_command)(), indent=2))
        elif args.command == "dataset":
            from .dataset import check_dataset, split_dataset

            report = check_dataset(args.data, layout=args.layout or "auto")
            if args.dataset_command == "check":
                if args.json:
                    print(json.dumps(report, indent=2, ensure_ascii=False))
                else:
                    print(f"Layout: {report['layout']} | Valid: {report['valid']}")
                    print(json.dumps(report.get("counts", {}), indent=2))
                    for issue in report.get("issues", [])[:20]:
                        print(json.dumps(issue, ensure_ascii=False))
                    print(
                        f"Excluded: {len(report.get('excluded', []))}; use --json for the complete audit"
                    )
                return 0 if report["valid"] else 1
            if not report["valid"]:
                raise ValueError("Dataset check failed; resolve issues before splitting")
            records = split_dataset(report["samples"], seed=args.seed)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            print(f"Saved split metadata: {args.output}")
        elif args.command in {"models", "doctor"}:
            from .backends import create_backend

            if args.command == "doctor":
                import importlib.util
                import os

                dependencies = {
                    n: importlib.util.find_spec(n) is not None
                    for n in ("torch", "transformers", "keyring", "jwt")
                }
                print(
                    json.dumps(
                        {
                            "provider": args.provider,
                            "dependencies": dependencies,
                            "openai_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
                        },
                        indent=2,
                    )
                )
            with create_backend(
                args.provider, base_url=args.base_url, timeout=getattr(args, "timeout", 10)
            ) as client:
                if args.command == "doctor" and args.models:
                    from .backends import parse_model

                    for name in args.models:
                        provider, resolved = parse_model(name)
                        if name.startswith(args.provider + ":"):
                            name = resolved
                        print(json.dumps(client.validate_model(name), indent=2))
                    return 0
                for model in client.list_models():
                    name = model.get("name", model.get("model", model.get("id", "")))
                    try:
                        info = (
                            client.validate_model(name)
                            if args.provider in {"ollama", "trocr"}
                            else model
                        )
                        print(
                            f"{name}\t{args.provider}\t{info.get('digest', info.get('revision', 'remote'))}"
                        )
                    except (ValueError, RuntimeError) as exc:
                        print(f"{name}\tunavailable: {exc}")
        elif args.command == "run":
            from .config import load_config
            from .engine import preview, run

            config = load_config(args.config)
            experiment = config["experiment"]
            selected = args.models or experiment.get("models")
            if not selected:
                raise ValueError("Supply --models or experiment.models")
            supplied = list(argv) if argv is not None else sys.argv[1:]

            def chosen(name, current, default):
                flag = "--" + name.replace("_", "-")
                explicit = any(a == flag or a.startswith(flag + "=") for a in supplied)
                return (
                    current
                    if explicit
                    else experiment.get(name, current if current is not None else default)
                )

            options = {
                "limit": chosen("limit", args.limit, None),
                "seed": chosen("seed", args.seed, 42),
                "layout": chosen("layout", args.layout, "auto"),
                "split": chosen("split", args.split, None),
                "content_type": chosen("content_type", args.content_type, None),
                "base_url": args.base_url,
                "timeout": args.timeout,
                "settings": config["models"],
            }
            data = Path(chosen("data", args.data, Path("data")))
            profiles = (
                [args.preprocess]
                if args.preprocess
                else experiment.get("preprocessing", [experiment.get("preprocess", "original")])
            )
            failed = False
            for profile in profiles:
                if profile not in {"original", "enhanced"}:
                    raise ValueError("Unknown preprocessing profile")
                if args.dry_run:
                    print(
                        json.dumps(preview(data, selected, preprocess=profile, **options), indent=2)
                    )
                    continue
                run_dir = run(
                    data,
                    selected,
                    args.output,
                    preprocess=profile,
                    num_predict=args.num_predict,
                    warmup=not args.no_warmup and experiment.get("warmup", True),
                    costs=config["costs"],
                    **options,
                )
                print(f"Run: {run_dir}")
                _report(run_dir)
                failed |= any(
                    r["status"] not in {"success", "unsupported"}
                    for r in read_records(run_dir / "results.jsonl")
                )
                failed |= (
                    json.loads((run_dir / "manifest.json").read_text()).get("status") == "paused"
                )
            return 2 if failed else 0
        elif args.command == "resume":
            manifest = json.loads((args.run / "manifest.json").read_text())
            if manifest.get("schema_version", 1) == 2:
                from .engine import resume as resume_new

                run_dir = resume_new(args.run, retry_failed=args.retry_failed)
            else:
                if args.retry_failed:
                    raise ValueError("--retry-failed is supported for schema-v2 runs")
                run_dir = resume_benchmark(args.run)
            _report(run_dir)
            if (
                any(
                    r["status"] not in {"success", "unsupported"}
                    for r in read_records(run_dir / "results.jsonl")
                )
                or json.loads((run_dir / "manifest.json").read_text()).get("status") == "paused"
            ):
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
                        if row.get("content_type") in {"equation", "math"}:
                            from .research import score_equation

                            row["metrics"] = score_equation(row["prediction"], row["reference"])
                        else:
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
