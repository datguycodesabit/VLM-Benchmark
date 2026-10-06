"""Command-line interface with explicitly selected local and cloud engines."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import sys
from pathlib import Path

from .export import export_run
from .metrics import score, summarize
from .runner import _digest, _lock, read_records, resume_benchmark


def parser():
    root = argparse.ArgumentParser(
        prog="vlm-bench",
        description="Benchmark handwriting recognition across local and cloud models.",
    )
    root.add_argument(
        "--json",
        action="store_true",
        help="Emit one machine-readable JSON object; progress and human text go to stderr",
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
    run.add_argument("--data", type=Path)
    run.add_argument("--prepared", type=Path, help="Use an immutable prepared benchmark")
    run.add_argument("--models", nargs="+")
    run.add_argument("--output", type=Path, default=Path("runs"))
    run.add_argument("--limit", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--base-url", default="http://localhost:11434")
    run.add_argument("--timeout", type=float, default=300)
    run.add_argument("--num-predict", type=int, default=4096)
    run.add_argument("--max-retries", type=_nonnegative_int_argument, default=None)
    run.add_argument("--concurrency", type=_positive_int_argument, default=None)
    run.add_argument("--max-requests", type=_positive_int_argument, default=None)
    run.add_argument("--max-spend-usd", type=_positive_float_argument, default=None)
    run.add_argument("--cache-dir", type=Path)
    run.add_argument("--no-warmup", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--config", type=Path)
    run.add_argument(
        "--strict-research",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require verified references, test labels, source documents, and protocol integrity",
    )
    run.add_argument(
        "--protocol",
        choices=["document-disjoint", "writer-disjoint"],
        default=None,
        help="Research split protocol to record and validate (default: document-disjoint)",
    )
    run.add_argument(
        "--formula-rendering",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable optional rendered-formula similarity scoring",
    )
    _dataset_options(run)
    suite = sub.add_parser(
        "suite", help="Preview or execute named experiment conditions and repetitions"
    )
    suite.add_argument("--config", type=Path, required=True)
    suite.add_argument("--output", type=Path, required=True)
    suite.add_argument(
        "--dry-run", action="store_true", help="Preview every condition and repetition"
    )
    suite.add_argument("--resume", action="store_true", help="Continue an existing suite output")
    suite.add_argument("--max-retries", type=_nonnegative_int_argument, default=None)
    suite.add_argument("--concurrency", type=_positive_int_argument, default=None)
    suite.add_argument("--max-requests", type=_positive_int_argument, default=None)
    suite.add_argument("--max-spend-usd", type=_positive_float_argument, default=None)
    suite.add_argument("--cache-dir", type=Path)
    imported = sub.add_parser(
        "import", help="Score saved external predictions against a prepared benchmark"
    )
    imported.add_argument("--prepared", type=Path, required=True)
    imported.add_argument("--predictions", type=Path, required=True)
    imported.add_argument("--output", type=Path, required=True)
    imported.add_argument("--system", required=True, help="External OCR/VLM system identity")
    provenance = imported.add_mutually_exclusive_group(required=True)
    provenance.add_argument("--provenance", help="Short description of the external run")
    provenance.add_argument("--provenance-file", type=Path, help="JSON provenance sidecar")
    imported.add_argument(
        "--formula-rendering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Score equations with the optional pinned MathText renderer",
    )
    resume = sub.add_parser(
        "resume", help="Continue unrecorded samples with unchanged data and models"
    )
    resume.add_argument("--run", type=Path, required=True)
    resume.add_argument("--retry-failed", action="store_true")
    resume.add_argument("--max-requests", type=_positive_int_argument, default=None)
    resume.add_argument("--max-spend-usd", type=_positive_float_argument, default=None)
    inspect = sub.add_parser("inspect", help="Review the worst saved transcription results")
    inspect.add_argument("--run", type=Path, required=True)
    inspect.add_argument("--worst", type=_positive_int_argument, default=20)
    inspect.add_argument(
        "--output", type=Path, help="HTML output path (default: RUN/inspection.html)"
    )
    report = sub.add_parser("report", help="Summarize saved results by sample metadata")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument(
        "--group-by",
        choices=["writer_id", "difficulty", "source_document", "sample_type"],
        required=True,
    )
    status = sub.add_parser(
        "status", help="Read progress, coverage, limits, and usage for a saved run"
    )
    status.add_argument("--run", type=Path, required=True)
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
    check.add_argument("--data", type=Path)
    check.add_argument("--prepared", type=Path)
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
    split_cmd.add_argument(
        "--protocol",
        choices=["document-disjoint", "writer-disjoint"],
        default="document-disjoint",
        help="Group samples by source document or writer when assigning splits",
    )
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
    compare = sub.add_parser("compare", help="Compare separate runs on the same frozen benchmark")
    compare.add_argument("--runs", type=Path, nargs="+", required=True)
    compare.add_argument("--output", type=Path, required=True)
    return root


def _dataset_options(command):
    command.add_argument(
        "--layout", choices=["auto", "paired", "iam-words", "iam-lines", "iam-forms"], default=None
    )
    command.add_argument("--preprocess", choices=["original", "enhanced"], default=None)
    command.add_argument("--split", choices=["train", "validation", "test"])
    command.add_argument("--content-type", choices=["prose", "equation", "word"])


def _positive_int_argument(value):
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _nonnegative_int_argument(value):
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return result


def _positive_float_argument(value):
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if result <= 0 or not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def _report(run_dir):
    rows = read_records(run_dir / "results.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") == 2:
        from .research import research_reports

        research = research_reports(
            rows, dict(manifest, warmups=read_records(run_dir / "warmups.jsonl"))
        )
        evaluation = research.get("evaluation", {})
        if not evaluation:
            evaluation = {
                key: manifest[key]
                for key in (
                    "strict_research",
                    "protocol",
                    "research_protocol_version",
                    "research",
                    "source_audit",
                    "validation_scope",
                    "research_validation_boundary",
                    "formula_rendering",
                    "task_metrics_version",
                    "formula_renderer",
                )
                if key in manifest
            }
        if evaluation:
            print(
                "Research eligibility: "
                f"{'strict' if evaluation.get('strict_research') else 'exploratory'}; "
                f"protocol {evaluation.get('protocol', 'unspecified')}"
            )
            if evaluation.get("validation_scope") or evaluation.get("research_validation_boundary"):
                print(
                    "Validation scope: "
                    f"{evaluation.get('validation_scope', evaluation.get('research_validation_boundary'))}"
                )
            if "formula_rendering" in evaluation:
                print(
                    f"Formula rendering: {'enabled' if evaluation['formula_rendering'] else 'disabled'}"
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


class _StderrTee:
    def __init__(self, capture, destination):
        self.capture = capture
        self.destination = destination

    def write(self, value):
        self.capture.write(value)
        self.destination.write(value)
        self.destination.flush()
        return len(value)

    def flush(self):
        self.capture.flush()
        self.destination.flush()


def _json_error_message(text):
    for line in reversed(text.splitlines()):
        if line.startswith("Error:"):
            return line.removeprefix("Error:").strip()
    return None


def _dataset_check_result(args):
    """Perform one dataset audit and return its complete structured report."""
    if args.prepared is not None:
        from .snapshot import load

        if args.data is not None or any(
            getattr(args, key) is not None
            for key in ("layout", "preprocess", "split", "content_type")
        ):
            raise ValueError("dataset check --prepared cannot be combined with source-data options")
        snapshot = load(args.prepared)
        return {
            "valid": True,
            "manifest_version": snapshot["manifest_version"],
            "benchmark_fingerprint": snapshot["benchmark_fingerprint"],
            "sample_count": len(snapshot["samples"]),
        }

    from .dataset import check_dataset

    return check_dataset(args.data or Path("data"), layout=args.layout or "auto")


def _safe_auth_status(value):
    """Expose useful account state while excluding stable IDs and credentials."""
    if not isinstance(value, dict):
        return {"provider": "chatgpt", "authenticated": False, "accounts": []}
    accounts = []
    for account in value.get("accounts", []):
        if not isinstance(account, dict):
            continue
        safe = {
            key: account[key]
            for key in ("provider", "email", "expires_at", "connected")
            if key in account
        }
        accounts.append(safe)
    return {
        "provider": value.get("provider", "chatgpt"),
        "authenticated": any(account.get("connected") is True for account in accounts),
        "accounts": accounts,
    }


def _json_cli(arguments):
    clean = [argument for argument in arguments if argument != "--json"]
    parser_stdout = io.StringIO()
    parser_stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(parser_stdout), contextlib.redirect_stderr(parser_stderr):
            args = parser().parse_args(clean)
    except SystemExit as exc:
        detail = parser_stderr.getvalue()
        if detail:
            sys.stderr.write(detail)
        output = parser_stdout.getvalue()
        payload = {
            "command": _command_hint(clean),
            "status": "ok" if exc.code == 0 else "error",
            "exit_code": int(exc.code or 0),
        }
        if output.strip():
            payload["messages"] = output.splitlines()
        if detail.strip() and exc.code:
            payload["error"] = detail.strip().splitlines()[-1]
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        return int(exc.code or 0)

    # Dataset audits can be expensive. Return the exact audit result used to
    # determine the exit code, including the full result for invalid datasets.
    if args.command == "dataset" and args.dataset_command == "check":
        try:
            result = _dataset_check_result(args)
            exit_code = 0 if result.get("valid") else 1
            payload = {
                "command": "dataset",
                "status": "ok" if exit_code == 0 else "error",
                "exit_code": exit_code,
                "result": result,
            }
            if exit_code:
                payload["error"] = "Dataset check failed; resolve issues before splitting"
        except Exception as exc:
            payload = {
                "command": "dataset",
                "status": "error",
                "exit_code": 1,
                "error": str(exc),
            }
            exit_code = 1
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        return exit_code

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    sensitive_auth = args.command == "auth"
    target_stdout = captured_stdout if sensitive_auth else _StderrTee(captured_stdout, sys.stderr)
    with contextlib.redirect_stdout(target_stdout), contextlib.redirect_stderr(captured_stderr):
        exit_code = main(clean)
    stderr_text = captured_stderr.getvalue()
    if stderr_text and not sensitive_auth:
        sys.stderr.write(stderr_text)
    output_text = captured_stdout.getvalue()
    command_status = (
        "ok"
        if exit_code == 0
        else "incomplete"
        if exit_code == 2
        else "interrupted"
        if exit_code == 130
        else "error"
    )
    payload = {
        "command": args.command,
        "status": command_status,
        "exit_code": exit_code,
    }
    error = _json_error_message(stderr_text)
    if error and not sensitive_auth:
        payload["error"] = error

    messages = [line for line in output_text.splitlines() if line.strip()]
    try:
        decoded = json.loads(output_text) if output_text.strip() else None
    except ValueError:
        decoded = None
    if sensitive_auth:
        payload["action"] = args.auth_command
        if args.auth_command == "status" and exit_code == 0:
            payload["result"] = _safe_auth_status(decoded)
        elif exit_code != 0:
            payload["error"] = "Authentication action failed"
    elif decoded is not None:
        payload["result"] = decoded
    elif messages:
        payload["messages"] = messages

    saved_paths = list(
        dict.fromkeys(
            line.removeprefix("Saved:").strip()
            for line in messages
            if line.startswith("Saved:")
            and line.removeprefix("Saved:").strip()
            and Path(line.removeprefix("Saved:").strip()).is_file()
        )
    )
    if saved_paths:
        payload["report_paths"] = saved_paths
    run_paths = []
    for line in messages:
        match = re.match(r"(?:Run|Imported run|Run directory):\s*(.+)$", line)
        if match:
            run_paths.append(str(Path(match.group(1).strip()).resolve()))
    run_paths = list(dict.fromkeys(run_paths))
    if args.command == "resume" and args.run.is_dir():
        run_paths.append(str(args.run.resolve()))
        run_paths = list(dict.fromkeys(run_paths))
    if run_paths:
        payload["run_directory"] = run_paths[-1] if len(run_paths) == 1 else run_paths
    if args.command == "suite":
        payload["suite_directory"] = str(args.output.resolve())
        suite_report = args.output / "suite-report.json"
        if not args.dry_run and suite_report.is_file():
            payload["report_paths"] = [str(suite_report.resolve())]
            if exit_code == 0:
                payload["result"] = json.loads(suite_report.read_text(encoding="utf-8"))
    elif args.command == "prepare":
        payload["output_directory"] = str((args.output or args.data / "prepared").resolve())
    elif args.command == "compare":
        payload["comparison_directory"] = str(args.output.resolve())
        comparison_files = [
            args.output / name
            for name in ("comparison.json", "comparison.csv", "paired.csv", "costs.csv")
            if (args.output / name).is_file()
        ]
        if comparison_files:
            payload["report_paths"] = [str(path.resolve()) for path in comparison_files]
    elif args.command == "inspect":
        inspection_path = args.output or args.run / "inspection.html"
        if inspection_path.is_file():
            payload["report_paths"] = [str(inspection_path.resolve())]
    elif args.command == "resume":
        if saved_paths:
            payload["report_paths"] = saved_paths
    elif args.command == "export":
        payload["run_directory"] = str(args.run.resolve())
        export_names = {
            "xlsx": "results.xlsx",
            "csv": ("summary.csv", "samples.csv"),
            "jsonl": "results.jsonl",
        }
        exported = []
        for file_format in args.formats:
            names = export_names[file_format]
            for name in names if isinstance(names, tuple) else (names,):
                path = args.run / name
                if path.is_file():
                    exported.append(path)
        if exported:
            payload["report_paths"] = list(dict.fromkeys(str(path.resolve()) for path in exported))
    elif args.command == "rescore":
        payload["run_directory"] = str(args.run.resolve())
    elif args.command == "report":
        payload["run_directory"] = str(args.run.resolve())
    elif args.command == "status":
        payload["run_directory"] = str(args.run.resolve())
    elif args.command == "dataset" and args.dataset_command == "split":
        payload["output_path"] = str(args.output.resolve())
    return_code = int(exit_code)
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    return return_code


def _command_hint(arguments):
    known = {
        "prepare",
        "models",
        "run",
        "suite",
        "import",
        "resume",
        "inspect",
        "report",
        "status",
        "auth",
        "doctor",
        "dataset",
        "export",
        "rescore",
        "compare",
    }
    return next((value for value in arguments if value in known), "vlm-bench")


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--json" in arguments:
        return _json_cli(arguments)
    args = parser().parse_args(arguments)
    try:
        if args.command == "prepare":
            output = args.output or args.data / "prepared"
            from .snapshot import freeze

            snapshot = freeze(
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
            print(f"Prepared {len(snapshot['samples'])} samples in {output.resolve()}")
            print(f"Benchmark: {snapshot['benchmark_fingerprint']}")
            print("Review the references and saved crops before benchmarking.")
        elif args.command == "auth":
            from . import auth

            print(json.dumps(getattr(auth, args.auth_command)(), indent=2))
        elif args.command == "dataset":
            from .dataset import split_dataset

            if args.dataset_command == "check":
                report = _dataset_check_result(args)
                if args.json:
                    print(json.dumps(report, indent=2, ensure_ascii=False))
                elif args.prepared is not None:
                    print(json.dumps(report, indent=2, ensure_ascii=False))
                else:
                    print(f"Layout: {report['layout']} | Valid: {report['valid']}")
                    print(json.dumps(report.get("counts", {}), indent=2))
                    for issue in report.get("issues", [])[:20]:
                        print(json.dumps(issue, ensure_ascii=False))
                    findings = report.get("findings", [])
                    for finding in findings[:20]:
                        print("Finding: " + json.dumps(finding, ensure_ascii=False))
                    if len(findings) > 20:
                        print(
                            f"Additional findings: {len(findings) - 20}; use --json for the complete audit"
                        )
                    print(
                        f"Excluded: {len(report.get('excluded', []))}; use --json for the complete audit"
                    )
                return 0 if report["valid"] else 1
            from .dataset import check_dataset

            report = check_dataset(args.data, layout=args.layout or "auto")
            if not report["valid"]:
                raise ValueError("Dataset check failed; resolve issues before splitting")
            records = split_dataset(report["samples"], seed=args.seed, protocol=args.protocol)
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
                flags = {flag}
                if name in {"strict_research", "formula_rendering"}:
                    flags.add("--no-" + name.replace("_", "-"))
                explicit = any(
                    a in flags or any(a.startswith(item + "=") for item in flags) for a in supplied
                )
                return (
                    current
                    if explicit
                    else experiment.get(name, current if current is not None else default)
                )

            source_keys = (
                "data",
                "limit",
                "seed",
                "layout",
                "split",
                "content_type",
                "preprocess",
                "preprocessing",
            )
            prepared = args.prepared or experiment.get("prepared")
            explicit_data = any(a == "--data" or a.startswith("--data=") for a in supplied)
            if explicit_data and args.prepared is None:
                prepared = None  # Explicit source input overrides configured snapshot.
            if prepared is not None:
                explicit_selection = [
                    key
                    for key in source_keys
                    if any(
                        a == "--" + key.replace("_", "-")
                        or a.startswith("--" + key.replace("_", "-") + "=")
                        for a in supplied
                    )
                ]
                if explicit_selection:
                    raise ValueError(
                        "--prepared cannot be combined with " + ", ".join(explicit_selection)
                    )
                data = None
                profiles = [None]
                selection = {"prepared": Path(prepared)}
            else:
                data = Path(chosen("data", args.data, Path("data")))
                profiles = (
                    [args.preprocess]
                    if args.preprocess
                    else experiment.get("preprocessing", [experiment.get("preprocess", "original")])
                )
                selection = {
                    "limit": chosen("limit", args.limit, None),
                    "seed": chosen("seed", args.seed, 42),
                    "layout": chosen("layout", args.layout, "auto"),
                    "split": chosen("split", args.split, None),
                    "content_type": chosen("content_type", args.content_type, None),
                }
            from .config import validate_settings

            selected_settings = validate_settings(selected, config["models"])
            if any(a == "--num-predict" or a.startswith("--num-predict=") for a in supplied):
                selected_settings = {
                    model: dict(selected_settings.get(model, {}), num_predict=args.num_predict)
                    for model in selected
                }
            options = dict(
                selection,
                base_url=args.base_url,
                timeout=args.timeout,
                settings=selected_settings,
                num_predict=args.num_predict,
                max_retries=chosen("max_retries", args.max_retries, 2),
                concurrency=chosen("concurrency", args.concurrency, 1),
                max_requests=chosen("max_requests", args.max_requests, None),
                max_spend_usd=chosen("max_spend_usd", args.max_spend_usd, None),
                cache_dir=(
                    args.cache_dir
                    if any(
                        item == "--cache-dir" or item.startswith("--cache-dir=")
                        for item in supplied
                    )
                    else experiment.get("cache_dir")
                ),
                strict_research=chosen("strict_research", args.strict_research, False),
                protocol=chosen("protocol", args.protocol, "document-disjoint"),
                formula_rendering=chosen("formula_rendering", args.formula_rendering, False),
            )
            failed = False
            for profile in profiles:
                if profile is not None:
                    if profile not in {"original", "enhanced"}:
                        raise ValueError("Unknown preprocessing profile")
                    options["preprocess"] = profile
                if args.dry_run:
                    print(
                        json.dumps(
                            preview(data, selected, costs=config["costs"], **options),
                            indent=2,
                        )
                    )
                    continue
                run_dir = run(
                    data,
                    selected,
                    args.output,
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
        elif args.command == "suite":
            from .suite import run_suite

            report = run_suite(
                args.config,
                args.output,
                dry_run=args.dry_run,
                resume=args.resume,
                max_retries=args.max_retries,
                concurrency=args.concurrency,
                max_requests=args.max_requests,
                max_spend_usd=args.max_spend_usd,
                cache_dir=args.cache_dir,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            if not args.dry_run:
                failed = any(
                    condition["status"] != "complete" for condition in report["conditions"].values()
                )
                if failed:
                    return 2
        elif args.command == "import":
            from .import_predictions import import_predictions

            run_dir = import_predictions(
                args.prepared,
                args.predictions,
                args.output,
                system=args.system,
                provenance=args.provenance,
                provenance_file=args.provenance_file,
                formula_rendering=args.formula_rendering,
            )
            print(f"Imported run: {run_dir}")
            _report(run_dir)
        elif args.command == "inspect":
            from .inspection import write_inspection_html

            output, report = write_inspection_html(args.run, args.output, worst=args.worst)
            print(
                f"{report['selected_count']} model/sample results selected from "
                f"{report['eligible_result_count']} eligible results across "
                f"{report['eligible_sample_count']} eligible samples; "
                f"{report['excluded_sample_count']} samples excluded by research eligibility."
            )
            print(f"{'Model':<32} {'Sample':<24} {'Status':<12} {'CER':>8} {'S/D/I':>11} Flags")
            for row in report["results"]:
                cer = "N/A" if row["cer"] is None else f"{row['cer']:.2%}"
                counts = row["alignment"].get("counts") or {}
                edits = (
                    f"{counts.get('substitutions', 0)}/"
                    f"{counts.get('deletions', 0)}/"
                    f"{counts.get('insertions', 0)}"
                )
                flags = ",".join(row["flags"]) or "none"
                print(
                    f"{row['model'][:32]:<32} {row['sample_id'][:24]:<24} "
                    f"{row['status']:<12} {cer:>8} {edits:>11} {flags}"
                )
            print(f"Saved inspection report: {output}")
        elif args.command == "report":
            from .inspection import group_report

            report = group_report(args.run, args.group_by)
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        elif args.command == "compare":
            from .comparison import compare

            report = compare(args.runs, args.output)
            for track, result in report["tracks"].items():
                if not result["eligible_sample_count"]:
                    continue
                print(f"{track}: {result['eligible_sample_count']} eligible samples")
                print(
                    f"{'Rank':>4} {'Model / run':<56} {'CER':>9} {'WER':>9} "
                    f"{'Exact':>9} {'p50 sec':>9} {'Coverage':>9} Status"
                )
                for row in result["model_results"]:
                    values = [
                        "N/A" if row.get(key) is None else f"{row[key]:.2%}"
                        for key in ("cer", "wer", "exact_match_rate", "sample_coverage")
                    ]
                    median = row.get("median_latency_seconds")
                    latency = "N/A" if median is None else f"{median:.2f}"
                    rank = row.get("rank") or "—"
                    print(
                        f"{rank:>4} {row['model']:<56} {values[0]:>9} {values[1]:>9} "
                        f"{values[2]:>9} {latency:>9} {values[3]:>9} {row['status']}"
                    )
            print(f"Saved comparison: {args.output.resolve()}")
        elif args.command == "resume":
            manifest = json.loads((args.run / "manifest.json").read_text())
            if manifest.get("schema_version", 1) == 2:
                from .engine import resume as resume_new

                run_dir = resume_new(
                    args.run,
                    retry_failed=args.retry_failed,
                    max_requests=args.max_requests,
                    max_spend_usd=args.max_spend_usd,
                )
            else:
                if args.max_requests is not None or args.max_spend_usd is not None:
                    raise ValueError("Execution limit overrides require a schema-v2 run")
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
        elif args.command == "status":
            from .status import run_status

            print(json.dumps(run_status(args.run), ensure_ascii=False, indent=2, allow_nan=False))
        elif args.command == "export":
            with _lock(args.run):
                for output in export_run(args.run, args.formats):
                    print(output)
        elif args.command == "rescore":
            with _lock(args.run):
                rows = read_records(args.run / "results.jsonl")
                try:
                    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise ValueError(f"Could not read run manifest: {exc}") from exc
                if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
                    raise ValueError(
                        "Rescoring requires the frozen sample list in the run manifest"
                    )
                schema_version = manifest.get("schema_version", 1)
                if schema_version == 2:
                    integrity = manifest.get("integrity")
                    immutable = {
                        key: value
                        for key, value in manifest.items()
                        if key not in {"integrity", "status", "updated_at"}
                    }
                    if not isinstance(integrity, str) or _digest(immutable) != integrity:
                        raise ValueError("Run manifest integrity check failed; refusing to rescore")
                    try:
                        from .snapshot import fingerprint

                        computed_fingerprint = fingerprint(
                            manifest["samples"], manifest.get("preprocess")
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Run manifest contains an invalid frozen snapshot: {exc}"
                        ) from exc
                    if computed_fingerprint != manifest.get("benchmark_fingerprint"):
                        raise ValueError(
                            "Frozen benchmark fingerprint does not match the run manifest; refusing to rescore"
                        )
                frozen_samples = {}
                for sample in manifest["samples"]:
                    if not isinstance(sample, dict) or not isinstance(sample.get("id"), str):
                        raise ValueError("Run manifest contains a malformed frozen sample")
                    if sample["id"] in frozen_samples:
                        raise ValueError(f"Duplicate frozen sample ID {sample['id']!r}")
                    frozen_samples[sample["id"]] = sample
                expected_models = manifest.get("models")
                if schema_version == 2 and (
                    not isinstance(expected_models, list)
                    or not all(isinstance(model, str) for model in expected_models)
                ):
                    raise ValueError("Schema-v2 run manifest has a malformed model list")
                if isinstance(expected_models, list):
                    expected_model_set = set(expected_models)
                    seen_pairs = set()
                    for index, row in enumerate(rows, start=1):
                        model = row.get("model")
                        sample_id = row.get("sample_id")
                        pair = (model, sample_id)
                        if (
                            not isinstance(model, str)
                            or model not in expected_model_set
                            or not isinstance(sample_id, str)
                            or sample_id not in frozen_samples
                        ):
                            raise ValueError(
                                f"Result row {index} has an unknown model/sample pair {pair!r}"
                            )
                        if pair in seen_pairs:
                            raise ValueError(f"Duplicate model/sample result pair {pair!r}")
                        seen_pairs.add(pair)
                formula_rendering = manifest.get("formula_rendering", False)
                if not isinstance(formula_rendering, bool):
                    raise ValueError("Run manifest formula_rendering must be a boolean")
                from .task_metrics import preflight_renderer, score_task

                if formula_rendering:
                    preflight_renderer()
                for row in rows:
                    if row["status"] == "success":
                        sample_id = row.get("sample_id")
                        sample = frozen_samples.get(sample_id)
                        if sample is None:
                            raise ValueError(
                                f"Result references unknown frozen sample {sample_id!r}"
                            )
                        reference = sample.get("reference")
                        prediction = row.get("prediction")
                        if not isinstance(reference, str) or not isinstance(prediction, str):
                            raise ValueError(
                                f"Cannot rescore malformed prediction/reference for sample {sample_id!r}"
                            )
                        row["reference"] = reference
                        sample_metadata = sample.get("metadata")
                        content_type = sample.get("content_type")
                        if content_type is None and isinstance(sample_metadata, dict):
                            content_type = sample_metadata.get("content_type")
                        if content_type in {"equation", "math"}:
                            from .research import score_equation

                            metrics = score_equation(prediction, reference)
                        else:
                            metrics = score(prediction, reference)
                        metrics["task_metrics"] = score_task(
                            prediction,
                            reference,
                            sample,
                            formula_rendering=formula_rendering,
                        )
                        row["metrics"] = metrics
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
