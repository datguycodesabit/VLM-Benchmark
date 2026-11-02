"""Prepare, run/resume, or report a controlled Qwen handwriting diagnostic study."""

import argparse
from pathlib import Path

from vlm_bench.diagnostic_study import prepare_study, report_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--filtered", type=Path, required=True)
    prepare.add_argument("--challenge-source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--challenge-size", type=int, default=30)
    run = commands.add_parser("run")
    run.add_argument("--study", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--study", type=Path, required=True)
    report.add_argument("--runs", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        plan = prepare_study(
            args.filtered, args.challenge_source, args.output, challenge_size=args.challenge_size
        )
        print(
            f"Prepared study: {args.output.resolve()} · {plan['planned_predictions']} predictions planned"
        )
    elif args.command == "run":
        from vlm_bench.diagnostic_study import _read_plan
        from vlm_bench.reporting import write_report
        from vlm_bench.suite import run_suite

        plan = _read_plan(args.study.resolve())
        import hashlib

        if (
            hashlib.sha256((args.study / "suite.toml").read_bytes()).hexdigest()
            != plan["config_sha256"]
        ):
            raise ValueError("Study configuration changed")
        result = run_suite(
            args.study / "suite.toml", args.output, resume=args.resume, dry_run=args.dry_run
        )
        if args.dry_run:
            for condition in result["conditions"]:
                preview = condition["preview"]
                print(
                    f"{condition['name']}: {preview['sample_count']} images, "
                    f"{len(preview['models'])} models, {condition['repeats']} pass(es)"
                )
        else:
            for name, condition in result["conditions"].items():
                print(
                    f"{name}: {condition['status']} "
                    f"({condition['completed_repetitions']}/{condition['planned_repetitions']} passes)"
                )
        if not args.dry_run:
            if all(row["status"] == "complete" for row in result["conditions"].values()):
                # Preserve an existing review, and generate diagnostics before
                # optional presentation exports so they survive export failures.
                if not (args.output / "diagnostics").exists():
                    report_study(args.study, args.output, args.output / "diagnostics")
                print(f"Review: {(args.output / 'diagnostics/review.html').resolve()}")
            for path in write_report(args.output):
                print(f"Saved: {path}")
    else:
        report_study(args.study, args.runs, args.output)
        print(f"Review: {(args.output / 'review.html').resolve()}")


if __name__ == "__main__":
    main()
