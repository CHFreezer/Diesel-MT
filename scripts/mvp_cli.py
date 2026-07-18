"""Unified CLI for Diesel-MT student training, validation, and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from artifact_io import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]


def _print(value: Any) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
    )


def _train(args: argparse.Namespace) -> dict[str, Any]:
    from mvp_training import compare_training_runs, run_training

    report = run_training(
        config_path=args.config.resolve(),
        repository_root=ROOT,
        output_dir=args.output.resolve() if args.output else None,
        dry_run=args.dry_run,
        checkpoint_root=(
            args.checkpoint_root.resolve() if args.checkpoint_root else None
        ),
        resume_from=args.resume_from.resolve() if args.resume_from else None,
        stop_after_optimizer_steps=args.stop_after_optimizer_step,
        input_cache_root=(
            args.input_cache_root.resolve() if args.input_cache_root else None
        ),
    )
    if args.compare_to is not None and not args.dry_run:
        baseline = json.loads(args.compare_to.resolve().read_text(encoding="utf-8"))
        report = {
            **report,
            "determinism_replay": compare_training_runs(baseline, report),
        }
    if args.report is not None and not args.dry_run:
        atomic_write_json(
            args.report.resolve(), report, sort_keys=True, allow_nan=True
        )
    return report


def _validate_student(args: argparse.Namespace) -> dict[str, Any]:
    from mvp_student import EncodingPolicy, run_td09_acceptance

    report = run_td09_acceptance(
        repository_root=ROOT,
        student_config_path=args.config.resolve(),
        fixture_path=args.fixture.resolve(),
        checkpoint_output=args.checkpoint_output.resolve(),
        policy=EncodingPolicy(
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        ),
    )
    atomic_write_json(args.report.resolve(), report, allow_nan=True)
    return report


def _validate_resume(args: argparse.Namespace) -> dict[str, Any]:
    from mvp_training import validate_resume_equivalence

    return validate_resume_equivalence(
        config_path=args.config.resolve(),
        repository_root=ROOT,
        runtime_root=args.runtime_root.resolve(),
        report_path=args.report.resolve(),
    )


def _validate_m1(args: argparse.Namespace) -> dict[str, Any]:
    from mvp_m1 import run_m1_acceptance

    runtime_root = args.runtime_root.resolve()
    if runtime_root.exists():
        raise RuntimeError(f"M1 runtime root already exists: {runtime_root}")
    report = run_m1_acceptance(
        repository_root=ROOT,
        acceptance_path=args.acceptance_config.resolve(),
        runtime_root=runtime_root,
    )
    atomic_write_json(args.report.resolve(), report, allow_nan=True)
    return report


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from mvp_evaluation import evaluate_checkpoint, publish_evaluation

    summary, samples = evaluate_checkpoint(
        repository_root=ROOT,
        evaluation_config_path=args.config.resolve(),
        checkpoint=args.checkpoint,
        split=args.split,
        allow_test=args.allow_test,
    )
    manifest = publish_evaluation(args.output_directory.resolve(), summary, samples)
    result = {
        **summary,
        "publication": {
            "path": str(args.output_directory.resolve()),
            "manifest": manifest,
        },
    }
    if args.report:
        atomic_write_json(args.report.resolve(), result)
    return result


def _benchmark_resources(args: argparse.Namespace) -> dict[str, Any]:
    from mvp_resource_benchmark import benchmark_candidates

    return benchmark_candidates(
        repository_root=ROOT,
        meta_path=args.config.resolve(),
        runtime_root=args.runtime_root.resolve(),
        report_path=args.report.resolve(),
    )


def _validate_resources(args: argparse.Namespace) -> dict[str, Any]:
    from mvp_resource_profile import build_td14_evidence

    return build_td14_evidence(
        repository_root=ROOT,
        profile_path=args.profile.resolve(),
        benchmark_path=args.benchmark.resolve(),
        soak_config_path=args.soak_config.resolve(),
        soak_report_path=args.soak_report.resolve(),
        resume_report_path=args.resume_report.resolve(),
        output_path=args.output.resolve(),
    )


def _build_ab(args: argparse.Namespace) -> dict[str, Any]:
    from mvp_distillation_ab import build_ab_cohort

    return build_ab_cohort(repository_root=ROOT, config_path=args.config.resolve())


def _validate_ab(args: argparse.Namespace) -> dict[str, Any]:
    from mvp_distillation_ab import validate_ab_release

    return validate_ab_release(
        repository_root=ROOT,
        config_path=args.config.resolve(),
        human_recipe_path=args.human_recipe.resolve(),
        distilled_recipe_path=args.distilled_recipe.resolve(),
        report_path=args.report.resolve(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Run or dry-run bounded training")
    train.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/mvp_training_td10_smoke.yaml",
    )
    train.add_argument("--output", type=Path)
    train.add_argument("--report", type=Path)
    train.add_argument("--compare-to", type=Path)
    train.add_argument("--checkpoint-root", type=Path)
    train.add_argument("--resume-from", type=Path)
    train.add_argument("--input-cache-root", type=Path)
    train.add_argument("--stop-after-optimizer-step", type=int)
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(handler=_train)

    student = commands.add_parser(
        "validate-student", help="Run the TD-09 student acceptance gate"
    )
    student.add_argument(
        "--config", type=Path, default=ROOT / "configs/mvp_e8_d2_v48k.yaml"
    )
    student.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "data/model/corpus/mvp/finalized/train.jsonl",
    )
    student.add_argument(
        "--checkpoint-output",
        type=Path,
        default=ROOT / "artifacts/model-training/runtime/student-checkpoint",
    )
    student.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "artifacts/model-training/reports/student/encoding-validation.json",
    )
    student.add_argument("--max-source-length", type=int, default=128)
    student.add_argument("--max-target-length", type=int, default=128)
    student.set_defaults(handler=_validate_student)

    resume = commands.add_parser(
        "validate-resume", help="Run the TD-11 exact-resume acceptance gate"
    )
    resume.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/mvp_training_td10_smoke.yaml",
    )
    resume.add_argument("--runtime-root", type=Path, required=True)
    resume.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "artifacts/model-training/reports/student/checkpoint-resume.json",
    )
    resume.set_defaults(handler=_validate_resume)

    m1 = commands.add_parser(
        "validate-m1", help="Run the TD-12 M1 overfit and resume gate"
    )
    m1.add_argument(
        "--acceptance-config",
        type=Path,
        default=ROOT / "configs/mvp_m1_acceptance.yaml",
    )
    m1.add_argument("--runtime-root", type=Path, required=True)
    m1.add_argument(
        "--report",
        type=Path,
        default=ROOT / "artifacts/model-training/reports/student/m1-overfit.json",
    )
    m1.set_defaults(handler=_validate_m1)

    evaluate = commands.add_parser(
        "evaluate", help="Run the frozen TD-13 evaluator"
    )
    evaluate.add_argument(
        "--config", type=Path, default=ROOT / "configs/mvp_evaluation.yaml"
    )
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluate.add_argument("--allow-test", action="store_true")
    evaluate.add_argument("--output-directory", type=Path, required=True)
    evaluate.add_argument("--report", type=Path)
    evaluate.set_defaults(handler=_evaluate)

    benchmark = commands.add_parser(
        "benchmark-resources", help="Run TD-14 resource candidates"
    )
    benchmark.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/mvp_training_td14_candidates.yaml",
    )
    benchmark.add_argument("--runtime-root", type=Path, required=True)
    benchmark.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "artifacts/model-training/reports/m2/resources/candidate-benchmark.json",
    )
    benchmark.set_defaults(handler=_benchmark_resources)

    resources = commands.add_parser(
        "validate-resources", help="Publish TD-14 selected-profile evidence"
    )
    resources.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs/mvp_training_m2_profile.yaml",
    )
    resources.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT
        / "artifacts/model-training/reports/m2/resources/candidate-benchmark.json",
    )
    resources.add_argument(
        "--soak-config",
        type=Path,
        default=ROOT / "configs/mvp_training_td14_soak.yaml",
    )
    resources.add_argument(
        "--soak-report",
        type=Path,
        default=ROOT / "artifacts/model-training/reports/m2/resources/soak-run.json",
    )
    resources.add_argument(
        "--resume-report",
        type=Path,
        default=ROOT / "artifacts/model-training/reports/m2/resources/resume-probe.json",
    )
    resources.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/model-training/reports/m2/resources/profile.json",
    )
    resources.set_defaults(handler=_validate_resources)

    build_ab = commands.add_parser(
        "build-ab", help="Build the TD-15 source-matched cohort"
    )
    build_ab.add_argument(
        "--config", type=Path, default=ROOT / "configs/mvp_distillation_ab.yaml"
    )
    build_ab.set_defaults(handler=_build_ab)

    validate_ab = commands.add_parser(
        "validate-ab", help="Validate and publish TD-15 fairness evidence"
    )
    validate_ab.add_argument(
        "--config", type=Path, default=ROOT / "configs/mvp_distillation_ab.yaml"
    )
    validate_ab.add_argument(
        "--human-recipe",
        type=Path,
        default=ROOT / "configs/mvp_training_m2_human.yaml",
    )
    validate_ab.add_argument(
        "--distilled-recipe",
        type=Path,
        default=ROOT / "configs/mvp_training_m2_distilled.yaml",
    )
    validate_ab.add_argument(
        "--report",
        type=Path,
        default=ROOT / "artifacts/model-training/reports/m2/distillation-ab.json",
    )
    validate_ab.set_defaults(handler=_validate_ab)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _print(args.handler(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
