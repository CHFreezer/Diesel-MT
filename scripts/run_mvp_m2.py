"""Run and close the bounded TD-16 human-only/distilled M2 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mvp_m2 import (
    ARM_CONFIGS,
    M2ContractError,
    finalize_m2_arm,
    prepare_m2_arm,
    run_selected_test_once,
    select_m2_candidate,
)
from mvp_training import git_identity, run_training


ROOT = Path(__file__).resolve().parents[1]


def _common_arm(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arm", choices=tuple(ARM_CONFIGS), required=True)
    parser.add_argument(
        "--ab-config",
        type=Path,
        default=ROOT / "configs" / "mvp_distillation_ab.yaml",
    )
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_arm = subparsers.add_parser(
        "run-arm",
        help="train one frozen arm, export every checkpoint, and run dev evaluation",
    )
    _common_arm(run_arm)
    run_arm.add_argument("--run-root", type=Path)
    run_arm.add_argument("--resume-from", type=Path)
    run_arm.add_argument("--stop-after-optimizer-step", type=int)
    run_arm.add_argument("--dry-run", action="store_true")

    finalize = subparsers.add_parser(
        "finalize-arm",
        help="recover post-training HF export/dev evaluation without retraining",
    )
    _common_arm(finalize)
    finalize.add_argument("--require-complete", action="store_true")

    select = subparsers.add_parser(
        "select",
        help="freeze the unique candidate from the two completed dev-only arms",
    )
    select.add_argument(
        "--ab-config",
        type=Path,
        default=ROOT / "configs" / "mvp_distillation_ab.yaml",
    )
    select.add_argument("--human-arm-manifest", type=Path, required=True)
    select.add_argument("--distilled-arm-manifest", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)

    test = subparsers.add_parser(
        "test",
        help="consume the single formal-test run for the frozen selected candidate",
    )
    test.add_argument("--selection", type=Path, required=True)
    test.add_argument("--output-directory", type=Path, required=True)
    test.add_argument("--receipt", type=Path, required=True)
    test.add_argument("--report", type=Path)
    return parser.parse_args()


def _context(args: argparse.Namespace) -> dict:
    return prepare_m2_arm(
        repository_root=ROOT,
        ab_config_path=args.ab_config.resolve(),
        arm=args.arm,
        require_clean_git=True,
    )


def _require_clean_git() -> None:
    if git_identity(ROOT)["dirty"]:
        raise M2ContractError("formal M2 execution requires a clean Git worktree")


def main() -> int:
    args = parse_args()
    if args.command == "run-arm":
        context = _context(args)
        if args.dry_run:
            training = run_training(
                config_path=context["recipe_path"],
                repository_root=ROOT,
                output_dir=None,
                dry_run=True,
            )
            result = {
                "schema_version": 1,
                "status": "dry_run_complete",
                "arm": args.arm,
                "training": training,
                "checkpoint_identity_sha256": context[
                    "checkpoint_identity_sha256"
                ],
                "test_access": "forbidden",
            }
        else:
            if args.run_root is None:
                raise SystemExit("run-arm requires --run-root unless --dry-run is used")
            training = run_training(
                config_path=context["recipe_path"],
                repository_root=ROOT,
                output_dir=args.run_root.resolve(),
                dry_run=False,
                checkpoint_root=args.checkpoint_root.resolve(),
                resume_from=(
                    args.resume_from.resolve() if args.resume_from is not None else None
                ),
                stop_after_optimizer_steps=args.stop_after_optimizer_step,
            )
            arm = finalize_m2_arm(
                repository_root=ROOT,
                context=context,
                checkpoint_root=args.checkpoint_root,
                candidate_root=args.candidate_root,
                evaluation_root=args.evaluation_root,
                require_complete=training["status"] == "complete",
            )
            result = {
                "schema_version": 1,
                "status": arm["status"],
                "arm": args.arm,
                "training": training,
                "finalization": arm,
            }
    elif args.command == "finalize-arm":
        context = _context(args)
        result = finalize_m2_arm(
            repository_root=ROOT,
            context=context,
            checkpoint_root=args.checkpoint_root,
            candidate_root=args.candidate_root,
            evaluation_root=args.evaluation_root,
            require_complete=args.require_complete,
        )
    elif args.command == "select":
        _require_clean_git()
        result = select_m2_candidate(
            ab_config_path=args.ab_config.resolve(),
            human_arm_manifest_path=args.human_arm_manifest.resolve(),
            distilled_arm_manifest_path=args.distilled_arm_manifest.resolve(),
            output_path=args.output.resolve(),
        )
    else:
        _require_clean_git()
        result = run_selected_test_once(
            repository_root=ROOT,
            selection_path=args.selection.resolve(),
            output_directory=args.output_directory.resolve(),
            receipt_path=args.receipt.resolve(),
            report_path=args.report.resolve() if args.report is not None else None,
        )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M2ContractError as exc:
        raise SystemExit(str(exc)) from exc
