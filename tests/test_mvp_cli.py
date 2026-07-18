from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mvp_cli import build_parser  # noqa: E402


def test_unified_mvp_cli_exposes_all_student_workflow_commands() -> None:
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if getattr(action, "choices", None)
    )
    assert set(subparsers.choices) == {
        "train",
        "validate-student",
        "validate-resume",
        "validate-m1",
        "evaluate",
        "benchmark-resources",
        "validate-resources",
        "build-ab",
        "validate-ab",
    }


def test_train_subcommand_preserves_dry_run_surface() -> None:
    args = build_parser().parse_args(
        ["train", "--dry-run", "--stop-after-optimizer-step", "3"]
    )
    assert args.command == "train"
    assert args.dry_run is True
    assert args.stop_after_optimizer_step == 3
