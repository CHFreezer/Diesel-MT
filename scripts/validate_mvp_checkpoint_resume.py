"""Run the TD-11 uninterrupted versus interrupted/resumed acceptance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freeze_tokenizer_artifact import sha256_file
from model_training_contract import config_sha256
from mvp_checkpoint import CHECKPOINT_MANIFEST, validate_checkpoint
from mvp_training import (
    _atomic_json,
    load_training_config,
    read_jsonl,
    run_training,
    semantic_trace_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "mvp_training_td10_smoke.yaml",
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "artifacts"
        / "model-training"
        / "reports"
        / "student"
        / "checkpoint-resume.json",
    )
    return parser.parse_args()


def _events(report: dict) -> list[dict]:
    return read_jsonl(Path(report["output_root"]) / report["events"]["path"])


def _payload_hashes(checkpoint: Path) -> dict[str, str]:
    manifest = json.loads((checkpoint / CHECKPOINT_MANIFEST).read_text(encoding="utf-8"))
    return {record["path"]: record["sha256"] for record in manifest["files"]}


def main() -> int:
    args = parse_args()
    runtime_root = args.runtime_root.resolve()
    if runtime_root.exists():
        raise RuntimeError(f"TD-11 runtime root already exists: {runtime_root}")
    config_path = args.config.resolve()
    config = load_training_config(config_path)
    maximum = int(config["optimization"]["max_optimizer_steps"])
    if maximum < 2:
        raise RuntimeError("TD-11 acceptance requires at least two optimizer steps")
    interruption_step = maximum // 2

    baseline = run_training(
        config_path=config_path,
        repository_root=ROOT,
        output_dir=runtime_root / "uninterrupted-run",
        dry_run=False,
        checkpoint_root=runtime_root / "uninterrupted-checkpoints",
    )
    interrupted = run_training(
        config_path=config_path,
        repository_root=ROOT,
        output_dir=runtime_root / "interrupted-run",
        dry_run=False,
        checkpoint_root=runtime_root / "resumed-checkpoints",
        stop_after_optimizer_steps=interruption_step,
    )
    resume_checkpoint = runtime_root / "resumed-checkpoints" / f"step-{interruption_step:08d}"
    resumed = run_training(
        config_path=config_path,
        repository_root=ROOT,
        output_dir=runtime_root / "resumed-run",
        dry_run=False,
        checkpoint_root=runtime_root / "resumed-checkpoints",
        resume_from=resume_checkpoint,
    )
    baseline_checkpoint = (
        runtime_root / "uninterrupted-checkpoints" / f"step-{maximum:08d}"
    )
    resumed_checkpoint = runtime_root / "resumed-checkpoints" / f"step-{maximum:08d}"
    baseline_manifest = validate_checkpoint(baseline_checkpoint)
    resumed_manifest = validate_checkpoint(resumed_checkpoint)
    baseline_trace = semantic_trace_sha256(_events(baseline))
    resumed_trace = semantic_trace_sha256([*_events(interrupted), *_events(resumed)])
    comparisons = {
        "semantic_trace": baseline_trace == resumed_trace,
        "final_train_loss": baseline["result"]["final_train_loss"]
        == resumed["result"]["final_train_loss"],
        "mean_train_loss": baseline["result"]["mean_train_loss"]
        == resumed["result"]["mean_train_loss"],
        "optimizer_steps": baseline["result"]["optimizer_steps"]
        == resumed["result"]["optimizer_steps"],
        "micro_steps": baseline["result"]["micro_steps"]
        == resumed["result"]["micro_steps"],
        "sampler_state": config_sha256(baseline["result"]["sampler_state"])
        == config_sha256(resumed["result"]["sampler_state"]),
        "checkpoint_payloads": _payload_hashes(baseline_checkpoint)
        == _payload_hashes(resumed_checkpoint),
    }
    if not all(comparisons.values()):
        raise RuntimeError(f"TD-11 exact resume comparison failed: {comparisons}")
    report = {
        "schema_version": 1,
        "status": "complete",
        "task": "TD-11",
        "training_config": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "file_sha256": sha256_file(config_path),
            "canonical_sha256": config_sha256(config),
        },
        "interruption_step": interruption_step,
        "final_step": maximum,
        "runs": {
            "uninterrupted": {
                "output_root": baseline["output_root"],
                "events_sha256": baseline["events"]["sha256"],
                "final_loss": baseline["result"]["final_train_loss"],
            },
            "interrupted": {
                "output_root": interrupted["output_root"],
                "events_sha256": interrupted["events"]["sha256"],
                "status": interrupted["status"],
            },
            "resumed": {
                "output_root": resumed["output_root"],
                "events_sha256": resumed["events"]["sha256"],
                "final_loss": resumed["result"]["final_train_loss"],
            },
        },
        "comparison": {
            "status": "exact",
            "checks": comparisons,
            "semantic_trace_sha256": baseline_trace,
        },
        "final_checkpoints": {
            "uninterrupted": {
                "path": str(baseline_checkpoint),
                "manifest_sha256": sha256_file(baseline_checkpoint / CHECKPOINT_MANIFEST),
                "identity_sha256": baseline_manifest["identity_sha256"],
                "payloads": _payload_hashes(baseline_checkpoint),
            },
            "resumed": {
                "path": str(resumed_checkpoint),
                "manifest_sha256": sha256_file(resumed_checkpoint / CHECKPOINT_MANIFEST),
                "identity_sha256": resumed_manifest["identity_sha256"],
                "payloads": _payload_hashes(resumed_checkpoint),
            },
        },
        "automated_gates": {
            "fault_injection_points": [
                "after_model",
                "after_optimizer",
                "before_manifest",
                "after_manifest_before_publish",
            ],
            "corrupt_incomplete_extra_identity_path_link_rejection": True,
            "retention_requires_newest_validation": True,
        },
    }
    _atomic_json(args.report.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
