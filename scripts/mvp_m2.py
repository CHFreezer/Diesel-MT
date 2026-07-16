"""TD-16 bounded M2 arm finalization, dev selection, and one-shot test gate."""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from freeze_tokenizer_artifact import sha256_file
from model_training_contract import config_sha256, load_student_config
from mvp_checkpoint import (
    build_checkpoint_identity,
    prune_after_validated_publish,
    validate_checkpoint,
)
from mvp_distillation_ab import (
    compare_recipe_configs,
    load_ab_config,
)
from mvp_evaluation import evaluate_checkpoint, publish_evaluation
from mvp_resource_profile import load_m2_profile
from mvp_student import (
    build_student,
    load_frozen_tokenizer,
    state_dict_sha256,
    validate_student_alignment,
)
from mvp_training import (
    ROUTE_ORDER,
    git_identity,
    load_training_config,
    prepare_training_run,
)


M2_SCHEMA_VERSION = 1
CANDIDATE_MANIFEST = "candidate-manifest.json"
ARM_MANIFEST = "arm-manifest.json"
ARM_CONFIGS = {
    "human-only": "configs/mvp_training_m2_human.yaml",
    "distilled": "configs/mvp_training_m2_distilled.yaml",
}
CHINESE_CONVERSION_ROUTES = {
    "zho_Hans->zho_Hant",
    "zho_Hant->zho_Hans",
}


class M2ContractError(RuntimeError):
    """Raised when a TD-16 identity, selection, or test boundary changes."""


def _code_identity(repository_root: Path) -> dict[str, str]:
    return {
        relative: sha256_file(repository_root / relative)
        for relative in ("scripts/mvp_m2.py", "scripts/run_mvp_m2.py")
    }


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise M2ContractError(f"refusing to overwrite frozen output: {path}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise M2ContractError(f"one-shot record already exists: {path}") from exc


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise M2ContractError(f"cannot load {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise M2ContractError(f"{context} must be a JSON object")
    return value


def _step_from_name(name: str) -> int:
    prefix = "step-"
    if not name.startswith(prefix) or len(name) != len(prefix) + 8:
        raise M2ContractError(f"invalid optimizer-step directory: {name}")
    try:
        step = int(name[len(prefix) :])
    except ValueError as exc:
        raise M2ContractError(f"invalid optimizer-step directory: {name}") from exc
    if step <= 0:
        raise M2ContractError("optimizer step must be positive")
    return step


def _release_accelerator() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def prepare_m2_arm(
    *,
    repository_root: Path,
    ab_config_path: Path,
    arm: str,
    require_clean_git: bool,
) -> dict[str, Any]:
    """Revalidate the frozen TD-15 release and build the exact checkpoint identity."""

    if arm not in ARM_CONFIGS:
        raise M2ContractError(f"unsupported M2 arm: {arm}")
    ab_config = load_ab_config(ab_config_path)
    profile_path = repository_root / ab_config["training"]["profile"]
    if sha256_file(profile_path) != ab_config["training"]["profile_sha256"]:
        raise M2ContractError("frozen M2 resource profile SHA-256 changed")
    profile = load_m2_profile(profile_path)
    recipe_paths = {
        name: repository_root / relative for name, relative in ARM_CONFIGS.items()
    }
    recipes = {
        name: load_training_config(path) for name, path in recipe_paths.items()
    }
    compare_recipe_configs(recipes["human-only"], recipes["distilled"], profile)

    td15_path = repository_root / "artifacts/model-training/reports/m2/distillation-ab.json"
    td15 = _load_json(td15_path, "TD-15 evidence")
    if td15.get("status") != "complete" or not all(td15.get("gates", {}).values()):
        raise M2ContractError("TD-15 evidence is incomplete or has a failed gate")
    if td15.get("contract", {}).get("file_sha256") != sha256_file(ab_config_path):
        raise M2ContractError("TD-15 evidence belongs to another A/B contract")
    for name in ARM_CONFIGS:
        record = td15.get("recipes", {}).get(name, {})
        if record.get("file_sha256") != sha256_file(recipe_paths[name]):
            raise M2ContractError(f"TD-15 {name} recipe SHA-256 changed")
        if record.get("canonical_sha256") != config_sha256(recipes[name]):
            raise M2ContractError(f"TD-15 {name} canonical recipe changed")

    recipe_path = recipe_paths[arm]
    recipe, student_config, training_report = prepare_training_run(
        config_path=recipe_path,
        repository_root=repository_root,
    )
    if require_clean_git and training_report["git"]["dirty"]:
        raise M2ContractError("formal M2 execution requires a clean Git worktree")
    expected_identity = build_checkpoint_identity(
        repository_root=repository_root,
        training_report=training_report,
        training_config=recipe,
    )
    return {
        "arm": arm,
        "ab_config": ab_config,
        "ab_config_path": ab_config_path,
        "profile": profile,
        "recipe": recipe,
        "recipe_path": recipe_path,
        "student_config": student_config,
        "training_report": training_report,
        "checkpoint_identity": expected_identity,
        "checkpoint_identity_sha256": config_sha256(expected_identity),
    }


def _candidate_payload_files(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": item.relative_to(path).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.name != CANDIDATE_MANIFEST
    ]


def validate_hf_candidate(
    path: Path,
    *,
    expected_arm: str | None = None,
    expected_source_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    manifest_path = path / CANDIDATE_MANIFEST
    manifest = _load_json(manifest_path, "M2 HF candidate manifest")
    required = {
        "schema_version",
        "status",
        "arm",
        "optimizer_step",
        "training_config",
        "checkpoint_identity_sha256",
        "code",
        "source_checkpoint",
        "state_dict_sha256",
        "model_alignment",
        "tokenizer_manifest_sha256",
        "files",
    }
    if set(manifest) != required:
        raise M2ContractError("M2 HF candidate manifest fields changed")
    if manifest["schema_version"] != M2_SCHEMA_VERSION or manifest["status"] != "complete":
        raise M2ContractError("M2 HF candidate is incomplete")
    if expected_arm is not None and manifest["arm"] != expected_arm:
        raise M2ContractError("M2 HF candidate belongs to another arm")
    source_sha = manifest["source_checkpoint"].get("manifest_sha256")
    if (
        expected_source_manifest_sha256 is not None
        and source_sha != expected_source_manifest_sha256
    ):
        raise M2ContractError("M2 HF candidate source checkpoint changed")
    records = manifest["files"]
    if not isinstance(records, list) or not records:
        raise M2ContractError("M2 HF candidate file list is empty")
    expected_files = {str(record.get("path")) for record in records}
    actual_files = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name != CANDIDATE_MANIFEST
    }
    if expected_files != actual_files:
        raise M2ContractError("M2 HF candidate file set changed")
    for record in records:
        payload = path / str(record["path"])
        if payload.stat().st_size != record.get("bytes"):
            raise M2ContractError("M2 HF candidate byte count changed")
        if sha256_file(payload) != record.get("sha256"):
            raise M2ContractError("M2 HF candidate payload SHA-256 changed")
    return manifest


def export_hf_candidate(
    *,
    repository_root: Path,
    context: Mapping[str, Any],
    checkpoint: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Publish an evaluator-compatible HF checkpoint from an exact-resume checkpoint."""

    source_manifest = validate_checkpoint(
        checkpoint,
        expected_identity=context["checkpoint_identity"],
    )
    source_manifest_sha256 = sha256_file(checkpoint / "checkpoint-manifest.json")
    step = int(source_manifest["summary"]["global_step"])
    if output_directory.name != checkpoint.name or _step_from_name(checkpoint.name) != step:
        raise M2ContractError("candidate path and checkpoint optimizer step differ")
    if output_directory.exists():
        return validate_hf_candidate(
            output_directory,
            expected_arm=str(context["arm"]),
            expected_source_manifest_sha256=source_manifest_sha256,
        )

    student_config = context["student_config"]
    tokenizer, tokenizer_identity = load_frozen_tokenizer(
        student_config, repository_root
    )
    model, _ = build_student(student_config, tokenizer)
    import torch

    state = torch.load(
        checkpoint / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    alignment = validate_student_alignment(model, tokenizer, student_config)
    state_sha256 = state_dict_sha256(model)

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.staging-",
            dir=output_directory.parent,
        )
    )
    try:
        model.save_pretrained(staging, safe_serialization=True)
        from transformers import M2M100ForConditionalGeneration

        reloaded = M2M100ForConditionalGeneration.from_pretrained(
            staging, local_files_only=True
        )
        reloaded_alignment = validate_student_alignment(
            reloaded, tokenizer, student_config
        )
        if reloaded_alignment != alignment or state_dict_sha256(reloaded) != state_sha256:
            raise M2ContractError("HF candidate changed across offline save/reload")
        manifest = {
            "schema_version": M2_SCHEMA_VERSION,
            "status": "complete",
            "arm": context["arm"],
            "optimizer_step": step,
            "training_config": {
                "path": context["training_report"]["training_config"]["path"],
                "file_sha256": context["training_report"]["training_config"][
                    "file_sha256"
                ],
                "canonical_sha256": context["training_report"]["training_config"][
                    "canonical_sha256"
                ],
            },
            "checkpoint_identity_sha256": context["checkpoint_identity_sha256"],
            "code": _code_identity(repository_root),
            "source_checkpoint": {
                "path": str(checkpoint.resolve()),
                "manifest_sha256": source_manifest_sha256,
            },
            "state_dict_sha256": state_sha256,
            "model_alignment": alignment,
            "tokenizer_manifest_sha256": tokenizer_identity[
                "artifact_manifest_sha256"
            ],
            "files": _candidate_payload_files(staging),
        }
        _atomic_json(staging / CANDIDATE_MANIFEST, manifest, overwrite=False)
        os.replace(staging, output_directory)
        return validate_hf_candidate(
            output_directory,
            expected_arm=str(context["arm"]),
            expected_source_manifest_sha256=source_manifest_sha256,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        del model
        if "reloaded" in locals():
            del reloaded
        _release_accelerator()


def _load_dev_evaluation(
    evaluation_directory: Path,
    *,
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    summary_path = evaluation_directory / "summary.json"
    manifest_path = evaluation_directory / "manifest.json"
    summary = _load_json(summary_path, "M2 dev evaluation summary")
    publication = _load_json(manifest_path, "M2 dev evaluation manifest")
    if summary.get("status") != "passed" or summary.get("split") != "dev":
        raise M2ContractError("M2 candidate evaluation is not a passed dev run")
    if summary.get("test_access_explicitly_authorized"):
        raise M2ContractError("M2 arm finalization must never authorize test access")
    state_sha256 = summary.get("identities", {}).get("checkpoint_state_sha256")
    if state_sha256 != candidate_manifest["state_dict_sha256"]:
        raise M2ContractError("dev evaluation belongs to another candidate state")
    if publication.get("checkpoint_state_sha256") != state_sha256:
        raise M2ContractError("dev evaluation manifest state identity changed")
    return summary


def _candidate_record(
    *,
    candidate_directory: Path,
    candidate_manifest: Mapping[str, Any],
    evaluation_directory: Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "optimizer_step": int(candidate_manifest["optimizer_step"]),
        "candidate_path": str(candidate_directory.resolve()),
        "candidate_manifest_sha256": sha256_file(
            candidate_directory / CANDIDATE_MANIFEST
        ),
        "checkpoint_state_sha256": candidate_manifest["state_dict_sha256"],
        "dev_evaluation_path": str(evaluation_directory.resolve()),
        "dev_summary_sha256": sha256_file(evaluation_directory / "summary.json"),
        "dev_manifest_sha256": sha256_file(evaluation_directory / "manifest.json"),
        "dev_records": int(summary["records"]),
    }


def finalize_m2_arm(
    *,
    repository_root: Path,
    context: Mapping[str, Any],
    checkpoint_root: Path,
    candidate_root: Path,
    evaluation_root: Path,
    require_complete: bool,
) -> dict[str, Any]:
    """Export/evaluate every scheduled checkpoint, then enforce retention."""

    checkpoint_root = checkpoint_root.resolve()
    candidate_root = candidate_root.resolve()
    evaluation_root = evaluation_root.resolve()
    checkpoints = sorted(checkpoint_root.glob("step-*"))
    if not checkpoints and not evaluation_root.exists():
        raise M2ContractError("M2 arm has no exact checkpoints to finalize")

    for checkpoint in checkpoints:
        step = _step_from_name(checkpoint.name)
        validate_checkpoint(
            checkpoint,
            expected_identity=context["checkpoint_identity"],
        )
        candidate_directory = candidate_root / checkpoint.name
        candidate_manifest = export_hf_candidate(
            repository_root=repository_root,
            context=context,
            checkpoint=checkpoint,
            output_directory=candidate_directory,
        )
        if int(candidate_manifest["optimizer_step"]) != step:
            raise M2ContractError("exported candidate optimizer step changed")
        evaluation_directory = evaluation_root / checkpoint.name
        if evaluation_directory.exists():
            _load_dev_evaluation(
                evaluation_directory,
                candidate_manifest=candidate_manifest,
            )
        else:
            summary, samples = evaluate_checkpoint(
                repository_root=repository_root,
                evaluation_config_path=(
                    repository_root / context["ab_config"]["evaluation"]["protocol"]
                ),
                checkpoint=candidate_directory,
                split="dev",
                allow_test=False,
            )
            publish_evaluation(evaluation_directory, summary, samples)
            _load_dev_evaluation(
                evaluation_directory,
                candidate_manifest=candidate_manifest,
            )
            _release_accelerator()

    candidate_records: list[dict[str, Any]] = []
    if evaluation_root.exists():
        for evaluation_directory in sorted(evaluation_root.glob("step-*")):
            step = _step_from_name(evaluation_directory.name)
            candidate_directory = candidate_root / evaluation_directory.name
            candidate_manifest = validate_hf_candidate(
                candidate_directory,
                expected_arm=str(context["arm"]),
            )
            if int(candidate_manifest["optimizer_step"]) != step:
                raise M2ContractError("candidate/evaluation optimizer steps differ")
            summary = _load_dev_evaluation(
                evaluation_directory,
                candidate_manifest=candidate_manifest,
            )
            candidate_records.append(
                _candidate_record(
                    candidate_directory=candidate_directory,
                    candidate_manifest=candidate_manifest,
                    evaluation_directory=evaluation_directory,
                    summary=summary,
                )
            )

    optimization = context["recipe"]["optimization"]
    frequency = int(optimization["checkpoint_frequency"])
    maximum = int(optimization["max_optimizer_steps"])
    expected_steps = list(range(frequency, maximum + 1, frequency))
    actual_steps = [record["optimizer_step"] for record in candidate_records]
    missing_steps = sorted(set(expected_steps) - set(actual_steps))
    extra_steps = sorted(set(actual_steps) - set(expected_steps))
    if extra_steps:
        raise M2ContractError(f"M2 arm has unscheduled checkpoint steps: {extra_steps}")
    if require_complete and missing_steps:
        raise M2ContractError(f"M2 arm is missing scheduled dev checkpoints: {missing_steps}")

    removed: list[str] = []
    status = "complete" if not missing_steps else "incomplete"
    if status == "complete" and checkpoints:
        keep_last = int(context["profile"]["optimization"]["checkpoint_retention"])
        removed = [
            str(path.resolve())
            for path in prune_after_validated_publish(
                checkpoint_root,
                newest_checkpoint=checkpoints[-1],
                expected_identity=context["checkpoint_identity"],
                keep_last=keep_last,
            )
        ]
    retained = [str(path.resolve()) for path in sorted(checkpoint_root.glob("step-*"))]
    report = {
        "schema_version": M2_SCHEMA_VERSION,
        "status": status,
        "task": "TD-16",
        "arm": context["arm"],
        "ab_config_sha256": sha256_file(context["ab_config_path"]),
        "training_config": context["training_report"]["training_config"],
        "checkpoint_identity_sha256": context["checkpoint_identity_sha256"],
        "code": _code_identity(repository_root),
        "git": context["training_report"]["git"],
        "expected_optimizer_steps": expected_steps,
        "missing_optimizer_steps": missing_steps,
        "candidates": candidate_records,
        "checkpoint_retention": {
            "keep_last": int(
                context["profile"]["optimization"]["checkpoint_retention"]
            ),
            "removed": removed,
            "retained": retained,
            "applied_only_after_all_dev_candidates_published": status == "complete",
        },
        "test_access": "forbidden",
    }
    _atomic_json(evaluation_root / ARM_MANIFEST, report, overwrite=True)
    return report


def _cross_language_copy_rate(route_metrics: Mapping[str, Any]) -> float:
    rows = [
        value
        for route, value in route_metrics.items()
        if route not in CHINESE_CONVERSION_ROUTES
    ]
    samples = sum(int(row["samples"]) for row in rows)
    if samples <= 0:
        raise M2ContractError("dev route metrics have no cross-language samples")
    copied = sum(float(row["source_copy_rate"]) * int(row["samples"]) for row in rows)
    return copied / samples


def _candidate_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    aggregates = summary.get("aggregates", {})
    overall = aggregates.get("overall", {})
    route_metrics = aggregates.get("route20", {})
    if set(route_metrics) != set(ROUTE_ORDER):
        raise M2ContractError("dev summary does not contain all 20 routes")
    return {
        "overall": dict(overall),
        "route20": {route: dict(route_metrics[route]) for route in ROUTE_ORDER},
        "macro_route_chrf": sum(float(route_metrics[route]["chrf"]) for route in ROUTE_ORDER)
        / len(ROUTE_ORDER),
        "macro_route_sacrebleu": sum(
            float(route_metrics[route]["sacrebleu"]) for route in ROUTE_ORDER
        )
        / len(ROUTE_ORDER),
        "cross_language_source_copy_rate": _cross_language_copy_rate(route_metrics),
    }


def _eligible(metrics: Mapping[str, Any], rules: Mapping[str, Any]) -> dict[str, bool]:
    overall = metrics["overall"]
    route20 = metrics["route20"]
    loss = float(overall["loss"])
    return {
        "finite_loss": math.isfinite(loss),
        "all_20_routes": set(route20) == set(ROUTE_ORDER),
        "overall_script_compliance": float(overall["script_compliance_rate"])
        >= float(rules["minimum_overall_script_compliance"]),
        "per_route_script_compliance": min(
            float(row["script_compliance_rate"]) for row in route20.values()
        )
        >= float(rules["minimum_per_route_script_compliance"]),
        "empty_output_rate": float(overall["empty_output_rate"])
        <= float(rules["maximum_empty_output_rate"]),
        "cross_language_source_copy_rate": float(
            metrics["cross_language_source_copy_rate"]
        )
        <= float(rules["maximum_cross_language_source_copy_rate"]),
        "target_control_rate": float(overall["target_control_rate"])
        >= (1.0 if rules["require_target_control_rate"] else 0.0),
    }


def _load_arm_candidates(
    path: Path,
    *,
    expected_arm: str,
    expected_ab_sha256: str,
    rules: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arm_manifest = _load_json(path, f"{expected_arm} arm manifest")
    if arm_manifest.get("status") != "complete" or arm_manifest.get("arm") != expected_arm:
        raise M2ContractError(f"{expected_arm} arm is not complete")
    if arm_manifest.get("ab_config_sha256") != expected_ab_sha256:
        raise M2ContractError(f"{expected_arm} arm belongs to another A/B contract")
    candidates = []
    for record in arm_manifest.get("candidates", []):
        summary_path = Path(record["dev_evaluation_path"]) / "summary.json"
        if sha256_file(summary_path) != record["dev_summary_sha256"]:
            raise M2ContractError("frozen dev summary SHA-256 changed")
        summary = _load_json(summary_path, "frozen M2 dev summary")
        if summary.get("split") != "dev" or summary.get("test_access_explicitly_authorized"):
            raise M2ContractError("candidate selection may consume dev only")
        candidate_path = Path(record["candidate_path"])
        candidate_manifest = validate_hf_candidate(
            candidate_path,
            expected_arm=expected_arm,
        )
        if sha256_file(candidate_path / CANDIDATE_MANIFEST) != record[
            "candidate_manifest_sha256"
        ]:
            raise M2ContractError("frozen candidate manifest SHA-256 changed")
        if candidate_manifest["state_dict_sha256"] != record["checkpoint_state_sha256"]:
            raise M2ContractError("candidate state identity changed")
        metrics = _candidate_metrics(summary)
        gates = _eligible(metrics, rules)
        candidates.append({**record, "metrics": metrics, "eligibility": gates})
    if not candidates:
        raise M2ContractError(f"{expected_arm} arm has no dev candidates")
    return arm_manifest, candidates


def _best_within_arm(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [candidate for candidate in candidates if all(candidate["eligibility"].values())]
    if not eligible:
        raise M2ContractError("M2 arm has no checkpoint eligible under frozen dev rules")
    return min(
        eligible,
        key=lambda item: (
            -float(item["metrics"]["macro_route_chrf"]),
            -float(item["metrics"]["macro_route_sacrebleu"]),
            float(item["metrics"]["overall"]["loss"]),
            int(item["optimizer_step"]),
        ),
    )


def select_m2_candidate(
    *,
    ab_config_path: Path,
    human_arm_manifest_path: Path,
    distilled_arm_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze the unique M2 candidate using dev only."""

    ab_config = load_ab_config(ab_config_path)
    ab_sha256 = sha256_file(ab_config_path)
    eligibility_rules = ab_config["dev_selection"]["checkpoint_eligibility"]
    human_manifest, human_candidates = _load_arm_candidates(
        human_arm_manifest_path,
        expected_arm="human-only",
        expected_ab_sha256=ab_sha256,
        rules=eligibility_rules,
    )
    distilled_manifest, distilled_candidates = _load_arm_candidates(
        distilled_arm_manifest_path,
        expected_arm="distilled",
        expected_ab_sha256=ab_sha256,
        rules=eligibility_rules,
    )
    if human_manifest["expected_optimizer_steps"] != distilled_manifest[
        "expected_optimizer_steps"
    ]:
        raise M2ContractError("M2 arms used different optimizer-step budgets")
    repository_root = ab_config_path.resolve().parents[1]
    current_code = _code_identity(repository_root)
    current_git = git_identity(repository_root)
    if human_manifest.get("code") != current_code or distilled_manifest.get("code") != current_code:
        raise M2ContractError("M2 arm finalization code identity changed")
    if human_manifest.get("git") != distilled_manifest.get("git"):
        raise M2ContractError("M2 arms were finalized from different Git identities")
    if human_manifest.get("git", {}).get("commit") != current_git["commit"]:
        raise M2ContractError("M2 arm Git commit differs from candidate selection")
    human = _best_within_arm(human_candidates)
    distilled = _best_within_arm(distilled_candidates)
    human_metrics = human["metrics"]
    distilled_metrics = distilled["metrics"]
    rules = ab_config["dev_selection"]["distilled_vs_human"]
    route_deltas = {
        route: float(distilled_metrics["route20"][route]["chrf"])
        - float(human_metrics["route20"][route]["chrf"])
        for route in ROUTE_ORDER
    }
    comparison_gates = {
        "minimum_overall_chrf_gain": float(
            distilled_metrics["overall"]["chrf"]
        )
        - float(human_metrics["overall"]["chrf"])
        >= float(rules["minimum_overall_chrf_gain"]),
        "maximum_overall_sacrebleu_degradation": float(
            distilled_metrics["overall"]["sacrebleu"]
        )
        >= float(human_metrics["overall"]["sacrebleu"])
        - float(rules["maximum_overall_sacrebleu_degradation"]),
        "maximum_any_route_chrf_degradation": min(route_deltas.values())
        >= -float(rules["maximum_any_route_chrf_degradation"]),
        "maximum_script_compliance_degradation": float(
            distilled_metrics["overall"]["script_compliance_rate"]
        )
        >= float(human_metrics["overall"]["script_compliance_rate"])
        - float(rules["maximum_script_compliance_degradation"]),
        "maximum_empty_output_rate_increase": float(
            distilled_metrics["overall"]["empty_output_rate"]
        )
        <= float(human_metrics["overall"]["empty_output_rate"])
        + float(rules["maximum_empty_output_rate_increase"]),
        "maximum_source_copy_rate_increase": float(
            distilled_metrics["cross_language_source_copy_rate"]
        )
        <= float(human_metrics["cross_language_source_copy_rate"])
        + float(rules["maximum_source_copy_rate_increase"]),
    }
    selected_arm = "distilled" if all(comparison_gates.values()) else "human-only"
    selected = distilled if selected_arm == "distilled" else human
    selection = {
        "schema_version": M2_SCHEMA_VERSION,
        "status": "frozen",
        "task": "TD-16",
        "ab_config": {
            "path": str(ab_config_path.resolve()),
            "file_sha256": ab_sha256,
            "canonical_sha256": config_sha256(ab_config),
        },
        "code": current_code,
        "git": current_git,
        "optimizer_step_budget": human_manifest["expected_optimizer_steps"],
        "within_arm_priority": ab_config["dev_selection"]["within_arm_priority"],
        "best_by_arm": {
            "human-only": human,
            "distilled": distilled,
        },
        "comparison": {
            "gates": comparison_gates,
            "route_chrf_deltas": route_deltas,
            "zho_Hans_route_deltas": {
                route: delta for route, delta in route_deltas.items() if "zho_Hans" in route
            },
            "zho_Hant_route_deltas": {
                route: delta for route, delta in route_deltas.items() if "zho_Hant" in route
            },
            "fallback_if_any_gate_fails": rules["fallback_if_any_gate_fails"],
        },
        "selected": {
            "arm": selected_arm,
            "optimizer_step": selected["optimizer_step"],
            "candidate_path": selected["candidate_path"],
            "candidate_manifest_sha256": selected["candidate_manifest_sha256"],
            "checkpoint_state_sha256": selected["checkpoint_state_sha256"],
        },
        "test_access": {
            "authorized_after_selection": True,
            "runs_allowed": int(
                ab_config["evaluation"]["test_runs_allowed_after_selection"]
            ),
            "runs_consumed": 0,
        },
    }
    _atomic_json(output_path, selection, overwrite=False)
    return selection


def run_selected_test_once(
    *,
    repository_root: Path,
    selection_path: Path,
    output_directory: Path,
    receipt_path: Path,
    report_path: Path | None,
) -> dict[str, Any]:
    """Consume the sole formal-test authorization for the frozen candidate."""

    selection = _load_json(selection_path, "M2 candidate selection")
    if selection.get("status") != "frozen":
        raise M2ContractError("M2 candidate selection is not frozen")
    if selection.get("code") != _code_identity(repository_root):
        raise M2ContractError("M2 selection code identity changed before formal test")
    if selection.get("git", {}).get("commit") != git_identity(repository_root)["commit"]:
        raise M2ContractError("M2 selection Git commit changed before formal test")
    access = selection.get("test_access", {})
    if access.get("runs_allowed") != 1 or access.get("runs_consumed") != 0:
        raise M2ContractError("M2 formal-test authorization is not exactly one unused run")
    ab_path = Path(selection["ab_config"]["path"])
    if sha256_file(ab_path) != selection["ab_config"]["file_sha256"]:
        raise M2ContractError("selection A/B contract SHA-256 changed")
    ab_config = load_ab_config(ab_path)
    selected = selection["selected"]
    candidate_path = Path(selected["candidate_path"])
    candidate_manifest = validate_hf_candidate(
        candidate_path,
        expected_arm=str(selected["arm"]),
    )
    if sha256_file(candidate_path / CANDIDATE_MANIFEST) != selected[
        "candidate_manifest_sha256"
    ]:
        raise M2ContractError("selected candidate manifest SHA-256 changed")
    if candidate_manifest["state_dict_sha256"] != selected["checkpoint_state_sha256"]:
        raise M2ContractError("selected candidate state identity changed")
    if output_directory.exists() or (report_path is not None and report_path.exists()):
        raise M2ContractError("formal test output already exists")

    receipt = {
        "schema_version": M2_SCHEMA_VERSION,
        "status": "running",
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "candidate_path": str(candidate_path.resolve()),
        "checkpoint_state_sha256": selected["checkpoint_state_sha256"],
        "attempt": 1,
    }
    _exclusive_json(receipt_path, receipt)
    try:
        summary, samples = evaluate_checkpoint(
            repository_root=repository_root,
            evaluation_config_path=repository_root
            / ab_config["evaluation"]["protocol"],
            checkpoint=candidate_path,
            split="test",
            allow_test=True,
        )
        publication = publish_evaluation(output_directory, summary, samples)
        result = {
            **summary,
            "publication": {
                "path": str(output_directory.resolve()),
                "manifest": publication,
            },
        }
        if report_path is not None:
            _atomic_json(report_path, result, overwrite=False)
        completed = {
            **receipt,
            "status": "complete",
            "test_output": str(output_directory.resolve()),
            "test_manifest_sha256": sha256_file(output_directory / "manifest.json"),
        }
        _atomic_json(receipt_path, completed, overwrite=True)
        return result
    except BaseException as exc:
        failed = {
            **receipt,
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        _atomic_json(receipt_path, failed, overwrite=True)
        raise
    finally:
        _release_accelerator()
