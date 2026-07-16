"""TD-15 immutable human/distilled source-matched A/B cohort builder."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from freeze_tokenizer_artifact import sha256_file
from model_training_contract import config_sha256, directed_routes, load_student_config
from mvp_evaluation import target_script_compliant
from mvp_student import build_student, load_frozen_tokenizer, state_dict_sha256
from mvp_training import DeterministicRouteSampler, ROUTE_ORDER, load_route_dataset


class DistillationABError(RuntimeError):
    """Raised before TD-16 when the A/B comparison is not source-fair."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _repo_path(root: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DistillationABError(f"{context} must be a repository-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise DistillationABError(f"{context} escapes the repository")
    return root / path


def _verified(root: Path, path_value: Any, digest: Any, context: str) -> Path:
    path = _repo_path(root, path_value, context)
    if sha256_file(path) != digest:
        raise DistillationABError(f"{context} SHA-256 changed")
    return path


def load_ab_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DistillationABError(f"cannot load A/B config: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise DistillationABError("unsupported A/B config schema")
    expected = {"schema_version", "identity", "human", "distilled", "student", "training", "evaluation", "dev_selection", "outputs"}
    if set(value) != expected:
        raise DistillationABError("A/B config fields are incomplete or unknown")
    if value["identity"] != {
        "name": "mvp-m2-human-distilled-ab-v1",
        "status": "frozen_before_m2_results",
        "scope": "twenty_route_accepted_intersection_train_only",
    }:
        raise DistillationABError("A/B identity changed")
    distilled = value["distilled"]
    if distilled.get("required_routes") != 20 or distilled.get("minimum_accepted_per_route", 0) < 2000:
        raise DistillationABError("A/B requires 20 routes with at least 2,000 accepted each")
    if distilled.get("required_identity_name") != "hymt2-sequence-distillation-d1-20route-composite-v2":
        raise DistillationABError("D0/D1 standalone input is forbidden")
    accepted_path = str(distilled.get("accepted", ""))
    if "/distilled/d1-20route/" not in accepted_path or any(
        marker in accepted_path
        for marker in ("/distilled/d0-v1/", "/distilled/d1-v1/", "/distilled/d1-zh-conversion/")
    ):
        raise DistillationABError("only the frozen 20-route distilled composite is allowed")
    evaluation = value["evaluation"]
    if evaluation.get("references") != "human_only" or evaluation.get("test_access") != "forbidden_until_unique_td16_candidate_is_frozen":
        raise DistillationABError("dev/test boundary changed")
    if value["training"].get("target_length_compensation") != "forbidden":
        raise DistillationABError("target-length budget compensation is forbidden")
    return json.loads(json.dumps(value))


def _load_jsonl(path: Path, *, expected_records: int, context: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DistillationABError(f"invalid {context} JSON line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise DistillationABError(f"{context} contains a non-object row")
            records.append(row)
    if len(records) != expected_records:
        raise DistillationABError(f"{context} record count changed")
    return records


def _normalized(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        raise DistillationABError("parallel text must be non-empty")
    return " ".join(unicodedata.normalize("NFC", text).split())


def _source_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("sample_group_id", "")), str(row.get("src_lang", "")),
        str(row.get("tgt_lang", "")), _normalized(row.get("source_text")),
    )


def _cohort_id(key: tuple[str, str, str, str]) -> str:
    return "cohort-sha256:" + hashlib.sha256("\0".join(key).encode("utf-8")).hexdigest()


def _token_length(tokenizer: object, text: str, language: str) -> int:
    tokenizer.src_lang = language
    return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def _rate(numerator: int, denominator: int) -> float:
    return numerator / max(1, denominator)


def _publish(root: Path, files: Mapping[str, bytes], manifest: Mapping[str, Any]) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        for name, payload in files.items():
            path = staging / name
            with path.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if path.stat().st_size != len(payload) or sha256_file(path) != hashlib.sha256(payload).hexdigest():
                raise DistillationABError(f"staged A/B payload verification failed: {name}")
        with (staging / "manifest.json").open("wb") as handle:
            handle.write(_json_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        if root.exists():
            raise DistillationABError(f"refusing to overwrite A/B output: {root}")
        os.replace(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_ab_cohort(*, repository_root: Path, config_path: Path) -> dict[str, Any]:
    config = load_ab_config(config_path)
    human = config["human"]
    distilled = config["distilled"]
    _verified(repository_root, human["m0_manifest"], human["m0_manifest_sha256"], "human M0 manifest")
    human_path = _verified(repository_root, human["train"], human["train_sha256"], "human train")
    composite_config_path = _verified(repository_root, distilled["composite_config"], distilled["composite_config_sha256"], "distilled composite config")
    composite_manifest_path = _verified(repository_root, distilled["composite_manifest"], distilled["composite_manifest_sha256"], "distilled composite manifest")
    distilled_path = _verified(repository_root, distilled["accepted"], distilled["accepted_sha256"], "distilled accepted")
    composite_config = yaml.safe_load(composite_config_path.read_text(encoding="utf-8"))
    composite_manifest = json.loads(composite_manifest_path.read_text(encoding="utf-8"))
    if composite_config["identity"]["name"] != distilled["required_identity_name"] or composite_config["identity"]["scope"] != distilled["required_scope"]:
        raise DistillationABError("distilled input is not the required frozen composite")
    if composite_manifest.get("status") != "complete" or composite_manifest.get("scope", {}).get("dev_records") != 0 or composite_manifest.get("scope", {}).get("test_records") != 0:
        raise DistillationABError("distilled composite is incomplete or not train-only")
    route_counts = composite_manifest.get("route_counts", {})
    if set(route_counts) != set(ROUTE_ORDER) or any(int(count) < distilled["minimum_accepted_per_route"] for count in route_counts.values()):
        raise DistillationABError("distilled composite fails the 20-route minimum")

    human_records = _load_jsonl(human_path, expected_records=human["train_records"], context="human train")
    teacher_records = _load_jsonl(distilled_path, expected_records=distilled["accepted_records"], context="distilled accepted")
    human_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in human_records:
        if row.get("split") != "train" or row.get("provenance", {}).get("kind") != human["required_provenance_kind"]:
            raise DistillationABError("human input contains non-train/non-human provenance")
        key = _source_key(row)
        if key in human_index:
            raise DistillationABError("human source identity is not unique")
        human_index[key] = row

    student_path = _verified(repository_root, config["student"]["config"], config["student"]["config_file_sha256"], "student config")
    student = load_student_config(student_path)
    if config_sha256(student) != config["student"]["config_canonical_sha256"]:
        raise DistillationABError("student canonical identity changed")
    tokenizer, tokenizer_identity = load_frozen_tokenizer(student, repository_root)
    model, _ = build_student(student, tokenizer)
    initial_hash = state_dict_sha256(model)
    if initial_hash != config["student"]["initial_state_dict_sha256"]:
        raise DistillationABError("initial student state changed")
    del model

    common_rows: list[dict[str, Any]] = []
    human_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    accepted_without_human: Counter[str] = Counter()
    profile_path = _verified(repository_root, config["training"]["profile"], config["training"]["profile_sha256"], "M2 training profile")
    _verified(repository_root, config["evaluation"]["protocol"], config["evaluation"]["protocol_sha256"], "evaluation protocol")
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    max_source = int(profile["resource_profile"]["max_source_length"])
    max_target = int(profile["resource_profile"]["max_target_length"])
    for teacher in teacher_records:
        if teacher.get("split") != "train" or teacher.get("provenance", {}).get("kind") != distilled["required_provenance_kind"]:
            raise DistillationABError("distilled accepted contains invalid provenance or split")
        key = _source_key(teacher)
        if key in seen_keys:
            raise DistillationABError("distilled composite contains a duplicate source identity")
        seen_keys.add(key)
        human_row = human_index.get(key)
        if human_row is None:
            accepted_without_human[f"{key[1]}->{key[2]}"] += 1
            continue
        cohort_id = _cohort_id(key)
        route = f"{key[1]}->{key[2]}"
        if route not in ROUTE_ORDER:
            raise DistillationABError(f"cohort contains unsupported route: {route}")
        shared = {
            "license": human_row["license"], "sample_id": cohort_id,
            "sample_group_id": key[0], "source_id": human_row["source_id"],
            "source_version": human_row["source_version"], "source_text": key[3],
            "split": "train", "src_lang": key[1], "tgt_lang": key[2],
        }
        common_rows.append({
            **shared, "human_sample_id": human_row["sample_id"],
            "teacher_sample_id": teacher["sample_id"],
        })
        human_target = _normalized(human_row["target_text"])
        teacher_target = _normalized(teacher["target_text"])
        human_rows.append({
            **shared, "target_text": human_target,
            "provenance": {"ab_arm": "human-only", "kind": "human_parallel", "original_sample_id": human_row["sample_id"], "original": human_row["provenance"]},
        })
        teacher_rows.append({
            **shared, "target_text": teacher_target,
            "provenance": {"ab_arm": "distilled", "kind": "teacher_synthetic", "original_sample_id": teacher["sample_id"], "original": teacher["provenance"]},
        })
        source_tokens = _token_length(tokenizer, key[3], key[1])
        human_tokens = _token_length(tokenizer, human_target, key[2])
        teacher_tokens = _token_length(tokenizer, teacher_target, key[2])
        row_stats = stats[route]
        row_stats["samples"] += 1
        row_stats["source_tokens"] += source_tokens
        row_stats["human_target_tokens"] += human_tokens
        row_stats["distilled_target_tokens"] += teacher_tokens
        row_stats["source_truncated_samples"] += source_tokens > max_source
        row_stats["human_target_truncated_samples"] += human_tokens > max_target
        row_stats["distilled_target_truncated_samples"] += teacher_tokens > max_target
        row_stats["human_script_compliant"] += target_script_compliant(human_target, key[2])
        row_stats["distilled_script_compliant"] += target_script_compliant(teacher_target, key[2])
        row_stats["target_text_different"] += human_target != teacher_target

    if len(common_rows) + sum(accepted_without_human.values()) != distilled["accepted_records"] or set(stats) != set(ROUTE_ORDER):
        raise DistillationABError("cohort intersection is incomplete")
    if any(row["samples"] < distilled["minimum_accepted_per_route"] for row in stats.values()):
        raise DistillationABError("cohort route fell below the 2,000 source minimum")
    for human_row, teacher_row in zip(human_rows, teacher_rows, strict=True):
        human_shared = {key: value for key, value in human_row.items() if key not in {"target_text", "provenance"}}
        teacher_shared = {key: value for key, value in teacher_row.items() if key not in {"target_text", "provenance"}}
        if human_shared != teacher_shared:
            raise DistillationABError("A/B rows differ outside target/provenance")

    files = {
        "cohort.jsonl": b"".join(_json_bytes(row) for row in common_rows),
        "human.jsonl": b"".join(_json_bytes(row) for row in human_rows),
        "distilled.jsonl": b"".join(_json_bytes(row) for row in teacher_rows),
    }
    file_records = {
        name: {"bytes": len(payload), "records": len(common_rows), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in files.items()
    }
    source_order_sha256 = config_sha256([row["sample_id"] for row in common_rows])
    route_report = {}
    totals: Counter[str] = Counter()
    for route in ROUTE_ORDER:
        row = stats[route]
        totals.update(row)
        count = row["samples"]
        route_report[route] = {
            **dict(row),
            "source_truncation_rate": _rate(row["source_truncated_samples"], count),
            "human_target_truncation_rate": _rate(row["human_target_truncated_samples"], count),
            "distilled_target_truncation_rate": _rate(row["distilled_target_truncated_samples"], count),
            "human_script_compliance_rate": _rate(row["human_script_compliant"], count),
            "distilled_script_compliance_rate": _rate(row["distilled_script_compliant"], count),
            "target_difference_rate": _rate(row["target_text_different"], count),
        }
    manifest = {
        "schema_version": 1, "status": "complete",
        "identity": config["identity"],
        "config": {"path": config_path.relative_to(repository_root).as_posix(), "file_sha256": sha256_file(config_path), "canonical_sha256": config_sha256(config)},
        "inputs": {
            "human_train_sha256": human["train_sha256"], "human_records": len(human_records),
            "distilled_composite_manifest_sha256": distilled["composite_manifest_sha256"],
            "distilled_accepted_sha256": distilled["accepted_sha256"], "distilled_accepted_records": len(teacher_records),
        },
        "intersection": {
            "records": len(common_rows), "routes": len(stats), "minimum_route_records": min(row["samples"] for row in stats.values()),
            "human_records_excluded_symmetrically": len(human_records) - len(common_rows),
            "accepted_teacher_records_excluded_without_human_match": sum(accepted_without_human.values()),
            "accepted_teacher_exclusions_by_route": {route: accepted_without_human[route] for route in ROUTE_ORDER},
            "teacher_rejected_or_filtered_included": 0,
            "source_order_sha256": source_order_sha256,
        },
        "student_initial_state_dict_sha256": initial_hash,
        "tokenizer": tokenizer_identity,
        "training_profile_sha256": config["training"]["profile_sha256"],
        "evaluation_protocol_sha256": config["evaluation"]["protocol_sha256"],
        "files": file_records,
        "route20": route_report,
        "totals": {
            **dict(totals),
            "source_truncation_rate": _rate(totals["source_truncated_samples"], totals["samples"]),
            "human_target_truncation_rate": _rate(totals["human_target_truncated_samples"], totals["samples"]),
            "distilled_target_truncation_rate": _rate(totals["distilled_target_truncated_samples"], totals["samples"]),
            "human_script_compliance_rate": _rate(totals["human_script_compliant"], totals["samples"]),
            "distilled_script_compliance_rate": _rate(totals["distilled_script_compliant"], totals["samples"]),
            "target_difference_rate": _rate(totals["target_text_different"], totals["samples"]),
        },
        "dev_selection": config["dev_selection"],
        "test_access": config["evaluation"]["test_access"],
    }
    output_root = _repo_path(repository_root, config["outputs"]["root"], "A/B output root")
    _publish(output_root, files, manifest)
    return manifest


def dry_run_pair(
    *, repository_root: Path, config: Mapping[str, Any], human_path: Path, distilled_path: Path,
    human_sha256: str, distilled_sha256: str,
) -> dict[str, Any]:
    maximum = int(config["training"]["train_max_records_per_route"])
    human = load_route_dataset(human_path, expected_sha256=human_sha256, split="train", max_records_per_route=maximum)
    distilled = load_route_dataset(distilled_path, expected_sha256=distilled_sha256, split="train", max_records_per_route=maximum)
    weights = {route: float(config["training"]["route_weight"]) for route in ROUTE_ORDER}
    human_sampler = DeterministicRouteSampler(human, weights, int(config["training"]["seed"]))
    distilled_sampler = DeterministicRouteSampler(distilled, weights, int(config["training"]["seed"]))
    exposures = int(config["training"]["dry_run_source_exposures"])
    human_ids = [human_sampler.next_sample().record["sample_id"] for _ in range(exposures)]
    distilled_ids = [distilled_sampler.next_sample().record["sample_id"] for _ in range(exposures)]
    if human_ids != distilled_ids:
        raise DistillationABError("dry-run source exposure sequences differ")
    return {
        "status": "passed", "source_exposures": exposures,
        "source_sequence_sha256": config_sha256(human_ids),
        "human_selection_sha256": human.selection_sha256,
        "distilled_selection_sha256": distilled.selection_sha256,
        "selection_identity_exact": human.selection_sha256 == distilled.selection_sha256,
        "sampler_state_identity_exact": config_sha256(human_sampler.state_dict()) == config_sha256(distilled_sampler.state_dict()),
        "optimizer_step_boundary": config["training"]["profile_sha256"],
        "student_initial_state_dict_sha256": config["student"]["initial_state_dict_sha256"],
        "evaluation_protocol_sha256": config["evaluation"]["protocol_sha256"],
        "evaluation_split": "dev",
        "test_access": config["evaluation"]["test_access"],
    }


def compare_recipe_configs(
    human: Mapping[str, Any], distilled: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, bool]:
    human_copy = json.loads(json.dumps(human))
    distilled_copy = json.loads(json.dumps(distilled))
    expected_names = {"mvp-training-m2-human-only-v1", "mvp-training-m2-distilled-v1"}
    if {human_copy["identity"]["name"], distilled_copy["identity"]["name"]} != expected_names:
        raise DistillationABError("recipe arm identities changed")
    for value in (human_copy, distilled_copy):
        value["identity"]["name"] = "<arm>"
        value["data"]["train_path"] = "<arm-train>"
        value["data"]["train_sha256"] = "<arm-train-sha256>"
    if human_copy != distilled_copy:
        raise DistillationABError("recipes differ outside arm train target/provenance identity")
    profile_optimization = dict(profile["optimization"])
    profile_optimization.pop("checkpoint_retention")
    if dict(human["resource_profile"]) != dict(profile["resource_profile"]):
        raise DistillationABError("recipe resource profile differs from TD-14")
    if dict(human["optimization"]) != profile_optimization:
        raise DistillationABError("recipe optimization differs from TD-14")
    return {
        "only_allowed_recipe_fields_differ": True,
        "resource_profile_exact": True,
        "optimization_budget_exact": True,
    }


def _stream_pair_validation(human_path: Path, distilled_path: Path) -> dict[str, Any]:
    records = 0
    target_differences = 0
    source_digest = hashlib.sha256()
    with human_path.open("r", encoding="utf-8") as human_handle, distilled_path.open("r", encoding="utf-8") as distilled_handle:
        for line_number, (human_line, distilled_line) in enumerate(
            zip(human_handle, distilled_handle, strict=True), start=1
        ):
            human = json.loads(human_line)
            distilled = json.loads(distilled_line)
            human_shared = {key: value for key, value in human.items() if key not in {"target_text", "provenance"}}
            distilled_shared = {key: value for key, value in distilled.items() if key not in {"target_text", "provenance"}}
            if human_shared != distilled_shared:
                raise DistillationABError(f"A/B row {line_number} differs outside target/provenance")
            if human["provenance"].get("ab_arm") != "human-only" or distilled["provenance"].get("ab_arm") != "distilled":
                raise DistillationABError(f"A/B row {line_number} has wrong arm provenance")
            target_differences += human["target_text"] != distilled["target_text"]
            source_digest.update((human["sample_id"] + "\n").encode("utf-8"))
            records += 1
    if records == 0:
        raise DistillationABError("A/B recipe corpora are empty")
    return {
        "records": records,
        "source_rows_exact": True,
        "target_or_provenance_only": True,
        "target_difference_records": target_differences,
        "target_difference_rate": target_differences / records,
        "source_sequence_stream_sha256": source_digest.hexdigest(),
    }


def validate_ab_release(
    *, repository_root: Path, config_path: Path, human_recipe_path: Path,
    distilled_recipe_path: Path, report_path: Path,
) -> dict[str, Any]:
    from mvp_training import load_training_config, run_training

    config = load_ab_config(config_path)
    profile_path = _verified(repository_root, config["training"]["profile"], config["training"]["profile_sha256"], "M2 profile")
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    human_recipe = load_training_config(human_recipe_path)
    distilled_recipe = load_training_config(distilled_recipe_path)
    fairness = compare_recipe_configs(human_recipe, distilled_recipe, profile)
    manifest_path = _repo_path(repository_root, config["outputs"]["manifest"], "A/B manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest["config"]["file_sha256"] != sha256_file(config_path):
        raise DistillationABError("A/B cohort manifest is incomplete or belongs to another config")
    human_path = _verified(repository_root, human_recipe["data"]["train_path"], human_recipe["data"]["train_sha256"], "human recipe train")
    distilled_path = _verified(repository_root, distilled_recipe["data"]["train_path"], distilled_recipe["data"]["train_sha256"], "distilled recipe train")
    if human_recipe["data"]["manifest_sha256"] != sha256_file(manifest_path) or distilled_recipe["data"]["manifest_sha256"] != sha256_file(manifest_path):
        raise DistillationABError("recipe manifest identity differs from the A/B cohort")
    row_validation = _stream_pair_validation(human_path, distilled_path)
    if row_validation["records"] != manifest["intersection"]["records"]:
        raise DistillationABError("recipe record count differs from the common cohort")
    dry_pair = dry_run_pair(
        repository_root=repository_root, config=config,
        human_path=human_path, distilled_path=distilled_path,
        human_sha256=human_recipe["data"]["train_sha256"],
        distilled_sha256=distilled_recipe["data"]["train_sha256"],
    )
    human_dry = run_training(
        config_path=human_recipe_path, repository_root=repository_root,
        output_dir=None, dry_run=True,
    )
    distilled_dry = run_training(
        config_path=distilled_recipe_path, repository_root=repository_root,
        output_dir=None, dry_run=True,
    )
    runtime_keys = {
        "selected_device", "selected_precision", "device_total_bytes", "host_total_bytes",
        "resource_validation", "packages",
    }
    runtime_exact = {
        key: human_dry["runtime"][key] == distilled_dry["runtime"][key]
        for key in runtime_keys
    }
    gates = {
        **fairness,
        "common_cohort_complete": manifest["intersection"]["routes"] == 20 and manifest["intersection"]["minimum_route_records"] >= config["distilled"]["minimum_accepted_per_route"],
        "teacher_filtered_excluded_from_both": manifest["intersection"]["teacher_rejected_or_filtered_included"] == 0,
        "source_rows_exact": row_validation["source_rows_exact"],
        "dry_run_source_sequence_exact": dry_pair["selection_identity_exact"] and dry_pair["sampler_state_identity_exact"],
        "initial_student_exact": manifest["student_initial_state_dict_sha256"] == config["student"]["initial_state_dict_sha256"],
        "runtime_budget_exact": all(runtime_exact.values()),
        "dev_human_reference_only": config["evaluation"]["references"] == "human_only",
        "test_access_blocked": config["evaluation"]["test_access"] == "forbidden_until_unique_td16_candidate_is_frozen",
        "chinese_tags_separate_in_selection": config["dev_selection"]["distilled_vs_human"]["evaluate_zho_Hans_and_zho_Hant_separately"] is True,
    }
    if not all(gates.values()):
        raise DistillationABError(f"A/B fairness gates failed: {[key for key, value in gates.items() if not value]}")
    report = {
        "schema_version": 1, "status": "complete", "task": "TD-15",
        "contract": {"path": config_path.relative_to(repository_root).as_posix(), "file_sha256": sha256_file(config_path), "canonical_sha256": config_sha256(config)},
        "cohort": {"manifest_path": config["outputs"]["manifest"], "manifest_sha256": sha256_file(manifest_path), "intersection": manifest["intersection"], "files": manifest["files"]},
        "recipes": {
            "human-only": {"path": human_recipe_path.relative_to(repository_root).as_posix(), "file_sha256": sha256_file(human_recipe_path), "canonical_sha256": config_sha256(human_recipe), "train_sha256": human_recipe["data"]["train_sha256"]},
            "distilled": {"path": distilled_recipe_path.relative_to(repository_root).as_posix(), "file_sha256": sha256_file(distilled_recipe_path), "canonical_sha256": config_sha256(distilled_recipe), "train_sha256": distilled_recipe["data"]["train_sha256"]},
        },
        "frozen_identities": {
            "student_initial_state_dict_sha256": config["student"]["initial_state_dict_sha256"],
            "training_profile_sha256": config["training"]["profile_sha256"],
            "evaluation_protocol_sha256": config["evaluation"]["protocol_sha256"],
            "source_order_sha256": manifest["intersection"]["source_order_sha256"],
        },
        "pretraining_difference": {"route20": manifest["route20"], "totals": manifest["totals"], "row_validation": row_validation},
        "dry_runs": {"paired_sampler": dry_pair, "human": {"status": human_dry["status"], "training_config": human_dry["training_config"]}, "distilled": {"status": distilled_dry["status"], "training_config": distilled_dry["training_config"]}, "runtime_exact": runtime_exact},
        "dev_selection": config["dev_selection"],
        "test_access": {"policy": config["evaluation"]["test_access"], "runs_allowed_after_selection": config["evaluation"]["test_runs_allowed_after_selection"], "accessed_by_td15": False},
        "gates": gates,
    }
    descriptor, name = tempfile.mkstemp(prefix=f".{report_path.name}.", suffix=".tmp", dir=report_path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, report_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return report
