"""Validate the complete, immutable TD-02 through TD-05 ability-first chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from model_data_source_contract import canonical_sha256
from model_training_contract import directed_routes
from mvp_60m_data_pipeline import AbilityDataError, sha256_file, write_json


ROUTES = {f"{source}->{target}" for source, target in directed_routes()}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AbilityDataError(f"JSON object expected: {path}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise AbilityDataError("data-chain config schema differs")
    if value.get("identity", {}).get("status") != "frozen":
        raise AbilityDataError("data-chain config is not frozen")
    requirements = value.get("requirements", {})
    if requirements.get("formal_test_access") != "prohibited":
        raise AbilityDataError("formal test access must remain prohibited")
    if requirements.get("formal_devtest_access") != "prohibited":
        raise AbilityDataError("formal devtest access must remain prohibited")
    return value


def _verify_config(repository_root: Path, spec: Mapping[str, Any]) -> None:
    path = repository_root / PurePosixPath(str(spec["path"]))
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if canonical_sha256(value) != spec["canonical_sha256"]:
        raise AbilityDataError(f"canonical config hash drift: {path}")


def _require_complete(manifest: Mapping[str, Any], task: str) -> None:
    if manifest.get("status") != "complete":
        raise AbilityDataError(f"{task} is not complete")
    if manifest.get("formal_test_accessed") is not False:
        raise AbilityDataError(f"{task} formal-test isolation is unproven")


def validate(
    repository_root: Path, runtime_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    repo = config["repository_inputs"]
    runtime = config["runtime_inputs"]
    required = config["requirements"]
    _verify_config(repository_root, repo["source_config"])
    _verify_config(repository_root, repo["teacher_config"])
    _verify_config(repository_root, repo["review_config"])
    _verify_config(repository_root, repo["mixed_config"])
    source_lock = repository_root / PurePosixPath(repo["source_lock"]["path"])
    if sha256_file(source_lock) != repo["source_lock"]["file_sha256"]:
        raise AbilityDataError("source byte lock hash drift")

    td02_path = repository_root / PurePosixPath(repo["td02_report"])
    td02 = _json(td02_path)
    _require_complete(td02, "TD-02")
    if td02["policy"].get("formal_devtest") != "not opened":
        raise AbilityDataError("TD-02 formal-devtest isolation is unproven")
    if td02["selection"]["records"] != int(required["native_hant_source_records"]):
        raise AbilityDataError("TD-02 native Hant count drift")
    td02_selection = runtime_root / PurePosixPath(runtime["td02_selection"])
    if sha256_file(td02_selection) != td02["selection"]["sha256"]:
        raise AbilityDataError("TD-02 native Hant selection hash drift")
    if td02["selection"]["domain_shares"]["technical"] > 0.15:
        raise AbilityDataError("TD-02 technical share exceeds its ceiling")
    if td02["selection"]["domain_shares"]["legal_and_government"] > 0.20:
        raise AbilityDataError("TD-02 legal/government share exceeds its ceiling")

    td03_path = runtime_root / PurePosixPath(runtime["td03_manifest"])
    td03 = _json(td03_path)
    _require_complete(td03, "TD-03")
    td03_compact = _json(repository_root / PurePosixPath(repo["td03_compact"]))
    _require_complete(td03_compact, "TD-03 compact evidence")
    if td03["source_config_sha256"] != repo["source_config"]["canonical_sha256"]:
        raise AbilityDataError("TD-03 source config identity drift")
    if td03["source_lock_sha256"] != repo["source_lock"]["file_sha256"]:
        raise AbilityDataError("TD-03 source lock identity drift")
    counts = td03["source_bank"]["counts"]
    for language in ("eng_Latn", "zho_Hans", "jpn_Jpan", "kor_Hang"):
        if counts.get(language) != int(required["non_hant_source_records_per_language"]):
            raise AbilityDataError(f"TD-03 {language} source count drift")
    if counts.get("zho_Hant") != int(required["native_hant_source_records"]):
        raise AbilityDataError("TD-03 native Hant source count drift")
    if td03["human_anchors"]["records"] != int(required["human_anchor_records"]):
        raise AbilityDataError("TD-03 human anchor count drift")
    for section in ("source_bank", "human_anchors"):
        for field in ("records", "sha256"):
            if td03_compact[section].get(field) != td03[section].get(field):
                raise AbilityDataError(f"TD-03 compact/runtime binding failed: {section}.{field}")
    for field in ("source_anchor_group_overlap", "exact_or_near_overlap", "flores_dev_contamination"):
        if td03["invariants"].get(field) != 0:
            raise AbilityDataError(f"TD-03 invariant failed: {field}")
    if td03["invariants"].get("zero_truncation") is not True:
        raise AbilityDataError("TD-03 zero-truncation gate failed")
    for key, section in (("td03_source_bank", "source_bank"), ("td03_human_anchors", "human_anchors")):
        if sha256_file(runtime_root / PurePosixPath(runtime[key])) != td03[section]["sha256"]:
            raise AbilityDataError(f"TD-03 runtime hash drift: {section}")

    td04_path = runtime_root / PurePosixPath(runtime["td04_manifest"])
    td04 = _json(td04_path)
    _require_complete(td04, "TD-04")
    if td04["generation_config_sha256"] != repo["teacher_config"]["canonical_sha256"]:
        raise AbilityDataError("TD-04 generation config identity drift")
    file_sections = {
        "td04_raw": "raw",
        "td04_accepted": "accepted_teacher",
        "td04_filtered": "filtered_teacher",
        "td04_reverse": "reverse_pairs",
    }
    for key, section in file_sections.items():
        if sha256_file(runtime_root / PurePosixPath(runtime[key])) != td04[section]["sha256"]:
            raise AbilityDataError(f"TD-04 runtime hash drift: {section}")
    fixed = [
        summary for route, summary in td04["routes"].items()
        if not route.startswith("zho_Hant->") and summary["accepted"] == int(required["fixed_teacher_records_per_route"])
    ]
    if len(fixed) != int(required["fixed_non_hant_teacher_routes"]):
        raise AbilityDataError("TD-04 fixed route gate failed")
    if set(td04["routes"]) != ROUTES:
        raise AbilityDataError("TD-04 route matrix differs from 20 routes")
    if td04["reverse_pairs"].get("counts_as_native_hant") is not False:
        raise AbilityDataError("TD-04 reverse pairs were misclassified as native Hant")
    for route, count in td04["reverse_pairs"]["route_counts"].items():
        original = int(td04["routes"][route]["accepted"])
        if int(count) > int(original * 0.50):
            raise AbilityDataError(f"TD-04 reverse-pair ceiling failed: {route}")
    for field, expected in (
        ("fixed_non_hant_routes_at_10000", int(required["fixed_non_hant_teacher_routes"])),
        ("outgoing_hant_quality_actual_no_refill", True),
        ("finite_text_outputs", True),
        ("second_teacher_call_for_reverse", False),
    ):
        if td04["invariants"].get(field) != expected:
            raise AbilityDataError(f"TD-04 invariant failed: {field}")

    review_path = runtime_root / PurePosixPath(runtime["td04_review_report"])
    review = _json(review_path)
    _require_complete(review, "TD-04 manual review")
    if review.get("blockers"):
        raise AbilityDataError("TD-04 manual review has blockers")
    for route in ROUTES:
        if review["route_kind_counts"].get(f"{route}|accepted") != 20:
            raise AbilityDataError(f"TD-04 accepted manual review count failed: {route}")
        filtered_count = int(review["route_kind_counts"].get(f"{route}|filtered", 0))
        if filtered_count < 0 or filtered_count > 20:
            raise AbilityDataError(f"TD-04 filtered manual review count failed: {route}")
    for key, section in (("td04_review_queue", "queue"), ("td04_review_decisions", "decisions")):
        if sha256_file(runtime_root / PurePosixPath(runtime[key])) != review[section]["sha256"]:
            raise AbilityDataError(f"TD-04 review hash drift: {section}")

    td05_path = runtime_root / PurePosixPath(runtime["td05_manifest"])
    td05 = _json(td05_path)
    _require_complete(td05, "TD-05")
    if sha256_file(runtime_root / PurePosixPath(runtime["td05_corpus"])) != td05["corpus"]["sha256"]:
        raise AbilityDataError("TD-05 corpus hash drift")
    if sha256_file(runtime_root / PurePosixPath(runtime["td05_sampling_plan"])) != td05["sampling_plan"]["sha256"]:
        raise AbilityDataError("TD-05 sampling plan hash drift")
    sampling_plan = _json(runtime_root / PurePosixPath(runtime["td05_sampling_plan"]))
    if not (
        td05["sampling_plan"]["teacher_weight"] == float(required["teacher_sampling_weight"])
        and td05["sampling_plan"]["human_weight"] == float(required["human_sampling_weight"])
    ):
        raise AbilityDataError("TD-05 sampling weights drift")
    preview = sampling_plan.get("preview", {})
    preview_exposures = int(preview.get("exposures", -1))
    expected_teacher = round(preview_exposures * float(required["teacher_sampling_weight"]))
    expected_human = preview_exposures - expected_teacher
    if preview.get("class_counts") != {"human": expected_human, "teacher": expected_teacher}:
        raise AbilityDataError("TD-05 sampling preview is not exact 80/20")
    if td05["corpus"]["records"] != td05["audit"]["records"]:
        raise AbilityDataError("TD-05 corpus/audit record count drift")
    for field, expected in (
        ("twenty_teacher_routes", True),
        ("zero_truncation", True),
        ("raw_duplicate_fill", False),
        ("teacher_human_group_overlap", 0),
        ("manifest_published_last", True),
    ):
        if td05["invariants"].get(field) != expected:
            raise AbilityDataError(f"TD-05 invariant failed: {field}")
    td05_compact_path = repository_root / PurePosixPath(repo["td05_compact"])
    td05_compact = _json(td05_compact_path)
    _require_complete(td05_compact, "TD-05 compact evidence")
    if td05_compact.get("runtime_manifest_sha256") != sha256_file(td05_path):
        raise AbilityDataError("TD-05 compact evidence does not bind runtime manifest")

    return {
        "schema_version": 1,
        "status": "complete",
        "scope": "TD-02-through-TD-05",
        "formal_test_accessed": False,
        "formal_devtest_accessed": False,
        "evidence": {
            "td02_report_sha256": sha256_file(td02_path),
            "td03_manifest_sha256": sha256_file(td03_path),
            "td04_manifest_sha256": sha256_file(td04_path),
            "td04_review_report_sha256": sha256_file(review_path),
            "td05_manifest_sha256": sha256_file(td05_path),
            "td05_corpus_sha256": td05["corpus"]["sha256"],
        },
        "counts": {
            "native_hant_sources": td02["selection"]["records"],
            "source_bank_records": td03["source_bank"]["records"],
            "human_anchor_records": td03["human_anchors"]["records"],
            "teacher_raw_records": td04["raw"]["records"],
            "teacher_accepted_records": td04["accepted_teacher"]["records"],
            "reverse_pair_records": td04["reverse_pairs"]["records"],
            "mixed_corpus_records": td05["corpus"]["records"],
        },
        "sampling": {"teacher": 0.80, "human": 0.20},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/mvp_60m_data_chain.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    config_path = args.config.resolve() if args.config.is_absolute() else (repository_root / args.config).resolve()
    config = load_config(config_path)
    report = validate(repository_root, args.runtime_root.resolve(), config)
    write_json(repository_root / PurePosixPath(config["output"]["report"]), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
