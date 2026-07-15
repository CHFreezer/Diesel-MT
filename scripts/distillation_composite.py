"""Validate and materialize the immutable 20-route D1 distilled composite."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from hymt2_distillation import atomic_write_bytes, atomic_write_json, canonical_json_bytes, sha256_file
from model_training_contract import (
    config_sha256,
    directed_routes,
    load_model_data_config,
    validate_parallel_sample,
)


COMPOSITE_IDENTITY = {
    "name": "hymt2-sequence-distillation-d1-20route-composite-v2",
    "status": "frozen",
    "scope": "mvp-train-only-twenty-route-distilled-composite",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CompositeError(RuntimeError):
    """Raised when a component or composite identity is incomplete."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise CompositeError(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        raise CompositeError(f"{context} unknown fields: {', '.join(unknown)}")


def _repo_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CompositeError(f"{context} must be a repository-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise CompositeError(f"{context} escapes the repository")
    return path.as_posix()


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise CompositeError(f"{context} must be a lowercase SHA-256")
    return value


def load_composite_config(path: Path) -> dict[str, Any]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CompositeError(f"cannot load composite config: {exc}") from exc
    if not isinstance(config, dict):
        raise CompositeError("composite config must be a mapping")
    _exact_keys(
        config,
        {"schema_version", "identity", "components", "gates", "human_m0", "outputs"},
        "composite config",
    )
    if config["schema_version"] != 1 or config["identity"] != COMPOSITE_IDENTITY:
        raise CompositeError("composite schema or identity changed")

    expected_routes = tuple(f"{source}->{target}" for source, target in directed_routes())
    components = config["components"]
    if not isinstance(components, list) or len(components) != 2:
        raise CompositeError("composite must contain the frozen v1 base and v2 addendum")
    seen_routes: set[str] = set()
    seen_ids: set[str] = set()
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            raise CompositeError(f"components[{index}] must be a mapping")
        _exact_keys(
            component,
            {
                "component_id",
                "manifest",
                "manifest_sha256",
                "accepted",
                "accepted_sha256",
                "accepted_records",
                "routes",
            },
            f"components[{index}]",
        )
        component_id = str(component["component_id"])
        if not component_id or component_id in seen_ids:
            raise CompositeError("component ids must be non-empty and unique")
        seen_ids.add(component_id)
        _repo_path(component["manifest"], f"{component_id}.manifest")
        _repo_path(component["accepted"], f"{component_id}.accepted")
        _sha256(component["manifest_sha256"], f"{component_id}.manifest_sha256")
        _sha256(component["accepted_sha256"], f"{component_id}.accepted_sha256")
        if not isinstance(component["accepted_records"], int) or component["accepted_records"] <= 0:
            raise CompositeError(f"{component_id}.accepted_records must be positive")
        routes = component["routes"]
        if not isinstance(routes, list) or not routes or len(routes) != len(set(routes)):
            raise CompositeError(f"{component_id}.routes must be a non-empty unique list")
        if any(route not in expected_routes for route in routes):
            raise CompositeError(f"{component_id}.routes contains an unknown route")
        overlap = seen_routes & set(routes)
        if overlap:
            raise CompositeError(f"component routes overlap: {', '.join(sorted(overlap))}")
        seen_routes.update(routes)
    if seen_routes != set(expected_routes):
        raise CompositeError("component routes do not cover the complete 20-route contract")

    gates = config["gates"]
    if gates != {
        "required_routes": 20,
        "minimum_accepted_per_route": 2000,
        "require_disjoint_component_routes": True,
        "require_zero_dev_test_records": True,
        "require_teacher_synthetic_provenance": True,
    }:
        raise CompositeError("composite gates changed")
    human_m0 = config["human_m0"]
    if not isinstance(human_m0, dict):
        raise CompositeError("human_m0 must be a mapping")
    _exact_keys(human_m0, {"manifest", "manifest_sha256"}, "human_m0")
    _repo_path(human_m0["manifest"], "human_m0.manifest")
    _sha256(human_m0["manifest_sha256"], "human_m0.manifest_sha256")
    outputs = config["outputs"]
    if not isinstance(outputs, dict):
        raise CompositeError("outputs must be a mapping")
    _exact_keys(outputs, {"root", "accepted", "manifest", "evidence"}, "outputs")
    for name, value in outputs.items():
        _repo_path(value, f"outputs.{name}")
    root = str(outputs["root"]).rstrip("/")
    if not str(outputs["accepted"]).startswith(root + "/") or not str(outputs["manifest"]).startswith(root + "/"):
        raise CompositeError("composite data outputs must stay under outputs.root")
    return config


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompositeError(f"cannot load {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompositeError(f"{context} must contain an object")
    return value


def _load_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompositeError(f"cannot load {context}: {exc}") from exc
    if not records or any(not isinstance(record, dict) for record in records):
        raise CompositeError(f"{context} must contain JSON objects")
    return records


def build_composite(repository_root: Path, config_path: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    config = load_composite_config(config_path)
    data_config = load_model_data_config(repository_root / "configs" / "mvp_model_data.yaml")

    human_path = repository_root / PurePosixPath(config["human_m0"]["manifest"])
    if sha256_file(human_path) != config["human_m0"]["manifest_sha256"]:
        raise CompositeError("20-route human M0 manifest differs from the composite lock")
    human_manifest = _load_json(human_path, "human M0 manifest")
    if human_manifest.get("status") != "complete":
        raise CompositeError("20-route human M0 is not complete")

    route_order = {
        f"{source}->{target}": index
        for index, (source, target) in enumerate(directed_routes())
    }
    all_records: list[dict[str, Any]] = []
    component_evidence: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    created_at_values: list[str] = []
    for component in config["components"]:
        component_id = str(component["component_id"])
        manifest_path = repository_root / PurePosixPath(component["manifest"])
        accepted_path = repository_root / PurePosixPath(component["accepted"])
        if sha256_file(manifest_path) != component["manifest_sha256"]:
            raise CompositeError(f"{component_id} manifest differs from the lock")
        if sha256_file(accepted_path) != component["accepted_sha256"]:
            raise CompositeError(f"{component_id} accepted corpus differs from the lock")
        manifest = _load_json(manifest_path, f"{component_id} manifest")
        if manifest.get("status") != "complete" or manifest.get("scope", {}).get("accepted") != component["accepted_records"]:
            raise CompositeError(f"{component_id} manifest is incomplete or has a wrong count")
        output_identity = manifest.get("outputs", {}).get("accepted", {})
        if (
            output_identity.get("path") != component["accepted"]
            or output_identity.get("sha256") != component["accepted_sha256"]
            or output_identity.get("records") != component["accepted_records"]
        ):
            raise CompositeError(f"{component_id} manifest does not bind the accepted corpus")
        records = _load_jsonl(accepted_path, f"{component_id} accepted corpus")
        if len(records) != component["accepted_records"]:
            raise CompositeError(f"{component_id} accepted record count differs")
        counts: Counter[str] = Counter()
        for record in records:
            validate_parallel_sample(record, data_config)
            route = f"{record['src_lang']}->{record['tgt_lang']}"
            counts[route] += 1
            if record["split"] != "train" or record.get("provenance", {}).get("kind") != "teacher_synthetic":
                raise CompositeError(f"{component_id} contains non-train or non-teacher data")
            sample_id = str(record["sample_id"])
            if sample_id in seen_sample_ids:
                raise CompositeError(f"duplicate composite sample id: {sample_id}")
            seen_sample_ids.add(sample_id)
            all_records.append(record)
        if set(counts) != set(component["routes"]):
            raise CompositeError(f"{component_id} actual route coverage differs from the lock")
        component_evidence.append(
            {
                "component_id": component_id,
                "manifest": component["manifest"],
                "manifest_sha256": component["manifest_sha256"],
                "accepted": component["accepted"],
                "accepted_sha256": component["accepted_sha256"],
                "accepted_records": len(records),
                "route_counts": dict(sorted(counts.items(), key=lambda item: route_order[item[0]])),
            }
        )
        created_at_values.append(str(manifest.get("created_at", "")))

    route_counts = Counter(f"{record['src_lang']}->{record['tgt_lang']}" for record in all_records)
    expected_routes = set(route_order)
    if set(route_counts) != expected_routes or any(
        count < int(config["gates"]["minimum_accepted_per_route"])
        for count in route_counts.values()
    ):
        raise CompositeError("composite route coverage or minimum accepted gate failed")
    all_records.sort(
        key=lambda record: (
            route_order[f"{record['src_lang']}->{record['tgt_lang']}"],
            str(record["sample_id"]),
        )
    )
    accepted_bytes = b"".join(canonical_json_bytes(record) for record in all_records)
    accepted_path = repository_root / PurePosixPath(config["outputs"]["accepted"])
    manifest_path = repository_root / PurePosixPath(config["outputs"]["manifest"])
    manifest_path.unlink(missing_ok=True)
    atomic_write_bytes(accepted_path, accepted_bytes)
    accepted_sha256 = sha256_file(accepted_path)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "release": "d1-hymt2-distillation-20route-mvp-v2",
        "identity": config["identity"],
        "created_at": max(created_at_values),
        "composite_config_sha256": config_sha256(config),
        "human_m0_manifest_sha256": config["human_m0"]["manifest_sha256"],
        "scope": {
            "routes": 20,
            "input": len(all_records),
            "accepted": len(all_records),
            "teacher_synthetic": len(all_records),
            "dev_records": 0,
            "test_records": 0,
        },
        "route_counts": {
            route: int(route_counts[route])
            for route in route_order
        },
        "components": component_evidence,
        "outputs": {
            "accepted": {
                "path": config["outputs"]["accepted"],
                "records": len(all_records),
                "bytes": len(accepted_bytes),
                "sha256": accepted_sha256,
            }
        },
    }
    atomic_write_json(manifest_path, manifest)
    evidence = {
        "schema_version": 1,
        "status": "complete",
        "release": manifest["release"],
        "corpus_maturity": "mvp-20route-composite",
        "scope": manifest["scope"],
        "route_counts": manifest["route_counts"],
        "components": component_evidence,
        "identities": {
            "composite_config_sha256": manifest["composite_config_sha256"],
            "human_m0_manifest_sha256": manifest["human_m0_manifest_sha256"],
            "accepted_sha256": accepted_sha256,
            "manifest_sha256": sha256_file(manifest_path),
        },
        "test_accessed": False,
        "dev_accessed": False,
        "downstream_consumer": "TD-15",
        "td08_completed": True,
        "td09_started": False,
    }
    evidence_path = repository_root / PurePosixPath(config["outputs"]["evidence"])
    atomic_write_json(evidence_path, evidence)
    return evidence
