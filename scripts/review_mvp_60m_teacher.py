"""Build and verify the fixed TD-04 teacher-output manual review queue."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml

from model_training_contract import directed_routes
from mvp_60m_data_pipeline import (
    AbilityDataError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    stable_rank,
    write_json,
    write_jsonl,
)


ROUTES = tuple(f"{source}->{target}" for source, target in directed_routes())


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise AbilityDataError("TD-04 review config schema differs")
    if value.get("identity", {}).get("status") != "frozen":
        raise AbilityDataError("TD-04 review config is not frozen")
    if value.get("decisions", {}).get("formal_test_access") != "prohibited":
        raise AbilityDataError("formal test access must remain prohibited")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AbilityDataError(f"non-object JSONL: {path}:{line_number}")
            rows.append(value)
    return rows


def _route(row: Mapping[str, Any]) -> str:
    return f"{row['src_lang']}->{row['tgt_lang']}"


def build_queue(
    accepted: Sequence[Mapping[str, Any]],
    filtered: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_kind: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "accepted": defaultdict(list),
        "filtered": defaultdict(list),
    }
    for row in accepted:
        by_kind["accepted"][_route(row)].append(row)
    for row in filtered:
        route = str(row.get("route") or _route(row))
        by_kind["filtered"][route].append(row)
    if set(by_kind["accepted"]) != set(ROUTES):
        raise AbilityDataError("manual review input does not cover 20 accepted routes")

    seed = str(config["sampling"]["seed"])
    limits = {
        "accepted": int(config["sampling"]["accepted_per_route"]),
        "filtered": int(config["sampling"]["filtered_per_route"]),
    }
    queue: list[dict[str, Any]] = []
    for route in ROUTES:
        for kind in ("accepted", "filtered"):
            values = sorted(
                by_kind[kind].get(route, []),
                key=lambda row: stable_rank(
                    seed, kind, route, str(row.get("record_id") or row.get("job_id"))
                ),
            )
            selected = values[: limits[kind]]
            if kind == "accepted" and len(selected) != limits[kind]:
                raise AbilityDataError(f"{route} lacks accepted manual-review records")
            for row in selected:
                input_id = str(row.get("record_id") or row.get("job_id"))
                target_text = str(row.get("target_text") or row.get("normalized_output") or "")
                review_id = sha256_bytes(canonical_json_bytes([seed, kind, route, input_id]))
                queue.append(
                    {
                        "review_id": review_id,
                        "kind": kind,
                        "route": route,
                        "input_record_id": input_id,
                        "source_text": str(row["source_text"]),
                        "target_text": target_text,
                        "automated_rejection_reasons": list(row.get("rejection_reasons", [])),
                    }
                )
    queue.sort(key=lambda row: (ROUTES.index(str(row["route"])), str(row["kind"]), str(row["review_id"])))
    return queue


def create_queue(runtime_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    manifest_path = runtime_root / PurePosixPath(inputs["td04_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise AbilityDataError("TD-04 must be complete before manual review")
    accepted_path = runtime_root / PurePosixPath(inputs["accepted_teacher"])
    filtered_path = runtime_root / PurePosixPath(inputs["filtered_teacher"])
    if sha256_file(accepted_path) != manifest["accepted_teacher"]["sha256"]:
        raise AbilityDataError("accepted teacher hash drift")
    if sha256_file(filtered_path) != manifest["filtered_teacher"]["sha256"]:
        raise AbilityDataError("filtered teacher hash drift")
    queue = build_queue(read_jsonl(accepted_path), read_jsonl(filtered_path), config)
    output = runtime_root / PurePosixPath(config["outputs"]["queue"])
    records, digest = write_jsonl(output, queue)
    result = {
        "status": "review-pending",
        "records": records,
        "sha256": digest,
        "counts": dict(sorted(Counter(f"{row['route']}|{row['kind']}" for row in queue).items())),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def verify_decisions(runtime_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    outputs = config["outputs"]
    queue_path = runtime_root / PurePosixPath(outputs["queue"])
    decisions_path = runtime_root / PurePosixPath(outputs["decisions"])
    queue = read_jsonl(queue_path)
    decisions = read_jsonl(decisions_path)
    expected = {str(row["review_id"]): row for row in queue}
    actual: dict[str, dict[str, Any]] = {}
    allowed = set(config["decisions"]["allowed"])
    required = set(config["decisions"]["required_fields"])
    for row in decisions:
        if not required.issubset(row):
            raise AbilityDataError("manual review decision fields are incomplete")
        review_id = str(row["review_id"])
        if review_id not in expected or review_id in actual:
            raise AbilityDataError("manual review decision identity mismatch")
        if row["decision"] not in allowed:
            raise AbilityDataError("manual review decision is unsupported")
        if not isinstance(row["category"], str) or not isinstance(row["note"], str):
            raise AbilityDataError("manual review category/note must be strings")
        actual[review_id] = row
    if set(actual) != set(expected):
        raise AbilityDataError(f"manual review decisions incomplete: {len(actual)}/{len(expected)}")
    counts = Counter(str(row["decision"]) for row in decisions)
    blockers = [row for row in decisions if row["decision"] == "block"]
    report = {
        "schema_version": 1,
        "status": "blocked" if blockers else "complete",
        "task": "TD-04-manual-review",
        "formal_test_accessed": False,
        "queue": {"records": len(queue), "sha256": sha256_file(queue_path)},
        "decisions": {"records": len(decisions), "sha256": sha256_file(decisions_path)},
        "decision_counts": dict(sorted(counts.items())),
        "route_kind_counts": dict(sorted(Counter(f"{row['route']}|{row['kind']}" for row in queue).items())),
        "warnings": [row for row in decisions if row["decision"] == "warning"],
        "blockers": blockers,
    }
    write_json(runtime_root / PurePosixPath(outputs["report"]), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if blockers:
        raise AbilityDataError(f"manual teacher review found {len(blockers)} blockers")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("queue", "verify"))
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/mvp_60m_teacher_review.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    if args.action == "queue":
        create_queue(args.runtime_root.resolve(), config)
    else:
        verify_decisions(args.runtime_root.resolve(), config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
