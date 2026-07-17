"""TD-04 high-throughput, resumable Hy-MT2 generation for the 60M MVP."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import yaml

from hymt2_distillation import (
    LlamaCppTeacher,
    build_prompt,
    filter_output,
    load_prompt_config,
    route_limits,
)
from model_data_source_contract import canonical_sha256
from mvp_60m_data_pipeline import (
    MODEL_TAGS,
    AbilityDataError,
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    stable_rank,
    write_json,
    write_jsonl,
)


_NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d[\d,.:/-]*\d|\d)(?!\w)")
_PLACEHOLDER_RE = re.compile(r"(?:\{\{[^{}]+\}\}|\{[^{}]+\}|<[^<>]+>|\[[A-Z][A-Z0-9_ -]*\])")
_CONVERSION_ROUTES = {"zho_Hans->zho_Hant", "zho_Hant->zho_Hans"}


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise AbilityDataError("TD-04 config schema differs")
    if value.get("identity", {}).get("status") != "frozen":
        raise AbilityDataError("TD-04 config is not frozen")
    if value.get("generation", {}).get("formal_test_access") != "prohibited":
        raise AbilityDataError("formal test access must remain prohibited")
    if int(value["generation"]["parallel_slots"]) > 96:
        raise AbilityDataError("TD-04 parallel slots exceed the measured safe profile")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AbilityDataError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise AbilityDataError(f"non-object JSONL {path}:{line_number}")
            rows.append(value)
    return rows


def build_jobs(
    source_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_language: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_language[str(row["language_tag"])].append(row)
    seed = str(config["generation"]["source_selection_seed"])
    scan_count = int(config["generation"]["candidate_scan_per_fixed_route"])
    jobs: list[dict[str, Any]] = []
    for source in MODEL_TAGS:
        values = by_language[source]
        for target in MODEL_TAGS:
            if source == target:
                continue
            route = f"{source}->{target}"
            ranked = sorted(
                values,
                key=lambda row: stable_rank(seed, route, str(row["record_id"])),
            )
            selected = ranked if source == "zho_Hant" else ranked[:scan_count]
            if source != "zho_Hant" and len(selected) != scan_count:
                raise AbilityDataError(f"{route} lacks {scan_count} generation candidates")
            for rank, row in enumerate(selected):
                job_id = sha256_bytes(
                    canonical_json_bytes([config["identity"]["name"], route, row["record_id"]])
                )
                jobs.append(
                    {
                        "job_id": job_id,
                        "job_rank": rank,
                        "route": route,
                        "src_lang": source,
                        "tgt_lang": target,
                        "source_record_id": row["record_id"],
                        "semantic_group_id": row["semantic_group_id"],
                        "source_text": row["text"],
                    }
                )
    return jobs


def _prompt_for_route(
    route: str, cross_config: Mapping[str, Any], conversion_config: Mapping[str, Any]
) -> Mapping[str, Any]:
    return conversion_config if route in _CONVERSION_ROUTES else cross_config


def generate_one(
    teacher: LlamaCppTeacher,
    job: Mapping[str, Any],
    *,
    cross_config: Mapping[str, Any],
    conversion_config: Mapping[str, Any],
    generation_identity: str,
) -> dict[str, Any]:
    route = str(job["route"])
    prompt_config = _prompt_for_route(route, cross_config, conversion_config)
    limit = route_limits(prompt_config)[route]
    profile_name = "greedy-v1"
    response = teacher.generate(
        prompt=build_prompt(prompt_config, str(job["source_text"]), str(job["tgt_lang"])),
        profile=prompt_config["decode_profiles"][profile_name],
        sample_id=str(job["job_id"]),
        max_tokens=int(limit["max_output_tokens"]),
        stop=limit["stop"],
    )
    filtered = filter_output(
        source_text=str(job["source_text"]),
        target_text=str(response["raw_output"]),
        target_language=str(job["tgt_lang"]),
        finish_reason=str(response["finish_reason"]),
        config=prompt_config,
    )
    return {
        **job,
        "generation_identity": generation_identity,
        "profile": profile_name,
        **response,
        **filtered,
    }


def _gpu_sampler(path: Path, stop_event: threading.Event) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        while not stop_event.is_set():
            process = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,utilization.gpu,power.draw,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            parts = [part.strip() for part in process.stdout.strip().split(",")]
            if len(parts) == 5:
                row = {
                    "timestamp": parts[0],
                    "gpu_utilization_percent": float(parts[1]),
                    "power_watts": float(parts[2]),
                    "memory_used_mib": float(parts[3]),
                    "memory_total_mib": float(parts[4]),
                }
                handle.write(canonical_json_bytes(row).decode("utf-8"))
                handle.flush()
            stop_event.wait(1.0)


def _load_completed(path: Path, generation_identity: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if row.get("generation_identity") != generation_identity:
            raise AbilityDataError("raw generation journal identity drift")
        job_id = str(row["job_id"])
        if job_id in completed:
            raise AbilityDataError(f"duplicate completed teacher job: {job_id}")
        completed[job_id] = row
    return completed


def generate(
    repository_root: Path,
    runtime_root: Path,
    config_path: Path,
    *,
    max_jobs: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    config_hash = canonical_sha256(config)
    td03_path = repository_root / PurePosixPath(config["inputs"]["td03_manifest_path"])
    if sha256_file(td03_path) != config["inputs"]["td03_manifest_sha256"]:
        raise AbilityDataError("TD-03 compact manifest hash drift")
    td03 = json.loads(td03_path.read_text(encoding="utf-8"))
    source_path = runtime_root / PurePosixPath(config["inputs"]["runtime_source_bank"])
    if sha256_file(source_path) != td03["source_bank"]["sha256"]:
        raise AbilityDataError("TD-03 runtime source bank hash drift")
    source_rows = _read_jsonl(source_path)
    jobs = build_jobs(source_rows, config)
    generation_identity = sha256_bytes(
        canonical_json_bytes([config_hash, td03["source_bank"]["sha256"], len(jobs)])
    )
    output_root = runtime_root / "td04"
    journal = output_root / "raw-generation.jsonl"
    completed = _load_completed(journal, generation_identity)
    pending = [job for job in jobs if job["job_id"] not in completed]
    if max_jobs is not None:
        pending = pending[:max_jobs]

    cross_path = repository_root / PurePosixPath(config["teacher"]["cross_language_prompt_path"])
    conversion_path = repository_root / PurePosixPath(config["teacher"]["chinese_conversion_prompt_path"])
    if sha256_file(cross_path) != config["teacher"]["cross_language_prompt_sha256"]:
        raise AbilityDataError("cross-language prompt hash drift")
    if sha256_file(conversion_path) != config["teacher"]["chinese_conversion_prompt_sha256"]:
        raise AbilityDataError("Chinese conversion prompt hash drift")
    cross = load_prompt_config(cross_path)
    conversion = load_prompt_config(conversion_path)
    server_config = copy.deepcopy(cross)
    slots = int(config["generation"]["parallel_slots"])
    server_config["runtime"]["maximum_batch_size"] = slots
    server_config["runtime"]["port"] = int(config["generation"]["server_port"])
    flush_records = int(config["generation"]["append_flush_records"])
    output_root.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    gpu_thread = threading.Thread(
        target=_gpu_sampler,
        args=(output_root / "gpu-samples.jsonl", stop_event),
        daemon=True,
    )
    started = time.monotonic()
    route_completed: Counter[str] = Counter(str(row["route"]) for row in completed.values())
    with LlamaCppTeacher(repository_root, server_config) as teacher:
        gpu_thread.start()
        with journal.open("a", encoding="utf-8", newline="\n") as handle:
            with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as executor:
                iterator = iter(pending)
                futures: dict[concurrent.futures.Future[dict[str, Any]], Mapping[str, Any]] = {}
                for _ in range(slots):
                    try:
                        job = next(iterator)
                    except StopIteration:
                        break
                    future = executor.submit(
                        generate_one,
                        teacher,
                        job,
                        cross_config=cross,
                        conversion_config=conversion,
                        generation_identity=generation_identity,
                    )
                    futures[future] = job
                newly_completed = 0
                while futures:
                    done, _ = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        job = futures.pop(future)
                        record = future.result()
                        handle.write(canonical_json_bytes(record).decode("utf-8"))
                        completed[str(record["job_id"])] = record
                        route_completed[str(record["route"])] += 1
                        newly_completed += 1
                        if newly_completed % flush_records == 0:
                            handle.flush()
                            os.fsync(handle.fileno())
                            elapsed = time.monotonic() - started
                            print(
                                json.dumps(
                                    {
                                        "event": "progress",
                                        "new_completed": newly_completed,
                                        "total_completed": len(completed),
                                        "total_jobs": len(jobs),
                                        "jobs_per_second": round(newly_completed / elapsed, 3),
                                        "accepted_so_far": sum(bool(row["accepted"]) for row in completed.values()),
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                        try:
                            next_job = next(iterator)
                        except StopIteration:
                            continue
                        next_future = executor.submit(
                            generate_one,
                            teacher,
                            next_job,
                            cross_config=cross,
                            conversion_config=conversion,
                            generation_identity=generation_identity,
                        )
                        futures[next_future] = next_job
                handle.flush()
                os.fsync(handle.fileno())
        stop_event.set()
        gpu_thread.join(timeout=5)
        server_log_tail = teacher.logs[-80:]
    summary = {
        "status": "generation-complete" if len(completed) == len(jobs) else "partial",
        "generation_identity": generation_identity,
        "total_jobs": len(jobs),
        "completed_jobs": len(completed),
        "pending_jobs": len(jobs) - len(completed),
        "route_completed": dict(sorted(route_completed.items())),
        "wall_seconds_this_run": round(time.monotonic() - started, 3),
        "server_log_tail": server_log_tail,
    }
    write_json(output_root / "generation-state.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _multiset(pattern: re.Pattern[str], text: str) -> Counter[str]:
    return Counter(pattern.findall(text))


def build_reverse_pairs(
    accepted: Sequence[Mapping[str, Any]], original_outgoing_counts: Mapping[str, int]
) -> list[dict[str, Any]]:
    by_forward_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in accepted:
        if row["tgt_lang"] == "zho_Hant" and row["src_lang"] != "zho_Hant":
            by_forward_source[str(row["src_lang"])].append(row)
    reversed_rows: list[dict[str, Any]] = []
    for target in (tag for tag in MODEL_TAGS if tag != "zho_Hant"):
        route = f"zho_Hant->{target}"
        limit = int(original_outgoing_counts.get(route, 0) * 0.50)
        candidates = sorted(
            by_forward_source[target],
            key=lambda row: stable_rank("td04-reverse-pair", route, str(row["job_id"])),
        )
        for row in candidates:
            source_text = str(row["normalized_output"])
            target_text = str(row["source_text"])
            if _multiset(_NUMBER_RE, source_text) != _multiset(_NUMBER_RE, target_text):
                continue
            if _multiset(_PLACEHOLDER_RE, source_text) != _multiset(_PLACEHOLDER_RE, target_text):
                continue
            reverse_id = sha256_bytes(canonical_json_bytes([row["job_id"], route, "reverse-pair-v1"]))
            reversed_rows.append(
                {
                    "record_id": f"reverse-{reverse_id[:24]}",
                    "semantic_group_id": row["semantic_group_id"],
                    "forward_job_id": row["job_id"],
                    "src_lang": "zho_Hant",
                    "tgt_lang": target,
                    "source_text": source_text,
                    "target_text": target_text,
                    "provenance": "one_hop_accepted_teacher_pair_reversal",
                    "counts_as_native_hant": False,
                    "generation_identity": row.get("generation_identity"),
                    "profile": row.get("profile"),
                }
            )
            if sum(item["tgt_lang"] == target for item in reversed_rows) == limit:
                break
    return reversed_rows


def finalize(repository_root: Path, runtime_root: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    output_root = runtime_root / "td04"
    raw = _read_jsonl(output_root / "raw-generation.jsonl")
    jobs = build_jobs(_read_jsonl(runtime_root / config["inputs"]["runtime_source_bank"]), config)
    expected_ids = {str(job["job_id"]) for job in jobs}
    actual_ids = {str(row["job_id"]) for row in raw}
    if actual_ids != expected_ids or len(raw) != len(jobs):
        raise AbilityDataError(
            f"teacher journal incomplete: {len(actual_ids)}/{len(expected_ids)} jobs"
        )
    generation_identities = {str(row.get("generation_identity")) for row in raw}
    if len(generation_identities) != 1 or "None" in generation_identities:
        raise AbilityDataError("teacher journal generation identity drift")
    generation_identity = next(iter(generation_identities))
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        by_route[str(row["route"])].append(row)
    fixed_target = int(config["generation"]["accepted_target_per_fixed_route"])
    accepted_raw: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    route_summary: dict[str, Any] = {}
    original_outgoing_counts: dict[str, int] = {}
    for route in sorted(by_route):
        values = sorted(by_route[route], key=lambda row: int(row["job_rank"]))
        passing = [row for row in values if row["accepted"]]
        source = route.split("->", 1)[0]
        if source != "zho_Hant" and len(passing) < fixed_target:
            raise AbilityDataError(
                f"{route} has {len(passing)} accepted records, below fixed target {fixed_target}"
            )
        selected = passing if source == "zho_Hant" else passing[:fixed_target]
        selected_ids = {row["job_id"] for row in selected}
        accepted_raw.extend(selected)
        filtered.extend(row for row in values if row["job_id"] not in selected_ids)
        if source == "zho_Hant":
            original_outgoing_counts[route] = len(selected)
        reasons = Counter(
            reason for row in values for reason in row.get("rejection_reasons", [])
        )
        route_summary[route] = {
            "generated": len(values),
            "passed_filter": len(passing),
            "accepted": len(selected),
            "acceptance_rate": round(len(passing) / len(values), 6),
            "rejection_reasons": dict(sorted(reasons.items())),
            "completion_tokens": sum(int(row["completion_tokens"]) for row in values),
            "latency_seconds_sum": round(sum(float(row["latency_seconds"]) for row in values), 3),
        }

    accepted_rows = [
        {
            "record_id": f"teacher-{str(row['job_id'])[:24]}",
            "semantic_group_id": row["semantic_group_id"],
            "teacher_job_id": row["job_id"],
            "src_lang": row["src_lang"],
            "tgt_lang": row["tgt_lang"],
            "source_text": row["source_text"],
            "target_text": row["normalized_output"],
            "provenance": "Hy-MT2-7B-GGUF-Q8_0-greedy-sequence-distillation",
            "generation_identity": row["generation_identity"],
            "profile": row["profile"],
        }
        for row in accepted_raw
    ]
    reverse = build_reverse_pairs(accepted_raw, original_outgoing_counts)
    accepted_count, accepted_hash = write_jsonl(output_root / "accepted-teacher.jsonl", accepted_rows)
    filtered_count, filtered_hash = write_jsonl(output_root / "filtered-teacher.jsonl", filtered)
    reverse_count, reverse_hash = write_jsonl(output_root / "reverse-pairs.jsonl", reverse)
    gpu_rows = _read_jsonl(output_root / "gpu-samples.jsonl")
    gpu = {
        "samples": len(gpu_rows),
        "utilization_average_percent": round(sum(row["gpu_utilization_percent"] for row in gpu_rows) / len(gpu_rows), 3),
        "utilization_peak_percent": max(row["gpu_utilization_percent"] for row in gpu_rows),
        "power_average_watts": round(sum(row["power_watts"] for row in gpu_rows) / len(gpu_rows), 3),
        "power_peak_watts": max(row["power_watts"] for row in gpu_rows),
        "memory_peak_mib": max(row["memory_used_mib"] for row in gpu_rows),
    }
    reverse_routes = Counter(f"{row['src_lang']}->{row['tgt_lang']}" for row in reverse)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "task": "TD-04",
        "generation_config_sha256": canonical_sha256(config),
        "generation_identity": generation_identity,
        "formal_test_accessed": False,
        "raw": {"records": len(raw), "sha256": sha256_file(output_root / "raw-generation.jsonl")},
        "accepted_teacher": {"records": accepted_count, "sha256": accepted_hash},
        "filtered_teacher": {"records": filtered_count, "sha256": filtered_hash},
        "reverse_pairs": {
            "records": reverse_count,
            "sha256": reverse_hash,
            "route_counts": dict(sorted(reverse_routes.items())),
            "counts_as_native_hant": False,
        },
        "routes": route_summary,
        "gpu": gpu,
        "invariants": {
            "fixed_non_hant_routes_at_10000": sum(
                summary["accepted"] == fixed_target
                for route, summary in route_summary.items()
                if not route.startswith("zho_Hant->")
            ),
            "outgoing_hant_quality_actual_no_refill": True,
            "finite_text_outputs": True,
            "second_teacher_call_for_reverse": False,
        },
    }
    write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "finalize"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mvp_60m_teacher_generation.yaml")
    )
    parser.add_argument("--max-jobs", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    config_path = (repository_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    if args.action == "generate":
        generate(repository_root, args.runtime_root.resolve(), config_path, max_jobs=args.max_jobs)
    else:
        if args.max_jobs is not None:
            raise AbilityDataError("--max-jobs is only valid for generate")
        finalize(repository_root, args.runtime_root.resolve(), config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
