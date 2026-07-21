"""Bounded decode-only A/B for the frozen Hy-MT2 teacher corpus."""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml

from artifact_io import canonical_json_bytes, sha256_bytes, sha256_file, write_json, write_jsonl
from deepseek_translation_review import (
    DeepSeekClient,
    _actual_cost,
    _aggregate,
    _decision_rows,
    estimate_text_tokens,
    load_api_key,
    load_config as load_review_config,
    load_full_review_items,
    make_batches,
    run_batches,
    stratified_review_order,
)
from generate_mvp_60m_teacher import _gpu_sampler, generate_one, load_config as load_generation_config
from hymt2_distillation import LlamaCppTeacher, filter_output, load_prompt_config, route_limits
from model_data_source_contract import canonical_sha256
from mvp_60m_data_pipeline import AbilityDataError


def load_decode_ab_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise AbilityDataError("Hy-MT2 decode A/B config schema differs")
    if value.get("identity", {}).get("status") != "frozen":
        raise AbilityDataError("Hy-MT2 decode A/B config is not frozen")
    inputs = value.get("inputs", {})
    if inputs.get("formal_test_access") != "prohibited" or inputs.get(
        "formal_devtest_access"
    ) != "prohibited":
        raise AbilityDataError("formal evaluation access must remain prohibited")
    ab = value.get("ab", {})
    if int(ab.get("records", 0)) <= 0:
        raise AbilityDataError("decode A/B record budget must be positive")
    if ab.get("baseline_profile") == ab.get("challenger_profile"):
        raise AbilityDataError("decode A/B profiles must differ")
    forced = ab.get("forced_cases", [])
    if not isinstance(forced, list) or not forced:
        raise AbilityDataError("decode A/B must retain known blocker cases")
    return value


def select_decode_ab_items(
    runtime_root: Path,
    review_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items, evidence = load_full_review_items(runtime_root, review_config)
    ordered = [
        dict(item)
        for item in stratified_review_order(items, review_config)
        if item["kind"] == "teacher"
    ]
    budget = int(config["ab"]["records"])
    if len(ordered) < budget:
        raise AbilityDataError("accepted teacher data is smaller than decode A/B budget")
    selected = ordered[:budget]
    forced_keys = {
        (str(case["route"]), str(case["source_record_id"]))
        for case in config["ab"]["forced_cases"]
    }
    by_key = {
        (str(item["route"]), str(item["source_record_id"])): item for item in ordered
    }
    missing_source = forced_keys - set(by_key)
    if missing_source:
        raise AbilityDataError(f"forced decode A/B cases missing: {sorted(missing_source)}")
    selected_keys = {
        (str(item["route"]), str(item["source_record_id"])) for item in selected
    }
    replacements = [by_key[key] for key in sorted(forced_keys - selected_keys)]
    replace_positions = [
        index
        for index in range(len(selected) - 1, -1, -1)
        if (
            str(selected[index]["route"]),
            str(selected[index]["source_record_id"]),
        )
        not in forced_keys
    ]
    if len(replace_positions) < len(replacements):
        raise AbilityDataError("decode A/B budget cannot accommodate forced cases")
    for position, replacement in zip(replace_positions, replacements, strict=False):
        selected[position] = dict(replacement)
    selected.sort(
        key=lambda item: hashlib.sha256(
            f"{config['ab']['selection_seed']}:{item['id']}".encode("utf-8")
        ).digest()
    )
    if len(selected) != budget or len({str(item["id"]) for item in selected}) != budget:
        raise AbilityDataError("decode A/B selection lost or duplicated records")
    actual_forced = {
        (str(item["route"]), str(item["source_record_id"])) for item in selected
    }
    if not forced_keys <= actual_forced:
        raise AbilityDataError("decode A/B selection lost a forced blocker case")
    ordered_hash = sha256_bytes("".join(f"{item['id']}\n" for item in selected).encode())
    return selected, {
        **evidence,
        "teacher_records_available": len(ordered),
        "selected_records": len(selected),
        "forced_cases": len(forced_keys),
        "ordered_ids_sha256": ordered_hash,
    }


def _load_inputs(
    repository_root: Path, runtime_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    generation_path = repository_root / PurePosixPath(config["inputs"]["generation_config"])
    review_path = repository_root / PurePosixPath(config["inputs"]["review_config"])
    if sha256_file(generation_path) != config["inputs"]["generation_config_sha256"]:
        raise AbilityDataError("generation config identity drift")
    if sha256_file(review_path) != config["inputs"]["review_config_sha256"]:
        raise AbilityDataError("review config identity drift")
    generation = load_generation_config(generation_path)
    review = load_review_config(review_path)
    manifest_path = runtime_root / PurePosixPath(review["inputs"]["td04_manifest"])
    if sha256_file(manifest_path) != config["inputs"]["td04_manifest_sha256"]:
        raise AbilityDataError("TD-04 manifest identity drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("accepted_teacher", {}).get("sha256") != config["inputs"][
        "accepted_teacher_sha256"
    ]:
        raise AbilityDataError("accepted teacher identity drift")
    selected, evidence = select_decode_ab_items(runtime_root, review, config)
    return generation, review, manifest, selected, evidence


def _load_prompts(
    repository_root: Path, generation: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    cross_path = repository_root / PurePosixPath(generation["teacher"]["cross_language_prompt_path"])
    conversion_path = repository_root / PurePosixPath(
        generation["teacher"]["chinese_conversion_prompt_path"]
    )
    if sha256_file(cross_path) != generation["teacher"]["cross_language_prompt_sha256"]:
        raise AbilityDataError("cross-language prompt identity drift")
    if sha256_file(conversion_path) != generation["teacher"]["chinese_conversion_prompt_sha256"]:
        raise AbilityDataError("Chinese conversion prompt identity drift")
    cross = load_prompt_config(cross_path)
    conversion = load_prompt_config(conversion_path)
    expected_routes = set(route_limits(cross)) | set(route_limits(conversion))
    limits = {
        str(route): int(value)
        for route, value in generation["generation"]["route_max_output_tokens"].items()
    }
    if set(limits) != expected_routes:
        raise AbilityDataError("decode A/B route limits differ from generation contract")
    return cross, conversion, limits


def _jobs(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "job_id": str(item["id"]),
            "job_rank": index,
            "route": str(item["route"]),
            "src_lang": str(item["source_language"]),
            "tgt_lang": str(item["target_language"]),
            "source_record_id": str(item["source_record_id"]),
            "semantic_group_id": str(item["semantic_group_id"]),
            "source_text": str(item["source_text"]),
        }
        for index, item in enumerate(items)
    ]


def _load_journal(path: Path, identity: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("generation_identity") != identity:
                raise AbilityDataError(f"decode A/B journal identity drift at line {line_number}")
            job_id = str(value["job_id"])
            if job_id in rows:
                raise AbilityDataError(f"duplicate decode A/B job: {job_id}")
            rows[job_id] = value
    return rows


def _generate_challenger(
    repository_root: Path,
    output_root: Path,
    generation: Mapping[str, Any],
    config: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    identity: str,
) -> list[dict[str, Any]]:
    cross, conversion, limits = _load_prompts(repository_root, generation)
    jobs = _jobs(items)
    journal = output_root / config["outputs"]["challenger_journal"]
    completed = _load_journal(journal, identity)
    pending = [job for job in jobs if str(job["job_id"]) not in completed]
    server_config = copy.deepcopy(cross)
    slots = int(config["runtime"]["parallel_slots"])
    server_config["runtime"]["maximum_batch_size"] = slots
    server_config["runtime"]["context_size"] = int(config["runtime"]["server_context_size"])
    server_config["runtime"]["port"] = int(config["runtime"]["server_port"])
    profile = str(config["ab"]["challenger_profile"])
    output_root.mkdir(parents=True, exist_ok=True)
    if pending:
        stop_event = threading.Event()
        gpu_thread = threading.Thread(
            target=_gpu_sampler,
            args=(output_root / config["outputs"]["gpu_samples"], stop_event),
            daemon=True,
        )
        started = time.monotonic()
        try:
            with LlamaCppTeacher(repository_root, server_config) as teacher:
                gpu_thread.start()
                with journal.open("a", encoding="utf-8", newline="\n") as handle:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as executor:
                        futures = {
                            executor.submit(
                                generate_one,
                                teacher,
                                job,
                                cross_config=cross,
                                conversion_config=conversion,
                                generation_identity=identity,
                                max_output_tokens=limits[str(job["route"])],
                                profile_name=profile,
                            ): job
                            for job in pending
                        }
                        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                            record = future.result()
                            handle.write(canonical_json_bytes(record).decode("utf-8"))
                            completed[str(record["job_id"])] = record
                            if index % int(config["runtime"]["append_flush_records"]) == 0:
                                handle.flush()
                                os.fsync(handle.fileno())
                                print(
                                    json.dumps(
                                        {
                                            "event": "decode-ab-progress",
                                            "completed": len(completed),
                                            "records": len(jobs),
                                            "records_per_second": round(index / (time.monotonic() - started), 3),
                                        },
                                        ensure_ascii=False,
                                    ),
                                    flush=True,
                                )
                    handle.flush()
                    os.fsync(handle.fileno())
        finally:
            stop_event.set()
            if gpu_thread.is_alive():
                gpu_thread.join(timeout=5)
    if set(completed) != {str(job["job_id"]) for job in jobs}:
        raise AbilityDataError("decode A/B challenger generation is incomplete")
    return [completed[str(job["job_id"])] for job in jobs]


def _pair_rows(
    items: Sequence[Mapping[str, Any]],
    challenger: Sequence[Mapping[str, Any]],
    cross: Mapping[str, Any],
    conversion: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item, generated in zip(items, challenger, strict=True):
        filter_config = (
            conversion
            if str(item["route"]) in {"zho_Hans->zho_Hant", "zho_Hant->zho_Hans"}
            else cross
        )
        baseline_filter = filter_output(
            source_text=str(item["source_text"]),
            target_text=str(item["candidate_translation"]),
            target_language=str(item["target_language"]),
            finish_reason="stop",
            config=filter_config,
        )
        rows.append(
            {
                "id": str(item["id"]),
                "route": str(item["route"]),
                "source_id": str(item["source_id"]),
                "source_record_id": str(item["source_record_id"]),
                "semantic_group_id": str(item["semantic_group_id"]),
                "source_language": str(item["source_language"]),
                "target_language": str(item["target_language"]),
                "source_text": str(item["source_text"]),
                "baseline_translation": str(item["candidate_translation"]),
                "challenger_translation": str(generated["normalized_output"]),
                "exact_match": str(item["candidate_translation"]) == str(generated["normalized_output"]),
                "baseline_filter_accepted": bool(baseline_filter["accepted"]),
                "challenger_filter_accepted": bool(generated["accepted"]),
                "challenger_rejection_reasons": list(generated["rejection_reasons"]),
                "challenger_finish_reason": str(generated["finish_reason"]),
                "challenger_completion_tokens": int(generated["completion_tokens"]),
                "challenger_latency_seconds": float(generated["latency_seconds"]),
                "challenger_seed": int(generated["seed"]),
            }
        )
    return rows


def _review(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    response_root: Path,
    review_config: Mapping[str, Any],
    api_key: str,
    concurrency: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items = [
        {
            "id": str(row["id"]),
            "kind": "teacher",
            "route": str(row["route"]),
            "source_language": str(row["source_language"]),
            "target_language": str(row["target_language"]),
            "source_text": str(row["source_text"]),
            "candidate_translation": str(row[field]),
            "semantic_group_id": str(row["semantic_group_id"]),
            "source_id": str(row["source_id"]),
            "source_record_id": str(row["source_record_id"]),
        }
        for row in rows
    ]
    batches = make_batches(items, review_config)
    responses = run_batches(
        batches,
        response_root=response_root,
        client=DeepSeekClient(review_config, api_key),
        config=review_config,
        concurrency=concurrency,
    )
    return _decision_rows(batches, responses), responses


def _estimate_review_cost(records: int, review_config: Mapping[str, Any]) -> float:
    prices = review_config["pricing"]["per_million_tokens"]
    input_tokens = records * 2 * 400
    output_tokens = records * 2 * int(review_config["pricing"]["estimated_output_tokens_per_record"])
    return round(
        input_tokens / 1_000_000 * float(prices["input_cache_miss"])
        + output_tokens / 1_000_000 * float(prices["output"]),
        6,
    )


def _comparison(
    rows: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    challenger: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    baseline_by_id = {str(row["review_id"]): row for row in baseline}
    challenger_by_id = {str(row["review_id"]): row for row in challenger}
    forced = {
        (str(case["route"]), str(case["source_record_id"]))
        for case in config["ab"]["forced_cases"]
    }
    classes = Counter()
    evidence: list[dict[str, Any]] = []
    both_pass: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()
    for row in rows:
        item_id = str(row["id"])
        baseline_verdict = str(baseline_by_id[item_id]["verdict"])
        raw_challenger_verdict = str(challenger_by_id[item_id]["verdict"])
        # An identical candidate cannot genuinely change quality. Reuse the
        # baseline decision so stochastic reviewer variance cannot become a
        # fake decode-profile win or regression.
        challenger_verdict = (
            baseline_verdict if bool(row["exact_match"]) else raw_challenger_verdict
        )
        if baseline_verdict == "pass" and challenger_verdict == "pass":
            classification = "both_pass"
            both_pass.append(row)
        elif baseline_verdict != "pass" and challenger_verdict == "pass":
            classification = "sampling_improves"
            selected_ids.add(item_id)
        elif baseline_verdict == "pass" and challenger_verdict != "pass":
            classification = "sampling_regresses"
            selected_ids.add(item_id)
        else:
            classification = "both_flagged"
            selected_ids.add(item_id)
        if (str(row["route"]), str(row["source_record_id"])) in forced:
            selected_ids.add(item_id)
        classes[classification] += 1
        evidence.append(
            {
                "id": item_id,
                "classification": classification,
                "baseline_verdict": baseline_verdict,
                "challenger_verdict": challenger_verdict,
                "raw_challenger_verdict": raw_challenger_verdict,
                "exact_match": bool(row["exact_match"]),
            }
        )
    both_pass.sort(
        key=lambda row: hashlib.sha256(
            f"{config['ab']['blind_seed']}:{row['id']}".encode("utf-8")
        ).digest()
    )
    selected_ids.update(
        str(row["id"])
        for row in both_pass[: int(config["ab"]["both_pass_manual_sample"])]
    )
    queue: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    evidence_by_id = {str(row["id"]): row for row in evidence}
    for row in rows:
        item_id = str(row["id"])
        if item_id not in selected_ids:
            continue
        baseline_is_a = hashlib.sha256(
            f"{config['ab']['blind_seed']}:side:{item_id}".encode("utf-8")
        ).digest()[0] % 2 == 0
        queue.append(
            {
                "ab_id": item_id,
                "route": str(row["route"]),
                "source_id": str(row["source_id"]),
                "source_record_id": str(row["source_record_id"]),
                "source_text": str(row["source_text"]),
                "candidate_a": str(
                    row["baseline_translation"] if baseline_is_a else row["challenger_translation"]
                ),
                "candidate_b": str(
                    row["challenger_translation"] if baseline_is_a else row["baseline_translation"]
                ),
            }
        )
        key.append(
            {
                **evidence_by_id[item_id],
                "candidate_a": "baseline_greedy" if baseline_is_a else "official_sampling",
                "candidate_b": "official_sampling" if baseline_is_a else "baseline_greedy",
            }
        )
    return queue, key, {
        "review_classification_counts": dict(sorted(classes.items())),
        "exact_translation_matches": sum(bool(row["exact_match"]) for row in rows),
        "baseline_filter_accepted": sum(bool(row["baseline_filter_accepted"]) for row in rows),
        "challenger_filter_accepted": sum(bool(row["challenger_filter_accepted"]) for row in rows),
        "blind_queue_records": len(queue),
        "reviewer_variance_on_exact_matches": sum(
            bool(row["exact_match"])
            and str(baseline_by_id[str(row["id"])]["verdict"])
            != str(challenger_by_id[str(row["id"])]["verdict"])
            for row in rows
        ),
    }


def run_decode_ab(
    *,
    repository_root: Path,
    runtime_root: Path,
    config_path: Path,
    api_key_env: str,
    auth_script: Path | None,
    cost_ceiling: float,
    review_concurrency: int | None,
) -> dict[str, Any]:
    config = load_decode_ab_config(config_path)
    generation, review_config, manifest, items, selection = _load_inputs(
        repository_root, runtime_root, config
    )
    output_root = runtime_root / PurePosixPath(config["outputs"]["root"])
    identity = sha256_bytes(
        canonical_json_bytes(
            [canonical_sha256(config), manifest["generation_identity"], selection["ordered_ids_sha256"]]
        )
    )
    estimate = _estimate_review_cost(len(items), review_config)
    if estimate > cost_ceiling:
        raise AbilityDataError(
            f"decode A/B review estimate ${estimate:.6f} exceeds ceiling ${cost_ceiling:.6f}"
        )
    plan = {
        "schema_version": 1,
        "status": "frozen",
        "task": "Hy-MT2-7B-decode-only-ab",
        "config_sha256": sha256_file(config_path),
        "identity": identity,
        "baseline_profile": config["ab"]["baseline_profile"],
        "challenger_profile": config["ab"]["challenger_profile"],
        "selection": selection,
        "estimated_review_cost_usd": estimate,
        "formal_test_accessed": False,
        "formal_devtest_accessed": False,
    }
    write_json(output_root / config["outputs"]["plan"], plan)
    generated = _generate_challenger(
        repository_root, output_root, generation, config, items, identity
    )
    cross, conversion, _ = _load_prompts(repository_root, generation)
    rows = _pair_rows(items, generated, cross, conversion)
    pair_count, pair_sha = write_jsonl(output_root / config["outputs"]["pairs"], rows)
    api_key = load_api_key(env_name=api_key_env, auth_script=auth_script)
    concurrency = review_concurrency or int(review_config["api"]["concurrency"])
    baseline_decisions, baseline_responses = _review(
        rows,
        field="baseline_translation",
        response_root=output_root / config["outputs"]["baseline_review_responses"],
        review_config=review_config,
        api_key=api_key,
        concurrency=concurrency,
    )
    challenger_decisions, challenger_responses = _review(
        rows,
        field="challenger_translation",
        response_root=output_root / config["outputs"]["challenger_review_responses"],
        review_config=review_config,
        api_key=api_key,
        concurrency=concurrency,
    )
    baseline_count, baseline_sha = write_jsonl(
        output_root / config["outputs"]["baseline_decisions"], baseline_decisions
    )
    challenger_count, challenger_sha = write_jsonl(
        output_root / config["outputs"]["challenger_decisions"], challenger_decisions
    )
    baseline_by_id = {str(row["review_id"]): row for row in baseline_decisions}
    challenger_by_id = {str(row["review_id"]): row for row in challenger_decisions}
    adjusted_challenger_decisions = [
        (
            {**baseline_by_id[str(row["id"])], "decision_source": "baseline_exact_match"}
            if bool(row["exact_match"])
            else {
                **challenger_by_id[str(row["id"])],
                "decision_source": "challenger_changed_output",
            }
        )
        for row in rows
    ]
    queue, key, comparison = _comparison(
        rows, baseline_decisions, challenger_decisions, config
    )
    queue_count, queue_sha = write_jsonl(output_root / config["outputs"]["blind_queue"], queue)
    key_count, key_sha = write_jsonl(output_root / config["outputs"]["blind_key"], key)
    gpu_path = output_root / config["outputs"]["gpu_samples"]
    gpu_samples = []
    if gpu_path.exists():
        with gpu_path.open(encoding="utf-8") as handle:
            gpu_samples = [json.loads(line) for line in handle if line.strip()]
    report = {
        "schema_version": 1,
        "status": "awaiting_manual_review",
        "task": "Hy-MT2-7B-decode-only-ab",
        "identity": identity,
        "records": pair_count,
        "pairs_sha256": pair_sha,
        "baseline_profile": config["ab"]["baseline_profile"],
        "challenger_profile": config["ab"]["challenger_profile"],
        "comparison": comparison,
        "baseline_review": {**_aggregate(baseline_decisions), "records": baseline_count, "sha256": baseline_sha},
        "challenger_review": {**_aggregate(challenger_decisions), "records": challenger_count, "sha256": challenger_sha},
        "challenger_review_exact_match_adjusted": _aggregate(
            adjusted_challenger_decisions
        ),
        "blind_queue": {"records": queue_count, "sha256": queue_sha},
        "blind_key": {"records": key_count, "sha256": key_sha},
        "review_cost": {
            "baseline": _actual_cost(baseline_responses, review_config),
            "challenger": _actual_cost(challenger_responses, review_config),
        },
        "gpu": {
            "samples": len(gpu_samples),
            "utilization_average_percent": round(
                sum(float(row["gpu_utilization_percent"]) for row in gpu_samples) / len(gpu_samples), 3
            ) if gpu_samples else None,
            "utilization_peak_percent": max(
                (float(row["gpu_utilization_percent"]) for row in gpu_samples), default=None
            ),
            "power_peak_watts": max((float(row["power_watts"]) for row in gpu_samples), default=None),
            "memory_peak_mib": max((float(row["memory_used_mib"]) for row in gpu_samples), default=None),
        },
        "formal_test_accessed": False,
        "formal_devtest_accessed": False,
    }
    write_json(output_root / config["outputs"]["report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report
