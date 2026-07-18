"""Bounded direct-translation A/B between Hy-MT2 and DeepSeek V4 Flash."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

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
    make_batches as make_review_batches,
    read_json,
    read_jsonl,
    run_batches as run_review_batches,
    stratified_review_order,
)
from hymt2_distillation import filter_output, load_yaml


PROMPT_VERSION = "deepseek-direct-translation-ab-v1"
SYSTEM_PROMPT = """You are the translation teacher for a compact multilingual model.
Translate every supplied source into the requested target language with maximum
semantic fidelity. Preserve every person, place, organization, era name, title,
product, number, date, unit, qualifier, relation, placeholder, and formatting token.
Use a conventional target-language name when one is established; otherwise use a
faithful transliteration. Never replace a proper name or technical/legal term with
an invented name or an unsupported descriptive gloss. Do not omit, summarize,
explain, annotate, or add information.

Return one JSON object with an `items` array in exactly the input order. Each item
must contain exactly `id` and `translation`. Output only the translated text in
`translation`, without quotation commentary or a language label. JSON only.

Example: {"items":[{"id":"x","translation":"译文"}]}
"""

LANGUAGE_NAMES = {
    "eng_Latn": "English",
    "zho_Hans": "Simplified Chinese",
    "zho_Hant": "Traditional Chinese",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
}
_NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d[\d,.:/-]*\d|\d)(?!\w)")


class TranslationABError(RuntimeError):
    """Raised when the bounded A/B identity or response contract changes."""


def load_ab_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise TranslationABError("direct-translation A/B config schema differs")
    if value.get("identity", {}).get("status") != "frozen":
        raise TranslationABError("direct-translation A/B config is not frozen")
    inputs = value.get("inputs", {})
    if inputs.get("formal_test_access") != "prohibited" or inputs.get(
        "formal_devtest_access"
    ) != "prohibited":
        raise TranslationABError("formal evaluation access must remain prohibited")
    api = value.get("api", {})
    if api.get("model") != "deepseek-v4-flash" or api.get("thinking") != "disabled":
        raise TranslationABError("A/B translation requires non-thinking DeepSeek V4 Flash")
    if api.get("response_format") != "json_object":
        raise TranslationABError("A/B translation requires JSON output")
    if int(value.get("ab", {}).get("records", 0)) <= 0:
        raise TranslationABError("A/B record budget must be positive")
    return value


def api_item(item: Mapping[str, Any]) -> dict[str, str]:
    target = str(item["target_language"])
    return {
        "id": str(item["id"]),
        "source_language": str(item["source_language"]),
        "target_language": target,
        "target_language_name": LANGUAGE_NAMES[target],
        "source": str(item["source_text"]),
    }


def user_payload(items: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(
        {
            "task": "direct_translation",
            "required_json_shape": {
                "items": [{"id": "same input ID", "translation": "target text"}]
            },
            "items": [api_item(item) for item in items],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def make_translation_batches(
    items: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    maximum_records = int(config["api"]["batch_max_records"])
    maximum_tokens = int(config["api"]["batch_max_estimated_input_tokens"])
    prompt_tokens = estimate_text_tokens(SYSTEM_PROMPT)
    batches: list[dict[str, Any]] = []
    current: list[Mapping[str, Any]] = []
    current_tokens = prompt_tokens
    for item in items:
        item_tokens = estimate_text_tokens(json.dumps(api_item(item), ensure_ascii=False)) + 8
        if current and (
            len(current) >= maximum_records
            or current_tokens + item_tokens > maximum_tokens
        ):
            batches.append({"items": list(current), "estimated_input_tokens": current_tokens})
            current = []
            current_tokens = prompt_tokens
        if item_tokens + prompt_tokens > maximum_tokens:
            raise TranslationABError(f"single translation item exceeds budget: {item['id']}")
        current.append(item)
        current_tokens += item_tokens
    if current:
        batches.append({"items": list(current), "estimated_input_tokens": current_tokens})
    for index, batch in enumerate(batches):
        batch["index"] = index
        batch["identity"] = sha256_bytes(
            canonical_json_bytes(
                [
                    PROMPT_VERSION,
                    config["api"]["model"],
                    config["api"]["thinking"],
                    [api_item(item) for item in batch["items"]],
                ],
                allow_nan=False,
            )
        )
    return batches


def estimate_cost(
    batches: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    records = sum(len(batch["items"]) for batch in batches)
    input_tokens = sum(int(batch["estimated_input_tokens"]) for batch in batches)
    output_tokens = records * int(config["pricing"]["estimated_output_tokens_per_record"])
    prices = config["pricing"]["per_million_tokens"]
    input_cost = input_tokens / 1_000_000 * float(prices["input_cache_miss"])
    output_cost = output_tokens / 1_000_000 * float(prices["output"])
    return {
        "records": records,
        "batches": len(batches),
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_total_usd": round(input_cost + output_cost, 6),
    }


def validate_translations(
    raw: Any, items: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        raise TranslationABError("translation response must contain an items array")
    values = raw["items"]
    expected_ids = [str(item["id"]) for item in items]
    actual_ids = [str(item.get("id", "")) for item in values if isinstance(item, dict)]
    if len(values) != len(items) or actual_ids != expected_ids:
        raise TranslationABError("translation response IDs/order differ from request")
    result: list[dict[str, str]] = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {"id", "translation"}:
            raise TranslationABError("translation response item fields differ")
        translation = unicodedata.normalize("NFC", str(value["translation"])).strip()
        if not translation or len(translation) > 10_000:
            raise TranslationABError("translation response contains empty/oversized text")
        result.append({"id": str(value["id"]), "translation": translation})
    return result


def _response_path(root: Path, batch: Mapping[str, Any]) -> Path:
    return root / f"batch-{int(batch['index']):06d}-{str(batch['identity'])[:16]}.json"


def run_translation_batches(
    batches: Sequence[Mapping[str, Any]],
    *,
    response_root: Path,
    client: DeepSeekClient,
    config: Mapping[str, Any],
    concurrency: int,
) -> list[dict[str, Any]]:
    response_root.mkdir(parents=True, exist_ok=True)
    complete: dict[str, dict[str, Any]] = {}
    pending: list[Mapping[str, Any]] = []
    for batch in batches:
        path = _response_path(response_root, batch)
        if path.exists():
            value = read_json(path)
            if value.get("status") != "complete" or value.get("batch_identity") != batch["identity"]:
                raise TranslationABError(f"translation response identity drift: {path}")
            validate_translations({"items": value.get("translations")}, batch["items"])
            complete[str(batch["identity"])] = value
        else:
            pending.append(batch)

    def worker(batch: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        response = client.request_json(
            system_prompt=SYSTEM_PROMPT,
            user_content=user_payload(batch["items"]),
            max_tokens=int(config["api"]["max_output_tokens"]),
        )
        translations = validate_translations(response.pop("json"), batch["items"])
        record = {
            "schema_version": 1,
            "status": "complete",
            "prompt_version": PROMPT_VERSION,
            "batch_index": int(batch["index"]),
            "batch_identity": str(batch["identity"]),
            "records": len(batch["items"]),
            **response,
            "translations": translations,
        }
        write_json(_response_path(response_root, batch), record)
        return str(batch["identity"]), record

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            futures = {executor.submit(worker, batch): batch for batch in pending}
            for future in concurrent.futures.as_completed(futures):
                identity, record = future.result()
                complete[identity] = record
    return [complete[str(batch["identity"])] for batch in batches]


def _translation_rows(
    batches: Sequence[Mapping[str, Any]], responses: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_identity = {str(response["batch_identity"]): response for response in responses}
    rows: list[dict[str, Any]] = []
    for batch in batches:
        response = by_identity[str(batch["identity"])]
        for item, translated in zip(batch["items"], response["translations"], strict=True):
            rows.append(
                {
                    "id": str(item["id"]),
                    "kind": str(item["kind"]),
                    "route": str(item["route"]),
                    "source_id": str(item["source_id"]),
                    "semantic_group_id": str(item["semantic_group_id"]),
                    "source_language": str(item["source_language"]),
                    "target_language": str(item["target_language"]),
                    "source_text": str(item["source_text"]),
                    "hymt_translation": str(item["candidate_translation"]),
                    "deepseek_translation": str(translated["translation"]),
                    "batch_identity": str(batch["identity"]),
                }
            )
    return rows


def _selected_items(
    runtime_root: Path, repo_root: Path, config: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    review_path = repo_root / PurePosixPath(config["inputs"]["review_config"])
    if sha256_file(review_path) != config["inputs"]["review_config_sha256"]:
        raise TranslationABError("review config identity drift")
    review_config = load_review_config(review_path)
    items, evidence = load_full_review_items(runtime_root, review_config)
    ordered = stratified_review_order(items, review_config)
    selected = ordered[: int(config["ab"]["records"])]
    review_root = runtime_root / PurePosixPath(review_config["outputs"]["root"])
    report_path = review_root / review_config["outputs"]["report"]
    assessment_path = runtime_root / PurePosixPath(config["inputs"]["stage_assessment"])
    if sha256_file(report_path) != config["inputs"]["stage_report_sha256"]:
        raise TranslationABError("512-stage review report identity drift")
    if sha256_file(assessment_path) != config["inputs"]["stage_assessment_sha256"]:
        raise TranslationABError("512-stage manual assessment identity drift")
    review_report = read_json(report_path)
    if review_report.get("records_reviewed") != len(selected):
        raise TranslationABError("reviewed stage size differs from A/B selection")
    decisions_path = review_root / review_config["outputs"]["decisions"]
    if sha256_file(decisions_path) != review_report["decisions_sha256"]:
        raise TranslationABError("review decisions identity drift")
    decisions = read_jsonl(decisions_path)
    if [str(row["review_id"]) for row in decisions] != [str(item["id"]) for item in selected]:
        raise TranslationABError("A/B selection differs from reviewed 512-stage order")
    return selected, evidence, decisions, review_config


def _numbers(text: str) -> list[str]:
    return [match.group(0).replace(",", "") for match in _NUMBER_RE.finditer(text)]


def _automated_checks(
    rows: Sequence[Mapping[str, Any]], repo_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    filter_path = repo_root / PurePosixPath(config["ab"]["filter_config"])
    if sha256_file(filter_path) != config["ab"]["filter_config_sha256"]:
        raise TranslationABError("automated filter config identity drift")
    filter_config = load_yaml(filter_path)
    counts = Counter()
    evidence: dict[str, dict[str, Any]] = {}
    for row in rows:
        per_system: dict[str, Any] = {}
        for system, field in (
            ("hymt2", "hymt_translation"),
            ("deepseek", "deepseek_translation"),
        ):
            result = filter_output(
                source_text=str(row["source_text"]),
                target_text=str(row[field]),
                target_language=str(row["target_language"]),
                finish_reason="stop",
                config=filter_config,
            )
            numeric_match = _numbers(str(row["source_text"])) == _numbers(str(row[field]))
            per_system[system] = {
                "accepted": bool(result["accepted"]),
                "rejection_reasons": list(result["rejection_reasons"]),
                "numeric_surface_match": numeric_match,
            }
            counts[f"{system}|filter_accepted={bool(result['accepted'])}"] += 1
            counts[f"{system}|numeric_surface_match={numeric_match}"] += 1
        evidence[str(row["id"])] = per_system
    return {"counts": dict(sorted(counts.items()))}, evidence


def _blind_queue(
    rows: Sequence[Mapping[str, Any]],
    hymt_decisions: Sequence[Mapping[str, Any]],
    direct_decisions: Sequence[Mapping[str, Any]],
    automated: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    hymt_by_id = {str(row["review_id"]): row for row in hymt_decisions}
    direct_by_id = {str(row["review_id"]): row for row in direct_decisions}
    pair_counts = Counter()
    classifications: dict[str, str] = {}
    for row in rows:
        item_id = str(row["id"])
        hymt_flag = hymt_by_id[item_id]["verdict"] != "pass"
        direct_flag = direct_by_id[item_id]["verdict"] != "pass"
        if not hymt_flag and not direct_flag:
            classification = "both_pass"
        elif hymt_flag and not direct_flag:
            classification = "deepseek_improves"
        elif not hymt_flag and direct_flag:
            classification = "deepseek_regresses"
        else:
            classification = "both_flagged"
        classifications[item_id] = classification
        pair_counts[classification] += 1
    both_pass = [row for row in rows if classifications[str(row["id"])] == "both_pass"]
    seed = str(config["ab"]["blind_seed"])
    both_pass.sort(
        key=lambda row: hashlib.sha256(f"{seed}:pass:{row['id']}".encode()).digest()
    )
    selected_ids = {
        str(row["id"])
        for row in rows
        if classifications[str(row["id"])] != "both_pass"
    }
    selected_ids.update(
        str(row["id"])
        for row in both_pass[: int(config["ab"]["both_pass_manual_sample"])]
    )
    queue: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for row in rows:
        item_id = str(row["id"])
        if item_id not in selected_ids:
            continue
        hymt_is_a = hashlib.sha256(f"{seed}:side:{item_id}".encode()).digest()[0] % 2 == 0
        candidate_a = str(row["hymt_translation"] if hymt_is_a else row["deepseek_translation"])
        candidate_b = str(row["deepseek_translation"] if hymt_is_a else row["hymt_translation"])
        queue.append(
            {
                "ab_id": item_id,
                "route": str(row["route"]),
                "source_id": str(row["source_id"]),
                "source_language": str(row["source_language"]),
                "target_language": str(row["target_language"]),
                "source_text": str(row["source_text"]),
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
            }
        )
        key.append(
            {
                "ab_id": item_id,
                "classification": classifications[item_id],
                "candidate_a": "hymt2" if hymt_is_a else "deepseek",
                "candidate_b": "deepseek" if hymt_is_a else "hymt2",
                "hymt_review": hymt_by_id[item_id]["verdict"],
                "direct_review": direct_by_id[item_id]["verdict"],
                "automated": automated[item_id],
            }
        )
    return queue, key, {
        "pair_classification_counts": dict(sorted(pair_counts.items())),
        "manual_queue_records": len(queue),
        "both_pass_sample_records": min(
            len(both_pass), int(config["ab"]["both_pass_manual_sample"])
        ),
    }


def run_ab(
    *,
    runtime_root: Path,
    repo_root: Path,
    config_path: Path,
    auth_script: Path | None,
    api_key_env: str,
    cost_ceiling: float,
    concurrency: int | None,
) -> dict[str, Any]:
    config = load_ab_config(config_path)
    selected, evidence, hymt_decisions, review_config = _selected_items(
        runtime_root, repo_root, config
    )
    batches = make_translation_batches(selected, config)
    estimate = estimate_cost(batches, config)
    review_estimate_records = len(selected) * int(
        review_config["pricing"]["estimated_output_tokens_per_record"]
    )
    review_estimate_usd = review_estimate_records / 1_000_000 * float(
        review_config["pricing"]["per_million_tokens"]["output"]
    )
    if float(estimate["estimated_total_usd"]) + review_estimate_usd > cost_ceiling:
        raise TranslationABError("translation plus independent-review estimate exceeds ceiling")
    output_root = runtime_root / PurePosixPath(config["outputs"]["root"])
    plan = {
        "schema_version": 1,
        "status": "frozen",
        "prompt_version": PROMPT_VERSION,
        "config_sha256": sha256_file(config_path),
        "records": len(selected),
        "selection": {
            **evidence,
            "ordered_ids_sha256": sha256_bytes(
                "".join(f"{item['id']}\n" for item in selected).encode()
            ),
        },
        "translation_estimate": estimate,
        "formal_test_accessed": False,
        "formal_devtest_accessed": False,
    }
    write_json(output_root / config["outputs"]["plan"], plan)
    api_key = load_api_key(env_name=api_key_env, auth_script=auth_script)
    translation_client = DeepSeekClient(config, api_key)
    responses = run_translation_batches(
        batches,
        response_root=output_root / config["outputs"]["translation_responses"],
        client=translation_client,
        config=config,
        concurrency=concurrency or int(config["api"]["concurrency"]),
    )
    rows = _translation_rows(batches, responses)
    translation_records, translation_sha = write_jsonl(
        output_root / config["outputs"]["translations"], rows
    )

    direct_review_items = [
        {
            **dict(item),
            "candidate_translation": str(row["deepseek_translation"]),
        }
        for item, row in zip(selected, rows, strict=True)
    ]
    review_batches = make_review_batches(direct_review_items, review_config)
    review_client = DeepSeekClient(review_config, api_key)
    review_responses = run_review_batches(
        review_batches,
        response_root=output_root / config["outputs"]["review_responses"],
        client=review_client,
        config=review_config,
        concurrency=concurrency or int(review_config["api"]["concurrency"]),
    )
    direct_decisions = _decision_rows(review_batches, review_responses)
    direct_records, direct_sha = write_jsonl(
        output_root / config["outputs"]["direct_review_decisions"], direct_decisions
    )
    automated_summary, automated = _automated_checks(rows, repo_root, config)
    queue, key, comparison = _blind_queue(
        rows, hymt_decisions, direct_decisions, automated, config
    )
    queue_records, queue_sha = write_jsonl(
        output_root / config["outputs"]["blind_queue"], queue
    )
    key_records, key_sha = write_jsonl(
        output_root / config["outputs"]["blind_key"], key
    )
    report = {
        "schema_version": 1,
        "status": "awaiting_manual_review",
        "task": "TD-04-independent-direct-translation-ab",
        "prompt_version": PROMPT_VERSION,
        "config_sha256": sha256_file(config_path),
        "records": len(rows),
        "translations": {"records": translation_records, "sha256": translation_sha},
        "hymt2_review": _aggregate(hymt_decisions),
        "deepseek_direct_review": _aggregate(direct_decisions),
        "comparison": comparison,
        "automated_checks": automated_summary,
        "blind_manual_queue": {"records": queue_records, "sha256": queue_sha},
        "blind_key": {"records": key_records, "sha256": key_sha},
        "cost": {
            "translation": _actual_cost(responses, config),
            "direct_translation_review": _actual_cost(review_responses, review_config),
        },
        "formal_test_accessed": False,
        "formal_devtest_accessed": False,
    }
    write_json(output_root / config["outputs"]["report"], report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deepseek_translation_ab.yaml"),
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--auth-script", type=Path)
    parser.add_argument("--max-cost-usd", type=float, default=0.25)
    parser.add_argument("--concurrency", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_cost_usd <= 0:
        raise TranslationABError("cost ceiling must be positive")
    if args.concurrency is not None and args.concurrency <= 0:
        raise TranslationABError("concurrency must be positive")
    report = run_ab(
        runtime_root=args.runtime_root.resolve(),
        repo_root=Path(__file__).resolve().parents[1],
        config_path=args.config.resolve(),
        auth_script=args.auth_script.resolve() if args.auth_script else None,
        api_key_env=args.api_key_env,
        cost_ceiling=float(args.max_cost_usd),
        concurrency=args.concurrency,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
