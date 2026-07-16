"""TD-13 standalone, route-complete MVP evaluation protocol."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from freeze_tokenizer_artifact import sha256_file
from hymt2_distillation import chinese_script_evidence
from model_data_pipeline import wrong_script_dominates
from model_training_contract import (
    LANGUAGE_TAGS,
    MODEL_TO_PRODUCT,
    TOKENIZER_MANIFEST_SHA256,
    config_sha256,
    directed_routes,
    load_student_config,
)
from mvp_student import (
    DirectionAwareCollator,
    EncodingPolicy,
    load_frozen_tokenizer,
    state_dict_sha256,
    validate_student_alignment,
)
from mvp_training import ROUTE_ORDER, load_route_dataset


EVALUATION_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PRODUCT_DIRECTION_ORDER = tuple(
    f"{source}->{target}"
    for source in ("Chinese", "English", "Japanese", "Korean")
    for target in ("Chinese", "English", "Japanese", "Korean")
    if source != target
)


class EvaluationContractError(RuntimeError):
    """Raised when an evaluation would violate the frozen TD-13 protocol."""


def _expect_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise EvaluationContractError(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        raise EvaluationContractError(f"{context} unknown fields: {', '.join(unknown)}")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationContractError(f"{context} must be a mapping")
    return value


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvaluationContractError(f"{context} must be a positive integer")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise EvaluationContractError(f"{context} must be a lowercase SHA-256")
    return value


def _repo_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EvaluationContractError(f"{context} must be a repository-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise EvaluationContractError(f"{context} escapes the repository boundary")
    return value


def validate_evaluation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _expect_keys(
        config,
        {"schema_version", "identity", "data", "encoding", "generation", "metrics", "runtime"},
        "evaluation config",
    )
    if config["schema_version"] != EVALUATION_SCHEMA_VERSION:
        raise EvaluationContractError("unsupported evaluation schema_version")

    identity = _mapping(config["identity"], "evaluation.identity")
    _expect_keys(
        identity,
        {
            "name", "purpose", "student_config", "student_config_file_sha256",
            "student_config_canonical_sha256", "tokenizer_manifest_sha256",
        },
        "evaluation.identity",
    )
    if identity["name"] != "mvp-evaluation-protocol-v1":
        raise EvaluationContractError("evaluation identity name changed")
    if identity["purpose"] != "standalone_evaluation_only":
        raise EvaluationContractError("evaluation may not be embedded in training")
    _repo_path(identity["student_config"], "evaluation.identity.student_config")
    for field in (
        "student_config_file_sha256", "student_config_canonical_sha256",
        "tokenizer_manifest_sha256",
    ):
        _sha256(identity[field], f"evaluation.identity.{field}")
    if identity["tokenizer_manifest_sha256"] != TOKENIZER_MANIFEST_SHA256:
        raise EvaluationContractError("tokenizer identity changed")

    data = _mapping(config["data"], "evaluation.data")
    _expect_keys(
        data,
        {
            "dev_path", "dev_sha256", "test_path", "test_sha256", "manifest_path",
            "manifest_sha256", "records_per_route", "selection",
        },
        "evaluation.data",
    )
    for field in ("dev_path", "test_path", "manifest_path"):
        _repo_path(data[field], f"evaluation.data.{field}")
    for field in ("dev_sha256", "test_sha256", "manifest_sha256"):
        _sha256(data[field], f"evaluation.data.{field}")
    _positive_int(data["records_per_route"], "evaluation.data.records_per_route")
    if data["selection"] != "first_in_frozen_canonical_order":
        raise EvaluationContractError("evaluation selection rule changed")

    encoding = _mapping(config["encoding"], "evaluation.encoding")
    _expect_keys(
        encoding,
        {"max_source_length", "max_target_length", "overflow_policy", "label_pad_id"},
        "evaluation.encoding",
    )
    _positive_int(encoding["max_source_length"], "evaluation.encoding.max_source_length")
    _positive_int(encoding["max_target_length"], "evaluation.encoding.max_target_length")
    if encoding["overflow_policy"] != "truncate_preserve_language_and_eos":
        raise EvaluationContractError("evaluation overflow policy changed")
    if encoding["label_pad_id"] != -100:
        raise EvaluationContractError("evaluation label pad ID must be -100")

    generation = _mapping(config["generation"], "evaluation.generation")
    _expect_keys(
        generation,
        {"decoding", "do_sample", "num_beams", "max_new_tokens", "length_penalty", "normalization"},
        "evaluation.generation",
    )
    if (
        generation["decoding"] != "greedy"
        or generation["do_sample"] is not False
        or generation["num_beams"] != 1
        or generation["length_penalty"] != 1.0
    ):
        raise EvaluationContractError("only the frozen greedy decode protocol is allowed")
    _positive_int(generation["max_new_tokens"], "evaluation.generation.max_new_tokens")
    if generation["normalization"] != "unicode_nfc_strip_collapse_whitespace":
        raise EvaluationContractError("evaluation normalization changed")

    metrics = _mapping(config["metrics"], "evaluation.metrics")
    _expect_keys(
        metrics,
        {
            "sacrebleu_version", "bleu_tokenize", "bleu_smooth_method",
            "bleu_effective_order", "chrf_char_order", "chrf_word_order", "chrf_beta",
            "chrf_whitespace", "chrf_eps_smoothing",
        },
        "evaluation.metrics",
    )
    expected_metrics = {
        "sacrebleu_version": "2.6.0", "bleu_tokenize": "13a",
        "bleu_smooth_method": "exp", "bleu_effective_order": True,
        "chrf_char_order": 6, "chrf_word_order": 0, "chrf_beta": 2,
        "chrf_whitespace": False, "chrf_eps_smoothing": False,
    }
    if dict(metrics) != expected_metrics:
        raise EvaluationContractError("metric semantics differ from the frozen protocol")

    runtime = _mapping(config["runtime"], "evaluation.runtime")
    _expect_keys(
        runtime,
        {"device", "precision", "batch_size", "fixed_examples_per_route", "test_requires_explicit_authorization"},
        "evaluation.runtime",
    )
    if runtime["device"] not in {"cpu", "cuda"}:
        raise EvaluationContractError("evaluation device must be cpu or cuda")
    if runtime["precision"] not in {"float32", "bf16"}:
        raise EvaluationContractError("evaluation precision must be float32 or bf16")
    _positive_int(runtime["batch_size"], "evaluation.runtime.batch_size")
    _positive_int(runtime["fixed_examples_per_route"], "evaluation.runtime.fixed_examples_per_route")
    if runtime["test_requires_explicit_authorization"] is not True:
        raise EvaluationContractError("test must require explicit authorization")
    return json.loads(json.dumps(config))


def load_evaluation_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvaluationContractError(f"cannot load evaluation config: {exc}") from exc
    if not isinstance(value, Mapping):
        raise EvaluationContractError("evaluation config must be a mapping")
    return validate_evaluation_config(value)


def normalize_prediction(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def target_script_compliant(text: str, target_language: str) -> bool:
    if target_language not in LANGUAGE_TAGS:
        raise EvaluationContractError(f"unsupported target language: {target_language}")
    if not text or wrong_script_dominates(text, target_language):
        return False
    if target_language in {"zho_Hans", "zho_Hant"}:
        evidence = chinese_script_evidence(text)
        if target_language == "zho_Hans":
            return evidence["simplified"] >= evidence["traditional"]
        return evidence["traditional"] >= evidence["simplified"]
    return True


def _metric_objects(config: Mapping[str, Any]) -> tuple[Any, Any]:
    from sacrebleu.metrics import BLEU, CHRF

    expected = str(config["metrics"]["sacrebleu_version"])
    actual = importlib.metadata.version("sacrebleu")
    if actual != expected:
        raise EvaluationContractError(f"sacrebleu version is {actual}, expected {expected}")
    metrics = config["metrics"]
    bleu = BLEU(
        tokenize=metrics["bleu_tokenize"],
        smooth_method=metrics["bleu_smooth_method"],
        effective_order=metrics["bleu_effective_order"],
    )
    chrf = CHRF(
        char_order=metrics["chrf_char_order"],
        word_order=metrics["chrf_word_order"],
        beta=metrics["chrf_beta"],
        whitespace=metrics["chrf_whitespace"],
        eps_smoothing=metrics["chrf_eps_smoothing"],
    )
    return bleu, chrf


def corpus_metrics(
    predictions: Sequence[str], references: Sequence[str], config: Mapping[str, Any]
) -> dict[str, Any]:
    if not predictions or len(predictions) != len(references):
        raise EvaluationContractError("metric inputs must be non-empty and aligned")
    bleu, chrf = _metric_objects(config)
    bleu_score = bleu.corpus_score(list(predictions), [list(references)])
    chrf_score = chrf.corpus_score(list(predictions), [list(references)])
    return {
        "sacrebleu": float(bleu_score.score),
        "sacrebleu_signature": str(bleu.get_signature()),
        "chrf": float(chrf_score.score),
        "chrf_signature": str(chrf.get_signature()),
    }


def _summarize_samples(samples: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    if not samples:
        raise EvaluationContractError("cannot summarize an empty sample set")
    metrics = corpus_metrics(
        [str(row["prediction"]) for row in samples],
        [str(row["reference"]) for row in samples],
        config,
    )
    count = len(samples)
    target_tokens = sum(int(row["target_loss_tokens"]) for row in samples)
    loss_sum = sum(float(row["loss_sum"]) for row in samples)
    result = {
        "samples": count,
        "target_loss_tokens": target_tokens,
        "loss": loss_sum / target_tokens if target_tokens else None,
        **metrics,
        "script_compliance_rate": sum(bool(row["script_compliant"]) for row in samples) / count,
        "empty_output_rate": sum(bool(row["empty_output"]) for row in samples) / count,
        "source_copy_rate": sum(bool(row["source_copy"]) for row in samples) / count,
        "target_control_rate": sum(bool(row["target_control"]) for row in samples) / count,
        "source_truncation_rate": sum(bool(row["source_truncated"]) for row in samples) / count,
        "target_truncation_rate": sum(bool(row["target_truncated"]) for row in samples) / count,
        "mean_length_ratio": sum(float(row["length_ratio"]) for row in samples) / count,
    }
    return result


def aggregate_results(samples: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    if not samples:
        raise EvaluationContractError("cannot aggregate an empty evaluation split")
    by_route: dict[str, list[Mapping[str, Any]]] = {route: [] for route in ROUTE_ORDER}
    for row in samples:
        route = str(row.get("route", ""))
        if route not in by_route:
            raise EvaluationContractError(f"sample has unsupported route: {route}")
        by_route[route].append(row)
    missing = [route for route, rows in by_route.items() if not rows]
    if missing:
        raise EvaluationContractError(f"evaluation samples are missing routes: {missing}")

    route_metrics = {route: _summarize_samples(by_route[route], config) for route in ROUTE_ORDER}
    product: dict[str, Any] = {}
    conversions: dict[str, Any] = {}
    for route in ROUTE_ORDER:
        source, target = route.split("->")
        source_product = MODEL_TO_PRODUCT[source]
        target_product = MODEL_TO_PRODUCT[target]
        if source_product == target_product:
            conversions[route] = {**route_metrics[route], "tag_routes": [route], "tag_route_weights": {route: 1.0}}
            continue
        direction = f"{source_product}->{target_product}"
        product.setdefault(direction, []).extend(by_route[route])
    if tuple(product) != PRODUCT_DIRECTION_ORDER:
        product = {direction: product[direction] for direction in PRODUCT_DIRECTION_ORDER}
    product_metrics: dict[str, Any] = {}
    for direction, rows in product.items():
        summary = _summarize_samples(rows, config)
        routes = [route for route in ROUTE_ORDER if by_route[route][0]["product_direction"] == direction]
        summary["tag_routes"] = routes
        summary["tag_route_weights"] = {
            route: len(by_route[route]) / len(rows) for route in routes
        }
        product_metrics[direction] = summary
    if set(conversions) != {"zho_Hans->zho_Hant", "zho_Hant->zho_Hans"}:
        raise EvaluationContractError("Chinese conversion routes are incomplete")
    return {
        "overall": _summarize_samples(samples, config),
        "route20": route_metrics,
        "product_directions12": product_metrics,
        "chinese_conversions2": conversions,
    }


def _runtime(config: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch

    device_name = config["runtime"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise EvaluationContractError("CUDA evaluation requested but unavailable")
    device = torch.device(device_name)
    precision = config["runtime"]["precision"]
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise EvaluationContractError("CUDA BF16 evaluation requested but unsupported")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    return device, dtype


def _load_model(checkpoint: Path) -> Any:
    from transformers import M2M100ForConditionalGeneration

    if not checkpoint.is_dir():
        raise EvaluationContractError(f"checkpoint directory is missing: {checkpoint}")
    try:
        model = M2M100ForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True
        )
    except Exception as exc:
        raise EvaluationContractError(f"cannot load offline checkpoint: {exc}") from exc
    return model.eval()


def _batch_loss_sums(loss: Any, labels: Any) -> tuple[float, int]:
    count = int(labels.ne(-100).sum().item())
    return float(loss.detach().float().item()) * count, count


def evaluate_checkpoint(
    *,
    repository_root: Path,
    evaluation_config_path: Path,
    checkpoint: Path,
    split: str,
    allow_test: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    if split not in {"dev", "test"}:
        raise EvaluationContractError("split must be dev or test")
    if split == "test" and not allow_test:
        raise EvaluationContractError("test access requires explicit --allow-test authorization")

    config = load_evaluation_config(evaluation_config_path)
    config_file_sha256 = sha256_file(evaluation_config_path)
    student_path = repository_root / config["identity"]["student_config"]
    if sha256_file(student_path) != config["identity"]["student_config_file_sha256"]:
        raise EvaluationContractError("student config file SHA-256 changed")
    student = load_student_config(student_path)
    if config_sha256(student) != config["identity"]["student_config_canonical_sha256"]:
        raise EvaluationContractError("student config canonical SHA-256 changed")
    manifest_path = repository_root / config["data"]["manifest_path"]
    if sha256_file(manifest_path) != config["data"]["manifest_sha256"]:
        raise EvaluationContractError("finalized split manifest SHA-256 changed")

    data_path = repository_root / config["data"][f"{split}_path"]
    dataset = load_route_dataset(
        data_path,
        expected_sha256=config["data"][f"{split}_sha256"],
        split=split,
        max_records_per_route=int(config["data"]["records_per_route"]),
    )
    tokenizer, tokenizer_report = load_frozen_tokenizer(student, repository_root)
    device, dtype = _runtime(config)
    # Hash the published checkpoint in its native float32 form before applying
    # the configured inference precision.  This keeps identity independent of
    # the selected evaluation device and avoids hashing a rounded runtime copy.
    model = _load_model(checkpoint.resolve())
    alignment = validate_student_alignment(model, tokenizer, student)
    checkpoint_state_sha256 = state_dict_sha256(model)
    model = model.to(device=device, dtype=dtype).eval()
    policy = EncodingPolicy(
        max_source_length=int(config["encoding"]["max_source_length"]),
        max_target_length=int(config["encoding"]["max_target_length"]),
    )
    collator = DirectionAwareCollator(tokenizer, policy)
    language_ids = tokenizer_report["language_token_ids"]
    samples: list[dict[str, Any]] = []
    batch_size = int(config["runtime"]["batch_size"])

    with torch.inference_mode():
        for route in ROUTE_ORDER:
            source_language, target_language = route.split("->")
            records = list(dataset.records_by_route[route])
            for start in range(0, len(records), batch_size):
                record_batch = records[start : start + batch_size]
                batch = collator(record_batch)
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "labels": batch["labels"].to(device),
                }
                output = model(**inputs)
                loss_sum, target_tokens = _batch_loss_sums(output.loss, inputs["labels"])
                token_counts = inputs["labels"].ne(-100).sum(dim=1).tolist()
                # Cross-entropy is a batch mean; assign its exact aggregate evenly by token.
                per_token_loss = loss_sum / target_tokens
                generated = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    forced_bos_token_id=int(language_ids[target_language]),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=int(config["generation"]["max_new_tokens"]),
                    length_penalty=1.0,
                )
                predictions = [
                    normalize_prediction(text)
                    for text in tokenizer.batch_decode(generated, skip_special_tokens=True)
                ]
                for index, (record, prediction) in enumerate(zip(record_batch, predictions, strict=True)):
                    encoded = collator([record])
                    route_stats = encoded["route_statistics"][route]
                    reference = normalize_prediction(str(record["target_text"]))
                    source = normalize_prediction(str(record["source_text"]))
                    compact_prediction = re.sub(r"\s+", "", prediction).casefold()
                    compact_source = re.sub(r"\s+", "", source).casefold()
                    generated_ids = generated[index].detach().cpu().tolist()
                    target_control = len(generated_ids) > 1 and generated_ids[1] == int(language_ids[target_language])
                    reference_length = max(1, len(reference))
                    product_direction = f"{MODEL_TO_PRODUCT[source_language]}->{MODEL_TO_PRODUCT[target_language]}"
                    samples.append(
                        {
                            "sample_id": record["sample_id"],
                            "sample_group_id": record["sample_group_id"],
                            "split": split,
                            "route": route,
                            "product_direction": product_direction,
                            "source_language": source_language,
                            "target_language": target_language,
                            "source": source,
                            "reference": reference,
                            "prediction": prediction,
                            "loss_sum": per_token_loss * int(token_counts[index]),
                            "target_loss_tokens": int(token_counts[index]),
                            "script_compliant": target_script_compliant(prediction, target_language),
                            "empty_output": not bool(prediction),
                            "source_copy": bool(compact_source) and compact_prediction == compact_source,
                            "target_control": target_control,
                            "length_ratio": len(prediction) / reference_length,
                            "source_truncated": route_stats["source_truncated_tokens"] > 0,
                            "target_truncated": route_stats["target_truncated_tokens"] > 0,
                        }
                    )

    aggregates = aggregate_results(samples, config)
    examples_per_route = int(config["runtime"]["fixed_examples_per_route"])
    fixed_examples = {
        route: [
            {
                "sample_id": row["sample_id"], "source": row["source"],
                "reference": row["reference"], "prediction": row["prediction"],
            }
            for row in samples if row["route"] == route
        ][:examples_per_route]
        for route in ROUTE_ORDER
    }
    summary = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "passed",
        "split": split,
        "test_access_explicitly_authorized": split == "test" and allow_test,
        "identities": {
            "evaluation_config_path": evaluation_config_path.relative_to(repository_root).as_posix(),
            "evaluation_config_file_sha256": config_file_sha256,
            "evaluation_config_canonical_sha256": config_sha256(config),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_state_sha256": checkpoint_state_sha256,
            "data_path": config["data"][f"{split}_path"],
            "data_sha256": dataset.file_sha256,
            "selection_sha256": dataset.selection_sha256,
            "split_manifest_sha256": config["data"]["manifest_sha256"],
            "tokenizer_manifest_sha256": tokenizer_report["artifact_manifest_sha256"],
            "encoding_policy_sha256": policy.identity_sha256,
        },
        "runtime": {
            "device": str(device), "precision": config["runtime"]["precision"],
            "batch_size": batch_size, "sacrebleu_version": config["metrics"]["sacrebleu_version"],
        },
        "model_alignment": alignment,
        "records": len(samples),
        "aggregates": aggregates,
        "fixed_examples": fixed_examples,
    }
    return summary, samples


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _markdown(summary: Mapping[str, Any]) -> str:
    overall = summary["aggregates"]["overall"]
    lines = [
        "# MVP evaluation report", "", f"- Split: `{summary['split']}`",
        f"- Checkpoint state SHA-256: `{summary['identities']['checkpoint_state_sha256']}`",
        f"- Samples: {summary['records']}", f"- Loss: {overall['loss']:.6f}",
        f"- SacreBLEU: {overall['sacrebleu']:.4f}", f"- chrF: {overall['chrf']:.4f}",
        f"- Script compliance: {overall['script_compliance_rate']:.4f}", "",
        "## 20 tag routes", "", "| Route | n | Loss | BLEU | chrF | Script | Empty | Copy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route, row in summary["aggregates"]["route20"].items():
        lines.append(
            f"| {route} | {row['samples']} | {row['loss']:.4f} | {row['sacrebleu']:.2f} | "
            f"{row['chrf']:.2f} | {row['script_compliance_rate']:.2f} | "
            f"{row['empty_output_rate']:.2f} | {row['source_copy_rate']:.2f} |"
        )
    lines.extend(["", "## 12 product directions", "", "| Direction | Tag routes | n | BLEU | chrF |", "|---|---|---:|---:|---:|"])
    for direction, row in summary["aggregates"]["product_directions12"].items():
        lines.append(f"| {direction} | {', '.join(row['tag_routes'])} | {row['samples']} | {row['sacrebleu']:.2f} | {row['chrf']:.2f} |")
    lines.extend(["", "## Chinese conversions", "", "| Route | n | BLEU | chrF |", "|---|---:|---:|---:|"])
    for route, row in summary["aggregates"]["chinese_conversions2"].items():
        lines.append(f"| {route} | {row['samples']} | {row['sacrebleu']:.2f} | {row['chrf']:.2f} |")
    return "\n".join(lines) + "\n"


def publish_evaluation(output_directory: Path, summary: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent))
    try:
        (staging / "samples.jsonl").write_bytes(b"".join(_json_bytes(row) for row in samples))
        (staging / "summary.json").write_bytes(_json_bytes(summary))
        (staging / "report.md").write_text(_markdown(summary), encoding="utf-8", newline="\n")
        files = [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(staging.iterdir()) if path.is_file()
        ]
        manifest = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "status": "complete",
            "evaluation_config_file_sha256": summary["identities"]["evaluation_config_file_sha256"],
            "checkpoint_state_sha256": summary["identities"]["checkpoint_state_sha256"],
            "data_sha256": summary["identities"]["data_sha256"],
            "files": files,
        }
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        if output_directory.exists():
            raise EvaluationContractError(f"refusing to overwrite evaluation output: {output_directory}")
        os.replace(staging, output_directory)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
