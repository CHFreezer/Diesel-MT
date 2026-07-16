"""TD-09 student construction and direction-aware encoding contracts.

The module deliberately stops at a single batch.  Multi-step optimization,
checkpointing, and quality evaluation belong to TD-10, TD-11, and TD-13.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import struct
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from freeze_tokenizer_artifact import sha256_file, verify_artifact_files
from model_training_contract import (
    LANGUAGE_TAGS,
    TOKENIZER_MANIFEST_SHA256,
    config_sha256,
    directed_routes,
    load_student_config,
    validate_route,
)
from tokenizer_utils import (
    build_language_mapping,
    reload_tokenizer,
    verify_tokenizer,
)


LABEL_PAD_ID = -100
DEFAULT_MAX_SOURCE_LENGTH = 128
DEFAULT_MAX_TARGET_LENGTH = 128
TD09_SCHEMA_VERSION = 1


class StudentContractError(RuntimeError):
    """Raised when TD-09 tokenizer, encoding, or model invariants fail."""


@dataclass(frozen=True)
class EncodingPolicy:
    """Identity-bearing single-batch encoding policy.

    TD-14 may select shorter maxima in a resource profile, but it may not
    change these semantic rules or exceed the model position ceiling.
    """

    max_source_length: int = DEFAULT_MAX_SOURCE_LENGTH
    max_target_length: int = DEFAULT_MAX_TARGET_LENGTH
    model_position_ceiling: int = 1_024
    overflow_policy: str = "truncate_preserve_language_and_eos"
    empty_text_policy: str = "reject"
    label_pad_id: int = LABEL_PAD_ID

    def __post_init__(self) -> None:
        for name in ("max_source_length", "max_target_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 3:
                raise StudentContractError(f"{name} must be an integer >= 3")
            if value > self.model_position_ceiling:
                raise StudentContractError(
                    f"{name}={value} exceeds model position ceiling "
                    f"{self.model_position_ceiling}"
                )
        if self.model_position_ceiling != 1_024:
            raise StudentContractError("model position ceiling must remain 1,024")
        if self.overflow_policy != "truncate_preserve_language_and_eos":
            raise StudentContractError("unsupported overflow policy")
        if self.empty_text_policy != "reject":
            raise StudentContractError("empty text must be rejected")
        if self.label_pad_id != LABEL_PAD_ID:
            raise StudentContractError("label padding must use the -100 loss ignore index")

    @property
    def identity_sha256(self) -> str:
        return config_sha256(asdict(self))


@dataclass(frozen=True)
class EncodedSample:
    sample_id: str
    sample_group_id: str
    source_language: str
    target_language: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    source_original_tokens: int
    target_original_tokens: int

    @property
    def route(self) -> str:
        return f"{self.source_language}->{self.target_language}"

    @property
    def source_truncated_tokens(self) -> int:
        return self.source_original_tokens - len(self.input_ids)

    @property
    def target_truncated_tokens(self) -> int:
        return self.target_original_tokens - len(self.labels)


def _require_nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StudentContractError(f"{field} must be non-empty after trimming")
    return value


def _require_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise StudentContractError(f"{field} must be a non-empty string")
    return value


def _encoded_ids(tokenizer: object, text: str, language: str) -> list[int]:
    original_language = tokenizer.src_lang
    try:
        tokenizer.src_lang = language
        encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    finally:
        tokenizer.src_lang = original_language
    ids = encoded.get("input_ids")
    if not isinstance(ids, list) or not ids or any(
        isinstance(value, bool) or not isinstance(value, int) for value in ids
    ):
        raise StudentContractError("tokenizer returned invalid input_ids")
    return ids


def _validate_sequence(
    ids: Sequence[int],
    *,
    language_id: int,
    eos_id: int,
    vocab_size: int,
    context: str,
) -> None:
    if len(ids) < 2 or ids[0] != language_id or ids[-1] != eos_id:
        raise StudentContractError(
            f"{context} must be prefixed by its language token and suffixed by </s>"
        )
    invalid = [value for value in ids if value < 0 or value >= vocab_size]
    if invalid:
        raise StudentContractError(f"{context} contains token IDs outside the vocabulary")


def _truncate_preserving_contract(ids: Sequence[int], maximum: int, eos_id: int) -> tuple[int, ...]:
    if len(ids) <= maximum:
        return tuple(ids)
    return tuple([*ids[: maximum - 1], eos_id])


def encode_language_text(
    tokenizer: object,
    text: str,
    language: str,
    *,
    language_mapping: Mapping[str, int] | None = None,
    vocab_size: int | None = None,
) -> tuple[int, ...]:
    """Encode one unique language/text value for safe cross-route reuse."""

    value = _require_nonempty_text(text, "text")
    mapping = language_mapping or build_language_mapping(tokenizer)
    if set(mapping) != set(LANGUAGE_TAGS) or language not in mapping:
        raise StudentContractError(
            "tokenizer language-token set differs from the five-tag contract"
        )
    ids = tuple(_encoded_ids(tokenizer, value, language))
    _validate_sequence(
        ids,
        language_id=mapping[language],
        eos_id=tokenizer.eos_token_id,
        vocab_size=len(tokenizer) if vocab_size is None else vocab_size,
        context="language/text encoding",
    )
    return ids


def encoded_sample_from_sequences(
    tokenizer: object,
    sample: Mapping[str, Any],
    policy: EncodingPolicy,
    *,
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    language_mapping: Mapping[str, int] | None = None,
    vocab_size: int | None = None,
) -> EncodedSample:
    """Build a validated sample from identity-bound reusable token sequences."""

    source_language = str(sample.get("src_lang", ""))
    target_language = str(sample.get("tgt_lang", ""))
    try:
        validate_route(source_language, target_language)
    except ValueError as exc:
        raise StudentContractError(str(exc)) from exc
    sample_id = _require_identifier(sample.get("sample_id"), "sample_id")
    sample_group_id = _require_identifier(
        sample.get("sample_group_id"), "sample_group_id"
    )
    _require_nonempty_text(sample.get("source_text"), "source_text")
    _require_nonempty_text(sample.get("target_text"), "target_text")
    mapping = language_mapping or build_language_mapping(tokenizer)
    if set(mapping) != set(LANGUAGE_TAGS):
        raise StudentContractError(
            "tokenizer language-token set differs from the five-tag contract"
        )
    eos_id = tokenizer.eos_token_id
    resolved_vocab_size = len(tokenizer) if vocab_size is None else vocab_size
    _validate_sequence(
        source_ids,
        language_id=mapping[source_language],
        eos_id=eos_id,
        vocab_size=resolved_vocab_size,
        context="source encoding",
    )
    _validate_sequence(
        target_ids,
        language_id=mapping[target_language],
        eos_id=eos_id,
        vocab_size=resolved_vocab_size,
        context="target labels",
    )
    truncated_source = _truncate_preserving_contract(
        source_ids, policy.max_source_length, eos_id
    )
    truncated_target = _truncate_preserving_contract(
        target_ids, policy.max_target_length, eos_id
    )
    _validate_sequence(
        truncated_source,
        language_id=mapping[source_language],
        eos_id=eos_id,
        vocab_size=resolved_vocab_size,
        context="truncated source encoding",
    )
    _validate_sequence(
        truncated_target,
        language_id=mapping[target_language],
        eos_id=eos_id,
        vocab_size=resolved_vocab_size,
        context="truncated target labels",
    )
    return EncodedSample(
        sample_id=sample_id,
        sample_group_id=sample_group_id,
        source_language=source_language,
        target_language=target_language,
        input_ids=truncated_source,
        labels=truncated_target,
        source_original_tokens=len(source_ids),
        target_original_tokens=len(target_ids),
    )


def encode_parallel_sample(
    tokenizer: object,
    sample: Mapping[str, Any],
    policy: EncodingPolicy,
    *,
    language_mapping: Mapping[str, int] | None = None,
    vocab_size: int | None = None,
) -> EncodedSample:
    """Encode one route while preserving NLLB/M2M100 language control."""

    source_language = str(sample.get("src_lang", ""))
    target_language = str(sample.get("tgt_lang", ""))
    try:
        validate_route(source_language, target_language)
    except ValueError as exc:
        raise StudentContractError(str(exc)) from exc
    source_text = _require_nonempty_text(sample.get("source_text"), "source_text")
    target_text = _require_nonempty_text(sample.get("target_text"), "target_text")
    mapping = language_mapping or build_language_mapping(tokenizer)
    resolved_vocab_size = len(tokenizer) if vocab_size is None else vocab_size
    source_ids = encode_language_text(
        tokenizer,
        source_text,
        source_language,
        language_mapping=mapping,
        vocab_size=resolved_vocab_size,
    )
    target_ids = encode_language_text(
        tokenizer,
        target_text,
        target_language,
        language_mapping=mapping,
        vocab_size=resolved_vocab_size,
    )
    return encoded_sample_from_sequences(
        tokenizer,
        sample,
        policy,
        source_ids=source_ids,
        target_ids=target_ids,
        language_mapping=mapping,
        vocab_size=resolved_vocab_size,
    )


def _route_statistics(samples: Sequence[EncodedSample]) -> dict[str, dict[str, int]]:
    statistics: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "samples": 0,
            "source_original_tokens": 0,
            "source_used_tokens": 0,
            "source_truncated_tokens": 0,
            "target_original_tokens": 0,
            "target_used_tokens": 0,
            "target_truncated_tokens": 0,
        }
    )
    for sample in samples:
        row = statistics[sample.route]
        row["samples"] += 1
        row["source_original_tokens"] += sample.source_original_tokens
        row["source_used_tokens"] += len(sample.input_ids)
        row["source_truncated_tokens"] += sample.source_truncated_tokens
        row["target_original_tokens"] += sample.target_original_tokens
        row["target_used_tokens"] += len(sample.labels)
        row["target_truncated_tokens"] += sample.target_truncated_tokens
    return {route: statistics[route] for route in sorted(statistics)}


class DirectionAwareCollator:
    """Pad a validated multilingual batch without losing route metadata."""

    def __init__(self, tokenizer: object, policy: EncodingPolicy) -> None:
        self.tokenizer = tokenizer
        self.policy = policy
        self.language_mapping = build_language_mapping(tokenizer)
        self.vocab_size = len(tokenizer)
        if set(self.language_mapping) != set(LANGUAGE_TAGS):
            raise StudentContractError(
                "tokenizer language-token set differs from the five-tag contract"
            )
        if tokenizer.pad_token_id != 1:
            raise StudentContractError("tokenizer pad token ID must be 1")

    def __call__(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not records:
            raise StudentContractError("cannot collate an empty batch")
        samples = [
            encode_parallel_sample(
                self.tokenizer,
                record,
                self.policy,
                language_mapping=self.language_mapping,
                vocab_size=self.vocab_size,
            )
            for record in records
        ]
        return self.collate_encoded(samples)

    def collate_encoded(self, samples: Sequence[EncodedSample]) -> dict[str, Any]:
        """Pad already encoded samples while preserving their input order."""

        import torch

        if not samples:
            raise StudentContractError("cannot collate an empty encoded batch")
        max_source = max(len(sample.input_ids) for sample in samples)
        max_target = max(len(sample.labels) for sample in samples)
        input_ids = torch.full(
            (len(samples), max_source),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(samples), max_source), dtype=torch.long)
        labels = torch.full(
            (len(samples), max_target), LABEL_PAD_ID, dtype=torch.long
        )
        for index, sample in enumerate(samples):
            source = torch.tensor(sample.input_ids, dtype=torch.long)
            target = torch.tensor(sample.labels, dtype=torch.long)
            input_ids[index, : len(source)] = source
            attention_mask[index, : len(source)] = 1
            labels[index, : len(target)] = target
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_ids": [sample.sample_id for sample in samples],
            "sample_group_ids": [sample.sample_group_id for sample in samples],
            "routes": [sample.route for sample in samples],
            "route_statistics": _route_statistics(samples),
            "encoding_policy_sha256": self.policy.identity_sha256,
        }


def model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Strip audit metadata before forwarding a collated batch to Transformers."""

    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }


def load_frozen_tokenizer(
    student_config: Mapping[str, Any], repository_root: Path
) -> tuple[object, dict[str, Any]]:
    tokenizer_spec = student_config["tokenizer"]
    artifact_dir = (repository_root / tokenizer_spec["path"]).resolve()
    if not artifact_dir.is_dir():
        raise StudentContractError(f"frozen tokenizer directory is missing: {artifact_dir}")
    try:
        _, manifest_sha256 = verify_artifact_files(artifact_dir)
    except Exception as exc:
        raise StudentContractError(f"frozen tokenizer verification failed: {exc}") from exc
    expected_sha256 = str(tokenizer_spec["artifact_manifest_sha256"])
    if expected_sha256 != TOKENIZER_MANIFEST_SHA256 or manifest_sha256 != expected_sha256:
        raise StudentContractError("frozen tokenizer manifest SHA-256 changed")
    tokenizer = reload_tokenizer(artifact_dir)
    verify_tokenizer(
        tokenizer, expected_vocab_size=int(tokenizer_spec["expected_vocab_size"])
    )
    mapping = build_language_mapping(tokenizer)
    expected_languages = tuple(tokenizer_spec["required_language_tokens"])
    if tuple(expected_languages) != LANGUAGE_TAGS or set(mapping) != set(expected_languages):
        raise StudentContractError("frozen tokenizer language-token contract changed")
    return tokenizer, {
        "path": str(tokenizer_spec["path"]),
        "artifact_manifest_sha256": manifest_sha256,
        "vocab_size": len(tokenizer),
        "language_token_ids": mapping,
        "is_fast": bool(tokenizer.is_fast),
    }


def construct_m2m100(
    tokenizer: object, model_values: Mapping[str, Any], seed: int
) -> object:
    """Construct a deterministic CPU M2M100 model without changing caller RNG state."""

    import torch
    from transformers import M2M100Config, M2M100ForConditionalGeneration

    python_state = random.getstate()
    try:
        random.seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = M2M100ForConditionalGeneration(
                M2M100Config(**dict(model_values))
            )
    finally:
        random.setstate(python_state)
    model.tie_weights()
    return model


def validate_student_alignment(
    model: object, tokenizer: object, student_config: Mapping[str, Any]
) -> dict[str, Any]:
    expected = student_config["model"]
    config = model.config
    checks = {
        "vocab_size": int(config.vocab_size),
        "input_embeddings": int(model.get_input_embeddings().num_embeddings),
        "output_embeddings": int(model.get_output_embeddings().out_features),
        "bos_token_id": int(config.bos_token_id),
        "pad_token_id": int(config.pad_token_id),
        "eos_token_id": int(config.eos_token_id),
        "decoder_start_token_id": int(config.decoder_start_token_id),
        "tied_embeddings": (
            model.get_input_embeddings().weight.data_ptr()
            == model.get_output_embeddings().weight.data_ptr()
        ),
    }
    if checks["vocab_size"] != len(tokenizer) or checks["vocab_size"] != 49_152:
        raise StudentContractError("model vocabulary does not match the frozen tokenizer")
    if checks["input_embeddings"] != 49_152 or checks["output_embeddings"] != 49_152:
        raise StudentContractError("a vocabulary-facing model dimension is not 49,152")
    for name in ("bos_token_id", "pad_token_id", "eos_token_id", "decoder_start_token_id"):
        if checks[name] != int(expected[name]):
            raise StudentContractError(f"model {name} differs from the student config")
    if not checks["tied_embeddings"] or not bool(config.tie_word_embeddings):
        raise StudentContractError("student input and output embeddings are not tied")
    generation = model.generation_config
    for name in ("bos_token_id", "pad_token_id", "eos_token_id", "decoder_start_token_id"):
        if getattr(generation, name) != checks[name]:
            raise StudentContractError(f"generation config {name} is misaligned")
    return checks


def build_student(
    student_config: Mapping[str, Any], tokenizer: object
) -> tuple[object, dict[str, Any]]:
    model = construct_m2m100(
        tokenizer,
        student_config["model"],
        int(student_config["identity"]["random_seed"]),
    )
    alignment = validate_student_alignment(model, tokenizer, student_config)
    return model, alignment


def state_dict_sha256(model: object) -> str:
    """Hash names, dtypes, shapes, and canonical CPU tensor bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(value.dtype).encode("ascii")
        digest.update(struct.pack("<I", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack("<I", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack("<I", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack("<Q", dimension))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def load_route_fixture(path: Path, *, split: str = "train") -> list[dict[str, Any]]:
    """Read only until one frozen sample for each of the 20 routes is found."""

    expected = set(directed_routes())
    found: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StudentContractError(
                    f"invalid fixture JSON at line {line_number}: {exc}"
                ) from exc
            if record.get("split") != split:
                raise StudentContractError(
                    f"fixture {path} contains a non-{split} record"
                )
            route = (str(record.get("src_lang")), str(record.get("tgt_lang")))
            if route not in expected:
                raise StudentContractError(f"fixture contains an unsupported route: {route}")
            found.setdefault(route, record)
            if len(found) == len(expected):
                break
    missing = sorted(expected - set(found))
    if missing:
        raise StudentContractError(f"fixture is missing routes: {missing}")
    return [found[route] for route in directed_routes()]


def _file_records(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in directory.rglob("*") if item.is_file())
    ]


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_and_reload_student(
    output_dir: Path,
    *,
    model: object,
    tokenizer: object,
    student_config: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Save a TD-09 HF artifact and prove a fully local reload."""

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if output_dir.exists():
        raise StudentContractError(f"TD-09 output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        original_hash = state_dict_sha256(model)
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        payload_files = _file_records(staging)
        manifest = {
            "schema_version": TD09_SCHEMA_VERSION,
            "status": "complete",
            "purpose": "TD-09 offline save/reload acceptance; random weights only",
            "student_config_sha256": config_sha256(student_config),
            "state_dict_sha256": original_hash,
            "tokenizer": dict(tokenizer_identity),
            "files": payload_files,
        }
        _atomic_write_json(staging / "manifest.json", manifest)
        reloaded_tokenizer = AutoTokenizer.from_pretrained(
            staging, local_files_only=True
        )
        verify_tokenizer(reloaded_tokenizer, expected_vocab_size=49_152)
        if reloaded_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise StudentContractError("tokenizer vocabulary changed across TD-09 save/reload")
        reloaded_model = AutoModelForSeq2SeqLM.from_pretrained(
            staging, local_files_only=True
        )
        validate_student_alignment(reloaded_model, reloaded_tokenizer, student_config)
        reloaded_hash = state_dict_sha256(reloaded_model)
        if reloaded_hash != original_hash:
            raise StudentContractError("student state changed across offline save/reload")
        os.replace(staging, output_dir)
        return {
            "path": output_dir.as_posix(),
            "state_dict_sha256": reloaded_hash,
            "manifest_sha256": sha256_file(output_dir / "manifest.json"),
            "files": payload_files,
            "offline_model_reload": True,
            "offline_tokenizer_reload": True,
        }
    except BaseException:
        if staging.exists():
            import shutil

            shutil.rmtree(staging, ignore_errors=True)
        raise


def run_td09_acceptance(
    *,
    repository_root: Path,
    student_config_path: Path,
    fixture_path: Path,
    checkpoint_output: Path,
    policy: EncodingPolicy | None = None,
) -> dict[str, Any]:
    """Run the full 20-route CPU forward/backward and offline reload gate."""

    import torch

    policy = policy or EncodingPolicy()
    student_config = load_student_config(student_config_path)
    tokenizer, tokenizer_identity = load_frozen_tokenizer(
        student_config, repository_root
    )
    records = load_route_fixture(fixture_path, split="train")
    collator = DirectionAwareCollator(tokenizer, policy)
    batch = collator(records)

    first_model, _ = build_student(student_config, tokenizer)
    first_hash = state_dict_sha256(first_model)
    del first_model
    gc.collect()
    model, alignment = build_student(student_config, tokenizer)
    second_hash = state_dict_sha256(model)
    if first_hash != second_hash:
        raise StudentContractError("same seed/config produced different initial states")

    model.train()
    output = model(**model_inputs(batch))
    if output.loss is None or not bool(torch.isfinite(output.loss).item()):
        raise StudentContractError("20-route CPU forward produced a non-finite loss")
    output.loss.backward()
    gradient_tensors = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradient_tensors or not all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradient_tensors
    ):
        raise StudentContractError("20-route CPU backward produced invalid gradients")
    model.zero_grad(set_to_none=True)
    model.eval()

    checkpoint = save_and_reload_student(
        checkpoint_output,
        model=model,
        tokenizer=tokenizer,
        student_config=student_config,
        tokenizer_identity=tokenizer_identity,
    )
    return {
        "schema_version": TD09_SCHEMA_VERSION,
        "status": "complete",
        "task": "TD-09",
        "student_config": {
            "path": student_config_path.relative_to(repository_root).as_posix(),
            "sha256": sha256_file(student_config_path),
            "canonical_sha256": config_sha256(student_config),
        },
        "tokenizer": tokenizer_identity,
        "encoding_policy": {**asdict(policy), "sha256": policy.identity_sha256},
        "fixture": {
            "path": fixture_path.relative_to(repository_root).as_posix(),
            "sha256": sha256_file(fixture_path),
            "records": len(records),
            "routes": sorted(set(batch["routes"])),
        },
        "model": {
            "initial_state_dict_sha256": second_hash,
            "deterministic_rebuild": True,
            "alignment": alignment,
        },
        "smoke": {
            "device": "cpu",
            "forward_loss": float(output.loss.detach().cpu().item()),
            "backward_gradients_finite": True,
            "routes": len(set(batch["routes"])),
            "route_statistics": batch["route_statistics"],
        },
        "checkpoint": checkpoint,
        "warning": "Random initialization and fixture memorization plumbing only; no translation quality.",
    }
