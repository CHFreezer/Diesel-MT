"""Construction and validation helpers for the Diesel-MT NLLB tokenizer.

The project is intentionally pinned to Transformers 5.13.1, where
``NllbTokenizer`` is backed by the Rust ``tokenizers`` implementation.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import NllbTokenizer


PROJECT_LANGUAGES: tuple[str, ...] = (
    "eng_Latn",
    "zho_Hans",
    "zho_Hant",
    "jpn_Jpan",
    "kor_Hang",
)

TRAINING_LANGUAGES: tuple[str, ...] = (
    "eng_Latn",
    "zho_Hans",
    "jpn_Jpan",
    "kor_Hang",
)

REMOVED_LANGUAGE_PROBES: tuple[str, ...] = (
    "fra_Latn",
    "deu_Latn",
    "rus_Cyrl",
)

REQUIRED_SPECIAL_IDS: dict[str, int] = {
    "<s>": 0,
    "<pad>": 1,
    "</s>": 2,
    "<unk>": 3,
}


class TokenizerValidationError(RuntimeError):
    """Raised when a tokenizer violates a project invariant."""


class LanguageAllowlist:
    """Validate language codes before converting them to token IDs."""

    def __init__(self, allowed: Iterable[str] | None = None) -> None:
        values = PROJECT_LANGUAGES if allowed is None else tuple(allowed)
        self._allowed = frozenset(values)
        if not self._allowed:
            raise ValueError("language allowlist cannot be empty")

    def check(self, lang_code: str) -> str:
        if lang_code not in self._allowed:
            raise ValueError(
                f"Language code {lang_code!r} is not supported. "
                f"Allowed values: {sorted(self._allowed)}"
            )
        return lang_code

    @property
    def allowed(self) -> frozenset[str]:
        return self._allowed


PROJECT_LANGUAGE_ALLOWLIST = LanguageAllowlist()


def create_seed_tokenizer(*, src_lang: str = "eng_Latn") -> NllbTokenizer:
    """Create the empty fast NLLB BPE pipeline used for new training."""
    from transformers import NllbTokenizer

    PROJECT_LANGUAGE_ALLOWLIST.check(src_lang)
    tokenizer = NllbTokenizer(
        extra_special_tokens=list(PROJECT_LANGUAGES),
        src_lang=src_lang,
        legacy_behaviour=False,
    )
    if not tokenizer.is_fast:
        raise TokenizerValidationError("seed NllbTokenizer is not a fast backend")
    verify_backend_pipeline(tokenizer, expected_src_lang=src_lang)
    verify_special_token_ids(tokenizer)
    verify_language_allowlist(tokenizer)
    verify_dense_ids(tokenizer)
    return tokenizer


def verify_special_token_ids(tokenizer: NllbTokenizer) -> dict[str, int]:
    """Validate the M2M100/NLLB core special-token ID contract."""
    actual: dict[str, int] = {}
    for token, expected_id in REQUIRED_SPECIAL_IDS.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        actual[token] = token_id
        if token_id != expected_id:
            raise TokenizerValidationError(
                f"special token {token!r} has ID {token_id}, expected {expected_id}"
            )
    if tokenizer.eos_token_id != 2 or tokenizer.pad_token_id != 1 or tokenizer.unk_token_id != 3:
        raise TokenizerValidationError(
            "tokenizer properties disagree with required eos=2, pad=1, unk=3 IDs"
        )
    return actual


def verify_language_allowlist(tokenizer: NllbTokenizer) -> None:
    """Require exactly the project language tokens, not the NLLB-200 defaults."""
    vocab = tokenizer.get_vocab()
    for language in PROJECT_LANGUAGES:
        token_id = vocab.get(language)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise TokenizerValidationError(
                f"required language token {language!r} is absent or maps to <unk>"
            )
        encoded = tokenizer.encode(language, add_special_tokens=False)
        if encoded != [token_id]:
            raise TokenizerValidationError(
                f"language token {language!r} is split during encoding: {encoded}"
            )
    leaked = [language for language in REMOVED_LANGUAGE_PROBES if language in vocab]
    if leaked:
        raise TokenizerValidationError(f"removed NLLB language tokens leaked into vocab: {leaked}")


def build_language_mapping(tokenizer: NllbTokenizer) -> dict[str, int]:
    """Build a validated language-token to ID mapping from a loaded tokenizer."""
    verify_language_allowlist(tokenizer)
    mapping: dict[str, int] = {}
    for language in PROJECT_LANGUAGES:
        PROJECT_LANGUAGE_ALLOWLIST.check(language)
        token_id = tokenizer.convert_tokens_to_ids(language)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise TokenizerValidationError(f"language token {language!r} maps to <unk>")
        mapping[language] = token_id
    if len(set(mapping.values())) != len(mapping):
        raise TokenizerValidationError(f"language token IDs are not unique: {mapping}")
    return mapping


def forced_bos_token_id(tokenizer: NllbTokenizer, target_language: str) -> int:
    """Return a target language ID after application-level validation."""
    PROJECT_LANGUAGE_ALLOWLIST.check(target_language)
    token_id = tokenizer.convert_tokens_to_ids(target_language)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise TokenizerValidationError(
            f"target language {target_language!r} has no usable token ID"
        )
    return token_id


def verify_dense_ids(tokenizer: NllbTokenizer) -> int:
    """Ensure the vocabulary IDs are exactly ``0..len(tokenizer)-1``."""
    vocab = tokenizer.get_vocab()
    ids = list(vocab.values())
    if len(ids) != len(set(ids)):
        raise TokenizerValidationError("vocabulary contains duplicate token IDs")
    expected = list(range(len(tokenizer)))
    if sorted(ids) != expected:
        missing = sorted(set(expected) - set(ids))[:20]
        extra = sorted(set(ids) - set(expected))[:20]
        raise TokenizerValidationError(
            f"vocabulary IDs are not dense: missing={missing}, extra={extra}"
        )
    return len(ids)


def backend_pipeline(tokenizer: NllbTokenizer) -> dict:
    """Return the serialized Rust backend configuration."""
    return json.loads(tokenizer.backend_tokenizer.to_str())


def verify_backend_pipeline(
    tokenizer: NllbTokenizer,
    *,
    expected_src_lang: str | None = None,
) -> dict:
    """Validate the locked BPE, Metaspace, decoder and NLLB processor setup."""
    if not tokenizer.is_fast:
        raise TokenizerValidationError("tokenizer is not backed by tokenizers")
    config = backend_pipeline(tokenizer)
    model = config.get("model") or {}
    if model.get("type") != "BPE":
        raise TokenizerValidationError(f"backend model is not BPE: {model.get('type')!r}")
    if model.get("unk_token") != "<unk>":
        raise TokenizerValidationError(
            f"backend unk_token is {model.get('unk_token')!r}, expected '<unk>'"
        )
    if model.get("fuse_unk") is not True or model.get("byte_fallback") is not False:
        raise TokenizerValidationError(
            "backend must use fuse_unk=true and byte_fallback=false"
        )
    pre_tokenizer = config.get("pre_tokenizer") or {}
    decoder = config.get("decoder") or {}
    if pre_tokenizer.get("type") != "Metaspace" or pre_tokenizer.get("prepend_scheme") != "always":
        raise TokenizerValidationError(f"unexpected pre-tokenizer config: {pre_tokenizer}")
    if decoder.get("type") != "Metaspace" or decoder.get("prepend_scheme") != "always":
        raise TokenizerValidationError(f"unexpected decoder config: {decoder}")
    if expected_src_lang is not None:
        PROJECT_LANGUAGE_ALLOWLIST.check(expected_src_lang)
        processor = config.get("post_processor") or {}
        if processor.get("type") != "TemplateProcessing":
            raise TokenizerValidationError(f"unexpected post-processor: {processor}")
        special = processor.get("special_tokens") or {}
        expected_id = tokenizer.convert_tokens_to_ids(expected_src_lang)
        entry = special.get(expected_src_lang) or {}
        if entry.get("ids") != [expected_id]:
            raise TokenizerValidationError(
                f"post-processor does not prefix {expected_src_lang!r}: {processor}"
            )
        eos_entry = special.get("</s>") or {}
        if eos_entry.get("ids") != [tokenizer.eos_token_id]:
            raise TokenizerValidationError("post-processor does not suffix </s>")
    return config


def verify_tokenizer(tokenizer: NllbTokenizer, *, expected_vocab_size: int | None = None) -> None:
    """Run all tokenizer invariants used at save/reload boundaries."""
    verify_special_token_ids(tokenizer)
    verify_language_allowlist(tokenizer)
    verify_dense_ids(tokenizer)
    saved_src_lang = tokenizer.src_lang
    try:
        for language in PROJECT_LANGUAGES:
            tokenizer.src_lang = language
            verify_backend_pipeline(tokenizer, expected_src_lang=language)
            forced_bos_token_id(tokenizer, language)
    finally:
        tokenizer.src_lang = saved_src_lang
    if expected_vocab_size is not None and len(tokenizer) != expected_vocab_size:
        raise TokenizerValidationError(
            f"vocabulary size is {len(tokenizer)}, expected {expected_vocab_size}"
        )


def reload_tokenizer(artifact_dir: Path) -> NllbTokenizer:
    """Load a local artifact and reject slow or wrong tokenizer classes."""
    from transformers import AutoTokenizer, NllbTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir), local_files_only=True)
    if not isinstance(tokenizer, NllbTokenizer):
        raise TokenizerValidationError(
            f"expected NllbTokenizer, got {type(tokenizer).__name__}"
        )
    verify_tokenizer(tokenizer)
    return tokenizer


def atomic_write_json(path: Path, payload: Mapping | list) -> None:
    """Write JSON through a same-directory temporary file and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def save_language_mapping(mapping: Mapping[str, int], path: Path) -> None:
    """Persist the validated language mapping atomically."""
    expected_keys = set(PROJECT_LANGUAGES)
    if set(mapping) != expected_keys:
        raise TokenizerValidationError(
            f"language mapping keys are {sorted(mapping)}, expected {sorted(expected_keys)}"
        )
    atomic_write_json(
        path,
        {
            "description": "Diesel-MT language token to ID mapping",
            "mapping": dict(mapping),
            "usage": "Validate tgt_lang, then use convert_tokens_to_ids(tgt_lang) as forced_bos_token_id.",
        },
    )
