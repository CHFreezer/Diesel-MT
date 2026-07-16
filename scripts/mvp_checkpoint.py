"""TD-11 atomic, identity-bound checkpoint save and exact restore."""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from freeze_tokenizer_artifact import sha256_file
from model_training_contract import config_sha256


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_MANIFEST = "checkpoint-manifest.json"
PAYLOAD_FILES = (
    "model.pt",
    "gradients.pt",
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "rng.pt",
    "trainer-state.pt",
)
RETENTION_KEEP_LAST = 3


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is incomplete, damaged, or identity-mismatched."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
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


def _torch_save(path: Path, payload: Any) -> None:
    import torch

    with path.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _canonical_torch_tree(value: Any) -> Any:
    """Remove device/storage-layout noise before deterministic torch serialization."""

    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, Mapping):
        return {
            key: _canonical_torch_tree(value[key])
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item)))
        }
    if isinstance(value, list):
        return [_canonical_torch_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonical_torch_tree(item) for item in value)
    return value


def _torch_load(path: Path, *, map_location: Any = "cpu") -> Any:
    import torch

    return torch.load(path, map_location=map_location, weights_only=True)


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = os.lstat(path).st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & 0x400)


def _safe_payload_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CheckpointError("checkpoint manifest contains an unsafe payload path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise CheckpointError("checkpoint manifest contains path traversal")
    if path.as_posix() != value:
        raise CheckpointError("checkpoint manifest payload path is not normalized")
    return value


def capture_rng_state() -> dict[str, Any]:
    import numpy
    import torch

    numpy_state = numpy.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    import numpy
    import torch

    random.setstate(state["python"])
    numpy_state = state["numpy"]
    numpy.random.set_state(
        (
            str(numpy_state["algorithm"]),
            numpy.asarray(numpy_state["keys"], dtype=numpy.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda", [])
    if cuda_states:
        if not torch.cuda.is_available():
            raise CheckpointError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise CheckpointError("checkpoint CUDA RNG device count changed")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def _gradient_state(model: object) -> dict[str, Any]:
    return {
        name: (parameter.grad.detach().cpu() if parameter.grad is not None else None)
        for name, parameter in model.named_parameters()
    }


def _restore_gradients(model: object, gradients: Mapping[str, Any]) -> None:
    parameters = dict(model.named_parameters())
    if set(gradients) != set(parameters):
        raise CheckpointError("checkpoint gradient parameter set changed")
    for name, parameter in parameters.items():
        gradient = gradients[name]
        parameter.grad = None if gradient is None else gradient.to(parameter.device)


def _validate_trainer_state(state: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "global_step",
        "micro_step",
        "epoch",
        "consumed_samples",
        "consumed_tokens",
        "accumulation_phase",
        "loss_history",
        "route_counts",
        "token_audit",
        "sampler_state",
    }
    missing = sorted(required - set(state))
    unknown = sorted(set(state) - required)
    if missing:
        raise CheckpointError(f"trainer state missing fields: {', '.join(missing)}")
    if unknown:
        raise CheckpointError(f"trainer state unknown fields: {', '.join(unknown)}")
    for field in (
        "global_step",
        "micro_step",
        "epoch",
        "consumed_samples",
        "consumed_tokens",
        "accumulation_phase",
    ):
        value = state[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CheckpointError(f"trainer state {field} must be a non-negative integer")
    if not isinstance(state["loss_history"], list):
        raise CheckpointError("trainer state loss_history must be a list")
    if not isinstance(state["route_counts"], Mapping) or not isinstance(
        state["token_audit"], Mapping
    ):
        raise CheckpointError("trainer route/token audit state must be mappings")
    if not isinstance(state["sampler_state"], Mapping):
        raise CheckpointError("trainer sampler_state must be a mapping")
    return dict(state)


def checkpoint_name(global_step: int) -> str:
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise CheckpointError("global_step must be a non-negative integer")
    return f"step-{global_step:08d}"


def build_checkpoint_identity(
    *,
    repository_root: Path,
    training_report: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> dict[str, Any]:
    code_files = (
        "scripts/mvp_student.py",
        "scripts/mvp_training.py",
        "scripts/mvp_checkpoint.py",
        "scripts/train_mvp_model.py",
    )
    code = {
        relative: sha256_file(repository_root / relative) for relative in code_files
    }
    runtime = training_report["runtime"]
    return {
        "training_config_sha256": config_sha256(training_config),
        "student_config_sha256": training_report["student_config_canonical_sha256"],
        "tokenizer_manifest_sha256": training_config["identity"][
            "tokenizer_manifest_sha256"
        ],
        "data": {
            name: record["sha256"] for name, record in training_report["inputs"].items()
        },
        "code": code,
        "code_sha256": config_sha256(code),
        "dependencies_sha256": config_sha256(runtime["packages"]),
        "git": dict(training_report["git"]),
        "runtime": {
            "device": runtime["selected_device"],
            "precision": runtime["selected_precision"],
            "torch": runtime["torch"],
            "cuda_runtime": runtime["cuda_runtime"],
            "cuda_device_count": runtime["cuda_device_count"],
        },
    }


def save_checkpoint(
    checkpoint_root: Path,
    *,
    model: object,
    optimizer: object,
    scheduler: object,
    scaler: object,
    sampler: object,
    trainer_state: Mapping[str, Any],
    identity: Mapping[str, Any],
    fault_injector: Callable[[str], None] | None = None,
) -> Path:
    """Stage, hash, validate, and atomically publish one complete checkpoint."""

    state = dict(trainer_state)
    state["sampler_state"] = sampler.state_dict()
    state = _validate_trainer_state(state)
    name = checkpoint_name(state["global_step"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(checkpoint_root):
        raise CheckpointError("checkpoint root must not be a symlink or reparse point")
    target = checkpoint_root / name
    if target.exists():
        raise CheckpointError(f"checkpoint already exists: {target}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{name}.staging-", dir=checkpoint_root)
    )

    def inject(point: str) -> None:
        if fault_injector is not None:
            fault_injector(point)

    try:
        _torch_save(staging / "model.pt", model.state_dict())
        inject("after_model")
        _torch_save(staging / "gradients.pt", _gradient_state(model))
        _torch_save(
            staging / "optimizer.pt",
            _canonical_torch_tree(optimizer.state_dict()),
        )
        inject("after_optimizer")
        _torch_save(staging / "scheduler.pt", scheduler.state_dict())
        _torch_save(staging / "scaler.pt", scaler.state_dict())
        _torch_save(staging / "rng.pt", capture_rng_state())
        _torch_save(staging / "trainer-state.pt", state)
        files = [
            {
                "path": filename,
                "bytes": (staging / filename).stat().st_size,
                "sha256": sha256_file(staging / filename),
            }
            for filename in PAYLOAD_FILES
        ]
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "status": "complete",
            "checkpoint_id": name,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "summary": {
                field: state[field]
                for field in (
                    "global_step",
                    "micro_step",
                    "epoch",
                    "consumed_samples",
                    "consumed_tokens",
                    "accumulation_phase",
                )
            },
            "identity": dict(identity),
            "identity_sha256": config_sha256(identity),
            "files": files,
            "retention": {
                "policy": "keep-last-complete-after-new-checkpoint-validation",
                "keep_last": RETENTION_KEEP_LAST,
                "automatic_prune": False,
            },
        }
        inject("before_manifest")
        _atomic_json(staging / CHECKPOINT_MANIFEST, manifest)
        inject("after_manifest_before_publish")
        validate_checkpoint(staging, expected_identity=identity)
        os.replace(staging, target)
        validate_checkpoint(target, expected_identity=identity)
        return target
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_checkpoint(
    path: Path, *, expected_identity: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    if not path.is_dir() or _is_link_or_reparse(path):
        raise CheckpointError("checkpoint must be a real directory")
    manifest_path = path / CHECKPOINT_MANIFEST
    if not manifest_path.is_file() or _is_link_or_reparse(manifest_path):
        raise CheckpointError("checkpoint manifest is missing or linked")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"cannot read checkpoint manifest: {exc}") from exc
    required = {
        "schema_version",
        "status",
        "checkpoint_id",
        "created_at_utc",
        "summary",
        "identity",
        "identity_sha256",
        "files",
        "retention",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise CheckpointError("checkpoint manifest fields are incomplete or unknown")
    if manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION or manifest["status"] != "complete":
        raise CheckpointError("checkpoint manifest is not complete")
    checkpoint_id = str(manifest["checkpoint_id"])
    valid_staging_name = path.name.startswith(f".{checkpoint_id}.staging-")
    if checkpoint_id != path.name and not valid_staging_name:
        raise CheckpointError("checkpoint directory name differs from its manifest")
    identity = manifest["identity"]
    if not isinstance(identity, Mapping) or config_sha256(identity) != manifest["identity_sha256"]:
        raise CheckpointError("checkpoint identity hash is invalid")
    if expected_identity is not None and dict(identity) != dict(expected_identity):
        raise CheckpointError("checkpoint identity does not match this run")
    files = manifest["files"]
    if not isinstance(files, list):
        raise CheckpointError("checkpoint file list is invalid")
    records: dict[str, Mapping[str, Any]] = {}
    for record in files:
        if not isinstance(record, Mapping):
            raise CheckpointError("checkpoint file record is not an object")
        relative = _safe_payload_path(record.get("path"))
        if relative in records:
            raise CheckpointError("checkpoint file list contains a duplicate")
        records[relative] = record
    if set(records) != set(PAYLOAD_FILES):
        raise CheckpointError("checkpoint payload file set changed")
    actual_files = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file()
    }
    expected_files = set(PAYLOAD_FILES) | {CHECKPOINT_MANIFEST}
    if actual_files != expected_files:
        raise CheckpointError("checkpoint directory contains missing or extra files")
    for relative, record in records.items():
        payload = path / relative
        if _is_link_or_reparse(payload):
            raise CheckpointError("checkpoint payload must not be linked")
        if payload.stat().st_size != record.get("bytes"):
            raise CheckpointError(f"checkpoint byte count mismatch: {relative}")
        if sha256_file(payload) != record.get("sha256"):
            raise CheckpointError(f"checkpoint SHA-256 mismatch: {relative}")
    trainer_state = _torch_load(path / "trainer-state.pt")
    if not isinstance(trainer_state, Mapping):
        raise CheckpointError("checkpoint trainer state is not a mapping")
    trainer_state = _validate_trainer_state(trainer_state)
    if any(trainer_state[field] != manifest["summary"][field] for field in manifest["summary"]):
        raise CheckpointError("checkpoint manifest summary differs from trainer state")
    return dict(manifest)


def load_checkpoint(
    path: Path,
    *,
    model: object,
    optimizer: object,
    scheduler: object,
    scaler: object,
    sampler: object,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every byte before mutating the live training state."""

    validate_checkpoint(path, expected_identity=expected_identity)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = "cpu"
    model.load_state_dict(_torch_load(path / "model.pt", map_location=device), strict=True)
    _restore_gradients(
        model, _torch_load(path / "gradients.pt", map_location="cpu")
    )
    optimizer.load_state_dict(
        _torch_load(path / "optimizer.pt", map_location=device)
    )
    scheduler.load_state_dict(_torch_load(path / "scheduler.pt", map_location="cpu"))
    scaler.load_state_dict(_torch_load(path / "scaler.pt", map_location="cpu"))
    trainer_state = _torch_load(path / "trainer-state.pt", map_location="cpu")
    sampler.load_state_dict(trainer_state["sampler_state"])
    restore_rng_state(_torch_load(path / "rng.pt", map_location="cpu"))
    result = dict(trainer_state)
    result.pop("sampler_state")
    return result


def retention_candidates(
    checkpoint_root: Path,
    *,
    expected_identity: Mapping[str, Any],
    keep_last: int = RETENTION_KEEP_LAST,
) -> list[Path]:
    if keep_last < 1:
        raise CheckpointError("retention keep_last must be at least one")
    complete = []
    for path in sorted(checkpoint_root.glob("step-*")):
        validate_checkpoint(path, expected_identity=expected_identity)
        complete.append(path)
    return complete[:-keep_last]


def prune_after_validated_publish(
    checkpoint_root: Path,
    *,
    newest_checkpoint: Path,
    expected_identity: Mapping[str, Any],
    keep_last: int = RETENTION_KEEP_LAST,
) -> list[Path]:
    """Delete old checkpoints only after the newest complete checkpoint revalidates."""

    validate_checkpoint(newest_checkpoint, expected_identity=expected_identity)
    removed = retention_candidates(
        checkpoint_root, expected_identity=expected_identity, keep_last=keep_last
    )
    for path in removed:
        resolved = path.resolve()
        if resolved.parent != checkpoint_root.resolve():
            raise CheckpointError("retention candidate escaped the checkpoint root")
        shutil.rmtree(resolved)
    return removed
