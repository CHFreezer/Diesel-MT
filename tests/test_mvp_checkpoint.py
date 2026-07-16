from __future__ import annotations

import copy
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import mvp_checkpoint  # noqa: E402
from mvp_checkpoint import (  # noqa: E402
    CHECKPOINT_MANIFEST,
    CheckpointError,
    load_checkpoint,
    prune_after_validated_publish,
    retention_candidates,
    save_checkpoint,
    validate_checkpoint,
)
from model_training_contract import config_sha256, directed_routes, load_student_config  # noqa: E402
from mvp_student import construct_m2m100, load_frozen_tokenizer, state_dict_sha256  # noqa: E402
from mvp_training import (  # noqa: E402
    JsonlRunLogger,
    RouteDataset,
    execute_training,
    load_training_config,
    semantic_trace_sha256,
    validate_training_config,
)


IDENTITY = {
    "training_config_sha256": "1" * 64,
    "student_config_sha256": "2" * 64,
    "tokenizer_manifest_sha256": "3" * 64,
    "data": {"train": "4" * 64, "dev": "5" * 64},
    "code_sha256": "6" * 64,
    "dependencies_sha256": "7" * 64,
    "git": {"commit": "abc", "dirty": True},
    "runtime": {"device": "cpu", "precision": "fp32"},
}


class DummySampler:
    def __init__(self) -> None:
        self.value = {"position": 7, "order": [2, 0, 1]}

    def state_dict(self) -> dict:
        return dict(self.value)

    def load_state_dict(self, state: dict) -> None:
        self.value = dict(state)


def objects() -> tuple[object, object, object, object, DummySampler]:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 2)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 - step / 10)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sampler = DummySampler()
    loss = model(torch.ones(2, 4)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler, scaler, sampler


def trainer_state(step: int = 1) -> dict:
    return {
        "global_step": step,
        "micro_step": step * 2,
        "epoch": 0,
        "consumed_samples": step * 4,
        "consumed_tokens": step * 20,
        "accumulation_phase": 0,
        "loss_history": [1.0 / value for value in range(1, step + 1)],
        "route_counts": {"eng_Latn->jpn_Jpan": step * 4},
        "token_audit": {"source_original_tokens": step * 10},
    }


def test_checkpoint_roundtrip_restores_all_state_and_rng(tmp_path: Path) -> None:
    random.seed(77)
    np.random.seed(77)
    torch.manual_seed(77)
    model, optimizer, scheduler, scaler, sampler = objects()
    expected_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_gradients = {
        name: parameter.grad.detach().clone() if parameter.grad is not None else None
        for name, parameter in model.named_parameters()
    }
    checkpoint = save_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
        trainer_state=trainer_state(),
        identity=IDENTITY,
    )
    expected_random = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100)
            parameter.grad = None
    sampler.value = {"position": 999}
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    restored = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
        expected_identity=IDENTITY,
    )

    assert restored["global_step"] == 1
    assert sampler.value == {"position": 7, "order": [2, 0, 1]}
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected_weights.items())
    for name, parameter in model.named_parameters():
        expected = expected_gradients[name]
        assert (parameter.grad is None) == (expected is None)
        if expected is not None:
            assert torch.equal(parameter.grad, expected)
    assert random.random() == expected_random
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)


@pytest.mark.parametrize(
    "fault_point",
    ["after_model", "after_optimizer", "before_manifest", "after_manifest_before_publish"],
)
def test_fault_injection_never_publishes_partial_or_damages_old(
    tmp_path: Path, fault_point: str
) -> None:
    model, optimizer, scheduler, scaler, sampler = objects()
    old = save_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
        trainer_state=trainer_state(1),
        identity=IDENTITY,
    )
    old_manifest_hash = __import__("hashlib").sha256(
        (old / CHECKPOINT_MANIFEST).read_bytes()
    ).hexdigest()

    def fail(point: str) -> None:
        if point == fault_point:
            raise RuntimeError(f"injected {point}")

    with pytest.raises(RuntimeError, match="injected"):
        save_checkpoint(
            tmp_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sampler=sampler,
            trainer_state=trainer_state(2),
            identity=IDENTITY,
            fault_injector=fail,
        )

    validate_checkpoint(old, expected_identity=IDENTITY)
    assert __import__("hashlib").sha256(
        (old / CHECKPOINT_MANIFEST).read_bytes()
    ).hexdigest() == old_manifest_hash
    assert not (tmp_path / "step-00000002").exists()
    assert not list(tmp_path.glob(".step-00000002.staging-*"))


def test_corrupt_incomplete_extra_and_identity_mismatch_are_rejected(tmp_path: Path) -> None:
    model, optimizer, scheduler, scaler, sampler = objects()
    original = save_checkpoint(
        tmp_path / "source",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
        trainer_state=trainer_state(),
        identity=IDENTITY,
    )

    corrupt = tmp_path / "corrupt" / original.name
    shutil.copytree(original, corrupt)
    with (corrupt / "model.pt").open("ab") as handle:
        handle.write(b"damage")
    with pytest.raises(CheckpointError, match="byte count mismatch"):
        validate_checkpoint(corrupt, expected_identity=IDENTITY)

    incomplete = tmp_path / "incomplete" / original.name
    shutil.copytree(original, incomplete)
    manifest_path = incomplete / CHECKPOINT_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "staging"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CheckpointError, match="not complete"):
        validate_checkpoint(incomplete, expected_identity=IDENTITY)

    extra = tmp_path / "extra" / original.name
    shutil.copytree(original, extra)
    (extra / "unexpected.bin").write_bytes(b"extra")
    with pytest.raises(CheckpointError, match="missing or extra"):
        validate_checkpoint(extra, expected_identity=IDENTITY)

    with pytest.raises(CheckpointError, match="does not match"):
        validate_checkpoint(original, expected_identity={**IDENTITY, "changed": True})


def test_manifest_path_traversal_and_links_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, optimizer, scheduler, scaler, sampler = objects()
    original = save_checkpoint(
        tmp_path / "source",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
        trainer_state=trainer_state(),
        identity=IDENTITY,
    )
    traversal = tmp_path / "traversal" / original.name
    shutil.copytree(original, traversal)
    manifest_path = traversal / CHECKPOINT_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "../model.pt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CheckpointError, match="path traversal"):
        validate_checkpoint(traversal, expected_identity=IDENTITY)

    linked = tmp_path / "linked" / original.name
    shutil.copytree(original, linked)
    real_is_link_or_reparse = mvp_checkpoint._is_link_or_reparse
    with monkeypatch.context() as patch:
        patch.setattr(
            mvp_checkpoint,
            "_is_link_or_reparse",
            lambda path: path.name == "model.pt" or real_is_link_or_reparse(path),
        )
        with pytest.raises(CheckpointError, match="payload must not be linked"):
            validate_checkpoint(linked, expected_identity=IDENTITY)

    directory_link = tmp_path / "directory-link" / original.name
    directory_link.parent.mkdir(parents=True)
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/d", "/c", "mklink", "/J", str(directory_link), str(original)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
        directory_link.symlink_to(original, target_is_directory=True)
    try:
        assert real_is_link_or_reparse(directory_link)
        with pytest.raises(CheckpointError, match="real directory"):
            validate_checkpoint(directory_link, expected_identity=IDENTITY)
    finally:
        if os.name == "nt":
            os.rmdir(directory_link)
        else:
            directory_link.unlink()


def test_retention_only_prunes_after_newest_validation(tmp_path: Path) -> None:
    model, optimizer, scheduler, scaler, sampler = objects()
    checkpoints = []
    for step in range(1, 5):
        checkpoints.append(
            save_checkpoint(
                tmp_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                sampler=sampler,
                trainer_state=trainer_state(step),
                identity=IDENTITY,
            )
        )
    assert retention_candidates(
        tmp_path, expected_identity=IDENTITY, keep_last=2
    ) == checkpoints[:2]
    removed = prune_after_validated_publish(
        tmp_path,
        newest_checkpoint=checkpoints[-1],
        expected_identity=IDENTITY,
        keep_last=2,
    )
    assert removed == checkpoints[:2]
    assert not checkpoints[0].exists() and not checkpoints[1].exists()
    assert checkpoints[2].exists() and checkpoints[3].exists()


def _training_dataset(split: str) -> RouteDataset:
    texts = {
        "eng_Latn": "turn on the kitchen lights",
        "zho_Hans": "打开厨房的灯",
        "zho_Hant": "打開廚房的燈",
        "jpn_Jpan": "キッチンの照明をつけて",
        "kor_Hang": "부엌 조명을 켜 줘",
    }
    grouped = {}
    for route_index, (source, target) in enumerate(directed_routes()):
        route = f"{source}->{target}"
        grouped[route] = tuple(
            {
                "sample_id": f"{split}:{route_index}:{sample_index}",
                "sample_group_id": f"group:{route_index}:{sample_index}",
                "source_text": texts[source],
                "target_text": texts[target],
                "src_lang": source,
                "tgt_lang": target,
                "split": split,
            }
            for sample_index in range(2)
        )
    selection = {
        route: [record["sample_id"] for record in records]
        for route, records in grouped.items()
    }
    return RouteDataset(
        split=split,
        records_by_route=grouped,
        file_sha256="0" * 64,
        selection_sha256=config_sha256(selection),
    )


def _training_config() -> dict:
    config = copy.deepcopy(
        load_training_config(ROOT / "configs" / "mvp_training_td10_smoke.yaml")
    )
    config["identity"]["seed"] = 8822
    config["resource_profile"].update(
        {
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "max_source_length": 32,
            "max_target_length": 32,
        }
    )
    config["optimization"].update(
        {
            "max_optimizer_steps": 4,
            "max_train_tokens": 20000,
            "checkpoint_frequency": 2,
            "validation_frequency": 1,
            "validation_batches": 1,
        }
    )
    return validate_training_config(config)


def _training_model(tokenizer: object) -> object:
    values = {
        "vocab_size": len(tokenizer),
        "d_model": 32,
        "encoder_ffn_dim": 64,
        "decoder_ffn_dim": 64,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "encoder_attention_heads": 4,
        "decoder_attention_heads": 4,
        "max_position_embeddings": 64,
        "activation_function": "relu",
        "dropout": 0.1,
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "encoder_layerdrop": 0.0,
        "decoder_layerdrop": 0.0,
        "scale_embedding": True,
        "tie_word_embeddings": True,
        "use_cache": True,
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "decoder_start_token_id": 2,
        "forced_eos_token_id": None,
    }
    return construct_m2m100(tokenizer, values, 8822)


def test_uninterrupted_and_resumed_training_are_exact(tmp_path: Path) -> None:
    student = load_student_config(ROOT / "configs" / "mvp_e8_d2_v48k.yaml")
    tokenizer, _ = load_frozen_tokenizer(student, ROOT)
    tokenizer_path = ROOT / student["tokenizer"]["path"]
    config = _training_config()
    train = _training_dataset("train")
    dev = _training_dataset("dev")

    baseline_model = _training_model(tokenizer)
    baseline_logger = JsonlRunLogger()
    baseline = execute_training(
        model=baseline_model,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        train_dataset=train,
        dev_dataset=dev,
        config=config,
        logger=baseline_logger,
    )

    interrupted_model = _training_model(tokenizer)
    interrupted_logger = JsonlRunLogger()
    published: list[Path] = []

    def publish(context: dict) -> None:
        published.append(
            save_checkpoint(
                tmp_path / "checkpoints",
                model=context["model"],
                optimizer=context["optimizer"],
                scheduler=context["scheduler"],
                scaler=context["scaler"],
                sampler=context["sampler"],
                trainer_state=context["trainer_state"],
                identity=IDENTITY,
            )
        )

    interrupted = execute_training(
        model=interrupted_model,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        train_dataset=train,
        dev_dataset=dev,
        config=config,
        logger=interrupted_logger,
        checkpoint_callback=publish,
        stop_after_optimizer_steps=2,
    )
    assert interrupted["status"] == "interrupted"
    assert published[-1].name == "step-00000002"

    resumed_model = _training_model(tokenizer)
    resumed_logger = JsonlRunLogger()

    def resume(context: dict) -> dict:
        return load_checkpoint(
            published[-1],
            model=context["model"],
            optimizer=context["optimizer"],
            scheduler=context["scheduler"],
            scaler=context["scaler"],
            sampler=context["sampler"],
            expected_identity=IDENTITY,
        )

    resumed = execute_training(
        model=resumed_model,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        train_dataset=train,
        dev_dataset=dev,
        config=config,
        logger=resumed_logger,
        resume_loader=resume,
    )

    assert resumed["status"] == "complete"
    assert resumed["optimizer_steps"] == baseline["optimizer_steps"] == 4
    assert resumed["micro_steps"] == baseline["micro_steps"] == 8
    assert resumed["final_train_loss"] == baseline["final_train_loss"]
    assert resumed["mean_train_loss"] == baseline["mean_train_loss"]
    assert config_sha256(resumed["sampler_state"]) == config_sha256(
        baseline["sampler_state"]
    )
    assert state_dict_sha256(resumed_model) == state_dict_sha256(baseline_model)
    assert semantic_trace_sha256(
        [*interrupted_logger.events, *resumed_logger.events]
    ) == semantic_trace_sha256(baseline_logger.events)
