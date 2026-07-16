from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_training_contract import config_sha256, directed_routes, load_student_config  # noqa: E402
from mvp_student import EncodingPolicy, construct_m2m100, load_frozen_tokenizer  # noqa: E402
from mvp_training import (  # noqa: E402
    BatchEncoder,
    DeterministicRouteSampler,
    JsonlRunLogger,
    RouteDataset,
    TrainingContractError,
    execute_training,
    load_route_dataset,
    load_training_config,
    validate_resource_budget,
    validate_training_config,
    semantic_trace_sha256,
)


TRAINING_CONFIG = ROOT / "configs" / "mvp_training_td10_smoke.yaml"
STUDENT_CONFIG = ROOT / "configs" / "mvp_e8_d2_v48k.yaml"


@pytest.fixture(scope="module")
def frozen() -> tuple[dict, object, Path]:
    config = load_student_config(STUDENT_CONFIG)
    tokenizer, _ = load_frozen_tokenizer(config, ROOT)
    return config, tokenizer, ROOT / config["tokenizer"]["path"]


def route_records(per_route: int = 2) -> list[dict]:
    texts = {
        "eng_Latn": "turn on the kitchen lights",
        "zho_Hans": "打开厨房的灯",
        "zho_Hant": "打開廚房的燈",
        "jpn_Jpan": "キッチンの照明をつけて",
        "kor_Hang": "부엌 조명을 켜 줘",
    }
    records = []
    for route_index, (source, target) in enumerate(directed_routes()):
        for sample_index in range(per_route):
            records.append(
                {
                    "sample_id": f"sample:{route_index}:{sample_index}",
                    "sample_group_id": f"group:{route_index}:{sample_index}",
                    "source_text": texts[source],
                    "target_text": texts[target],
                    "src_lang": source,
                    "tgt_lang": target,
                    "split": "train",
                }
            )
    return records


def route_dataset(split: str = "train", per_route: int = 2) -> RouteDataset:
    grouped = {}
    records = route_records(per_route)
    for source, target in directed_routes():
        route = f"{source}->{target}"
        selected = []
        for record in records:
            if f"{record['src_lang']}->{record['tgt_lang']}" == route:
                value = dict(record)
                value["split"] = split
                value["sample_id"] = f"{split}:{value['sample_id']}"
                selected.append(value)
        grouped[route] = tuple(selected)
    identity = {
        route: [record["sample_id"] for record in grouped[route]]
        for route in grouped
    }
    return RouteDataset(
        split=split,
        records_by_route=grouped,
        file_sha256="0" * 64,
        selection_sha256=config_sha256(identity),
    )


def tiny_model(tokenizer: object) -> object:
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
        "dropout": 0.0,
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
    return construct_m2m100(tokenizer, values, 1234)


def smoke_config() -> dict:
    config = load_training_config(TRAINING_CONFIG)
    config = copy.deepcopy(config)
    config["identity"]["seed"] = 1234
    config["resource_profile"].update(
        {
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "max_source_length": 32,
            "max_target_length": 32,
            "dataloader_workers": 0,
        }
    )
    config["optimization"].update(
        {
            "max_optimizer_steps": 2,
            "max_train_tokens": 10000,
            "validation_frequency": 1,
            "validation_batches": 1,
            "checkpoint_frequency": 1,
        }
    )
    return validate_training_config(config)


def test_training_config_is_strict_and_has_no_test_input() -> None:
    config = load_training_config(TRAINING_CONFIG)
    assert set(config["data"]) == {
        "train_path",
        "train_sha256",
        "dev_path",
        "dev_sha256",
        "manifest_path",
        "manifest_sha256",
        "train_max_records_per_route",
        "dev_max_records_per_route",
        "route_weights",
    }
    assert len(config["data"]["route_weights"]) == 20
    changed = copy.deepcopy(config)
    changed["data"]["test_path"] = "data/model/test.jsonl"
    with pytest.raises(TrainingContractError, match="unknown fields: test_path"):
        validate_training_config(changed)


def test_resource_budget_uses_all_three_device_limits() -> None:
    profile = load_training_config(TRAINING_CONFIG)["resource_profile"]
    runtime = {
        "device_total_bytes": 32 * 1024 * 1024 * 1024,
        "host_available_bytes": 64 * 1024 * 1024 * 1024,
    }
    result = validate_resource_budget(profile, runtime)
    assert result["effective_device_limit_bytes"] == 8192 * 1024 * 1024

    insufficient = dict(profile)
    insufficient["device_memory_budget_mib"] = 30 * 1024
    with pytest.raises(TrainingContractError, match="cannot satisfy"):
        validate_resource_budget(insufficient, runtime)

    switched = dict(profile)
    switched["device_memory_budget_mib"] = 4096
    assert validate_resource_budget(switched, runtime)["effective_device_limit_bytes"] == (
        4096 * 1024 * 1024
    )


def test_sampler_sequence_and_resume_are_exact() -> None:
    dataset = route_dataset(per_route=3)
    weights = load_training_config(TRAINING_CONFIG)["data"]["route_weights"]
    first = DeterministicRouteSampler(dataset, weights, seed=55)
    prefix = [first.next_sample() for _ in range(17)]
    state = first.state_dict()
    expected = [first.next_sample() for _ in range(50)]

    resumed = DeterministicRouteSampler(dataset, weights, seed=999)
    resumed.load_state_dict(state)
    actual = [resumed.next_sample() for _ in range(50)]
    key = lambda row: (row.route, row.route_epoch, row.route_position, row.record["sample_id"])
    assert [key(row) for row in actual] == [key(row) for row in expected]
    assert len({row.route for row in prefix + expected[:3]}) == 20


def test_parallel_batch_encoding_preserves_order(
    frozen: tuple[dict, object, Path]
) -> None:
    _, tokenizer, tokenizer_path = frozen
    records = route_records(per_route=1)[:8]
    policy = EncodingPolicy(max_source_length=32, max_target_length=32)
    with BatchEncoder(
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        policy=policy,
        workers=0,
    ) as serial:
        expected = serial(records)
    with BatchEncoder(
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        policy=policy,
        workers=2,
    ) as parallel:
        actual = parallel(records)
    assert actual["sample_ids"] == expected["sample_ids"]
    assert actual["routes"] == expected["routes"]
    assert torch.equal(actual["input_ids"], expected["input_ids"])
    assert torch.equal(actual["labels"], expected["labels"])


def test_tiny_training_covers_accumulation_logging_and_dev(
    frozen: tuple[dict, object, Path]
) -> None:
    _, tokenizer, tokenizer_path = frozen
    logger = JsonlRunLogger()
    result = execute_training(
        model=tiny_model(tokenizer),
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        train_dataset=route_dataset("train"),
        dev_dataset=route_dataset("dev"),
        config=smoke_config(),
        logger=logger,
    )

    assert result["optimizer_steps"] == 2
    assert result["micro_steps"] == 4
    assert result["consumed_samples"] == 8
    assert result["consumed_tokens"] > 0
    assert result["exception_skips"] == 0
    assert all(math_value == math_value for math_value in (result["mean_train_loss"], result["final_train_loss"]))
    assert [event["accumulation_phase"] for event in logger.events if event["event"] == "micro_step"] == [0, 1, 0, 1]
    assert len([event for event in logger.events if event["event"] == "validation"]) == 2
    assert all("sample_ids" in event for event in logger.events if event["event"] == "optimizer_step")

    replay_logger = JsonlRunLogger()
    replay = execute_training(
        model=tiny_model(tokenizer),
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        train_dataset=route_dataset("train"),
        dev_dataset=route_dataset("dev"),
        config=smoke_config(),
        logger=replay_logger,
    )
    assert replay["mean_train_loss"] == result["mean_train_loss"]
    assert replay["final_train_loss"] == result["final_train_loss"]
    assert [
        (event.get("loss"), event.get("sample_ids")) for event in replay_logger.events
    ] == [(event.get("loss"), event.get("sample_ids")) for event in logger.events]
    assert semantic_trace_sha256(replay_logger.events) == semantic_trace_sha256(logger.events)


def test_nonfinite_loss_fails_explicitly(frozen: tuple[dict, object, Path]) -> None:
    _, tokenizer, tokenizer_path = frozen

    class BadModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.config = SimpleNamespace(use_cache=True)

        def forward(self, input_ids, attention_mask, labels):  # type: ignore[no-untyped-def]
            del input_ids, attention_mask
            loss = self.weight * torch.tensor(float("nan"), device=self.weight.device)
            logits = torch.zeros(
                (*labels.shape, len(tokenizer)), device=self.weight.device
            )
            return SimpleNamespace(loss=loss, logits=logits)

    with pytest.raises(TrainingContractError, match="NaN/Inf loss"):
        execute_training(
            model=BadModel(),
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
            train_dataset=route_dataset("train"),
            dev_dataset=route_dataset("dev"),
            config=smoke_config(),
            logger=JsonlRunLogger(),
        )


def test_route_dataset_rejects_empty_route_and_wrong_hash(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    record = route_records(per_route=1)[0]
    path.write_text(
        __import__("json").dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(TrainingContractError, match="empty for routes"):
        load_route_dataset(
            path,
            expected_sha256=digest,
            split="train",
            max_records_per_route=1,
        )
    with pytest.raises(TrainingContractError, match="SHA-256 changed"):
        load_route_dataset(
            path,
            expected_sha256="0" * 64,
            split="train",
            max_records_per_route=1,
        )
