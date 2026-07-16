from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_training_contract import directed_routes, load_student_config  # noqa: E402
from mvp_student import (  # noqa: E402
    DirectionAwareCollator,
    EncodingPolicy,
    StudentContractError,
    construct_m2m100,
    encode_parallel_sample,
    load_frozen_tokenizer,
    model_inputs,
    state_dict_sha256,
)


STUDENT_CONFIG = ROOT / "configs" / "mvp_e8_d2_v48k.yaml"


@pytest.fixture(scope="module")
def frozen() -> tuple[dict, object]:
    config = load_student_config(STUDENT_CONFIG)
    tokenizer, _ = load_frozen_tokenizer(config, ROOT)
    return config, tokenizer


def route_records() -> list[dict]:
    texts = {
        "eng_Latn": "turn on the kitchen lights",
        "zho_Hans": "打开厨房的灯",
        "zho_Hant": "打開廚房的燈",
        "jpn_Jpan": "キッチンの照明をつけて",
        "kor_Hang": "부엌 조명을 켜 줘",
    }
    return [
        {
            "sample_id": f"fixture:{source}:{target}",
            "sample_group_id": f"group:{index}",
            "source_text": texts[source],
            "target_text": texts[target],
            "src_lang": source,
            "tgt_lang": target,
            "split": "train",
        }
        for index, (source, target) in enumerate(directed_routes())
    ]


def test_frozen_tokenizer_and_twenty_route_collator(frozen: tuple[dict, object]) -> None:
    _, tokenizer = frozen
    policy = EncodingPolicy(max_source_length=32, max_target_length=32)
    batch = DirectionAwareCollator(tokenizer, policy)(route_records())

    assert batch["input_ids"].shape[0] == 20
    assert batch["labels"].shape[0] == 20
    assert set(batch["routes"]) == {
        f"{source}->{target}" for source, target in directed_routes()
    }
    assert set(batch["route_statistics"]) == set(batch["routes"])
    assert (batch["labels"] == -100).any() or len(set(batch["labels"].shape)) > 0
    inputs = model_inputs(batch)
    assert set(inputs) == {"input_ids", "attention_mask", "labels"}
    assert not any(key in inputs for key in ("sample_ids", "routes"))


def test_encoding_preserves_language_and_eos_when_truncated(
    frozen: tuple[dict, object]
) -> None:
    _, tokenizer = frozen
    record = route_records()[0]
    record["source_text"] = "long " * 200
    record["target_text"] = "長い" * 200
    encoded = encode_parallel_sample(
        tokenizer,
        record,
        EncodingPolicy(max_source_length=8, max_target_length=9),
    )

    assert len(encoded.input_ids) == 8
    assert len(encoded.labels) == 9
    assert encoded.input_ids[0] == tokenizer.convert_tokens_to_ids(record["src_lang"])
    assert encoded.labels[0] == tokenizer.convert_tokens_to_ids(record["tgt_lang"])
    assert encoded.input_ids[-1] == tokenizer.eos_token_id
    assert encoded.labels[-1] == tokenizer.eos_token_id
    assert encoded.source_truncated_tokens > 0
    assert encoded.target_truncated_tokens > 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"source_text": "   "}, "source_text"),
        ({"target_text": ""}, "target_text"),
        ({"tgt_lang": "eng_Latn"}, "same-language"),
        ({"tgt_lang": "fra_Latn"}, "unknown language"),
    ],
)
def test_collator_rejects_invalid_samples(
    frozen: tuple[dict, object], change: dict, message: str
) -> None:
    _, tokenizer = frozen
    record = copy.deepcopy(route_records()[0])
    record.update(change)
    with pytest.raises(StudentContractError, match=message):
        DirectionAwareCollator(tokenizer, EncodingPolicy())([record])


def test_encoding_policy_rejects_semantic_drift() -> None:
    with pytest.raises(StudentContractError, match="integer >= 3"):
        EncodingPolicy(max_source_length=2)
    with pytest.raises(StudentContractError, match="position ceiling"):
        EncodingPolicy(max_target_length=1_025)
    with pytest.raises(StudentContractError, match="overflow policy"):
        EncodingPolicy(overflow_policy="drop")
    with pytest.raises(StudentContractError, match="-100"):
        EncodingPolicy(label_pad_id=0)


def test_small_m2m100_constructor_is_deterministic_and_trainable(
    frozen: tuple[dict, object]
) -> None:
    _, tokenizer = frozen
    tiny_values = {
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
    first = construct_m2m100(tokenizer, tiny_values, 123)
    second = construct_m2m100(tokenizer, tiny_values, 123)
    assert state_dict_sha256(first) == state_dict_sha256(second)
    assert first.get_input_embeddings().weight.data_ptr() == (
        first.get_output_embeddings().weight.data_ptr()
    )

    batch = DirectionAwareCollator(
        tokenizer, EncodingPolicy(max_source_length=32, max_target_length=32)
    )(route_records()[:2])
    output = first(**model_inputs(batch))
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in first.parameters())


def test_route_fixture_file_is_the_complete_frozen_matrix() -> None:
    fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "model_data" / "m0-routes.json").read_text(
            encoding="utf-8"
        )
    )
    assert [tuple(route) for route in fixture["routes"]] == list(directed_routes())
