from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_training_contract import directed_routes, load_student_config  # noqa: E402
from mvp_m1 import (  # noqa: E402
    M1AcceptanceError,
    generate_fixture,
    load_m1_acceptance,
    normalize_generation,
)
from mvp_student import load_frozen_tokenizer  # noqa: E402


ACCEPTANCE = ROOT / "configs" / "mvp_m1_acceptance.yaml"


@pytest.fixture(scope="module")
def tokenizer() -> object:
    student = load_student_config(ROOT / "configs" / "mvp_e8_d2_v48k.yaml")
    value, _ = load_frozen_tokenizer(student, ROOT)
    return value


def records() -> list[dict]:
    texts = {
        "eng_Latn": "turn on the kitchen lights",
        "zho_Hans": "打开厨房的灯",
        "zho_Hant": "打開廚房的燈",
        "jpn_Jpan": "キッチンの照明をつけて",
        "kor_Hang": "부엌 조명을 켜 줘",
    }
    return [
        {
            "sample_id": f"sample:{index}",
            "sample_group_id": f"group:{index}",
            "source_text": texts[source],
            "target_text": texts[target],
            "src_lang": source,
            "tgt_lang": target,
            "split": "train",
        }
        for index, (source, target) in enumerate(directed_routes())
    ]


def test_m1_contract_is_hash_bound_and_thresholds_are_frozen(tmp_path: Path) -> None:
    config = load_m1_acceptance(ACCEPTANCE, ROOT)
    assert config["resume"] == {
        "interruption_optimizer_step": 150,
        "final_optimizer_step": 300,
        "comparison": "exact",
    }
    assert config["thresholds"]["final_eval_loss_to_initial_ratio_max"] == 0.10
    changed = copy.deepcopy(config)
    changed["thresholds"]["normalized_exact_match_routes"] = 19
    path = tmp_path / "changed.yaml"
    import yaml

    path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    with pytest.raises(M1AcceptanceError, match="thresholds changed"):
        load_m1_acceptance(path, ROOT)


def test_generation_normalization_is_nfc_strip_and_whitespace() -> None:
    assert normalize_generation("  e\u0301\n\ttext  ") == "é text"


def test_generation_regression_can_replay_all_twenty_targets(tokenizer: object) -> None:
    fixture = records()
    queued = []
    original = tokenizer.src_lang
    try:
        for record in fixture:
            tokenizer.src_lang = record["tgt_lang"]
            target_ids = tokenizer(record["target_text"])["input_ids"]
            queued.append(torch.tensor([[tokenizer.eos_token_id, *target_ids]], dtype=torch.long))
    finally:
        tokenizer.src_lang = original

    class ReplayModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))

        def generate(self, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            return queued.pop(0).to(self.anchor.device)

    generation = load_m1_acceptance(ACCEPTANCE, ROOT)["generation"]
    outputs = generate_fixture(ReplayModel(), tokenizer, fixture, generation)
    assert len(outputs) == 20
    assert all(row["normalized_exact_match"] for row in outputs)
    assert all(row["target_language_control"] for row in outputs)
    assert not any(row["empty_output"] for row in outputs)
    assert not any(row["cross_language_source_copy"] for row in outputs)
