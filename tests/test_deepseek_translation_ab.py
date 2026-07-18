from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from deepseek_translation_ab import (  # noqa: E402
    TranslationABError,
    estimate_cost,
    load_ab_config,
    make_translation_batches,
    validate_translations,
)


def _config() -> dict:
    return copy.deepcopy(load_ab_config(ROOT / "configs/deepseek_translation_ab.yaml"))


def _item(index: int) -> dict[str, str]:
    return {
        "id": f"item-{index}",
        "kind": "teacher",
        "route": "eng_Latn->zho_Hans",
        "source_language": "eng_Latn",
        "target_language": "zho_Hans",
        "source_text": f"Source {index}",
        "candidate_translation": f"旧译文 {index}",
        "semantic_group_id": f"group-{index}",
        "source_id": "fixture",
    }


def test_direct_translation_batches_ignore_hymt_candidate_and_are_bounded() -> None:
    config = _config()
    config["api"]["batch_max_records"] = 2
    items = [_item(index) for index in range(3)]
    first = make_translation_batches(items, config)
    assert [len(batch["items"]) for batch in first] == [2, 1]
    changed = [dict(item) for item in items]
    changed[0]["candidate_translation"] = "changed old candidate"
    second = make_translation_batches(changed, config)
    assert [batch["identity"] for batch in first] == [
        batch["identity"] for batch in second
    ]
    estimate = estimate_cost(first, config)
    assert estimate["records"] == 3
    assert 0 < estimate["estimated_total_usd"] < 0.01


def test_translation_response_requires_exact_ids_fields_and_nonempty_text() -> None:
    items = [_item(0), _item(1)]
    valid = {
        "items": [
            {"id": "item-0", "translation": "译文零"},
            {"id": "item-1", "translation": "译文一"},
        ]
    }
    assert validate_translations(valid, items)[0]["translation"] == "译文零"
    reordered = {"items": list(reversed(valid["items"]))}
    with pytest.raises(TranslationABError, match="IDs/order"):
        validate_translations(reordered, items)
    extra = copy.deepcopy(valid)
    extra["items"][0]["note"] = "not allowed"
    with pytest.raises(TranslationABError, match="fields"):
        validate_translations(extra, items)
    empty = copy.deepcopy(valid)
    empty["items"][0]["translation"] = "  "
    with pytest.raises(TranslationABError, match="empty/oversized"):
        validate_translations(empty, items)


def test_ab_config_prohibits_formal_evaluation_and_thinking_translation() -> None:
    config_path = ROOT / "configs/deepseek_translation_ab.yaml"
    config = _config()
    assert config["inputs"]["formal_test_access"] == "prohibited"
    assert config["inputs"]["formal_devtest_access"] == "prohibited"
    assert config["api"]["thinking"] == "disabled"
    assert load_ab_config(config_path)["ab"]["records"] == 512
