from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mvp_60m_data_pipeline import (  # noqa: E402
    TextCandidate,
    apply_domain_ceilings,
    near_identity,
    normalized_identity,
    quality_decision,
    written_cantonese,
)


def candidate(text: str, *, language: str = "zho_Hant") -> TextCandidate:
    return TextCandidate("fixture", "1", language, "daily_and_dialogue", text)


def test_quality_gate_keeps_standard_hant_and_rejects_cantonese_and_overflow() -> None:
    accepted = quality_decision(
        candidate("這是一段來自臺灣的標準繁體中文，用來驗證資料品質。"), token_count=18
    )
    assert accepted.accepted
    assert accepted.reason is None

    cantonese = "佢而家喺邊度做緊乜嘢？我真係唔知道，亦都冇人同我講過。"
    assert written_cantonese(cantonese)
    assert quality_decision(candidate(cantonese), token_count=15).reason == "written_cantonese"
    overflow = "這是一段足夠長的繁體中文內容，用來確認超過模型長度時一定拒絕而不截斷。"
    assert quality_decision(candidate(overflow), token_count=300).reason == "token_overflow"


def test_normalized_identity_is_nfc_whitespace_and_casefold_stable() -> None:
    assert normalized_identity("Ａ  B") != normalized_identity("A b")
    assert normalized_identity("Cafe\u0301   Test") == normalized_identity("Café test")
    assert near_identity("測試：版本一。") == near_identity("測試版本一")


def test_domain_ceilings_are_exact_and_never_refill_base() -> None:
    rows = []
    for index in range(65):
        rows.append({"domain": "general", "selection_rank": f"b{index:03}", "source_record_id": str(index)})
    for index in range(100):
        rows.append({"domain": "technical", "selection_rank": f"t{index:03}", "source_record_id": str(index)})
        rows.append({"domain": "legal_and_government", "selection_rank": f"l{index:03}", "source_record_id": str(index)})
    selected = apply_domain_ceilings(rows)
    technical = sum(row["domain"] == "technical" for row in selected)
    legal = sum(row["domain"] == "legal_and_government" for row in selected)
    assert technical / len(selected) <= 0.15
    assert legal / len(selected) <= 0.20
    assert sum(row["domain"] == "general" for row in selected) == 65
