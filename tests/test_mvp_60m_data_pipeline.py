from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mvp_60m_data_pipeline import (  # noqa: E402
    TextCandidate,
    ParallelGroup,
    apply_domain_ceilings,
    near_identity,
    normalized_identity,
    quality_decision,
    written_cantonese,
    anchor_rows,
    select_group_role,
    select_unpc_side,
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


def test_group_role_selection_is_disjoint_and_anchor_expansion_is_directed() -> None:
    groups = [
        ParallelGroup("fixture", str(index), {"eng_Latn": f"English sentence number {index}.", "jpn_Jpan": f"これは十分に長い日本語の文です。番号{index}。"})
        for index in range(5)
    ]
    used_groups: set[tuple[str, str]] = set()
    used_exact: set[str] = set()
    used_near: set[str] = set()
    first = select_group_role(
        groups, count=2, seed="first", languages=("eng_Latn", "jpn_Jpan"),
        used_groups=used_groups, used_exact=used_exact, used_near=used_near,
    )
    second = select_group_role(
        groups, count=3, seed="second", languages=("eng_Latn",),
        used_groups=used_groups, used_exact=used_exact, used_near=used_near,
    )
    assert len(first) == 2 and len(second) == 3
    assert not ({group.source_group_id for group in first} & {group.source_group_id for group in second})
    rows = anchor_rows(first, ("eng_Latn", "jpn_Jpan"))
    assert len(rows) == 4
    assert {f"{row['src_lang']}->{row['tgt_lang']}" for row in rows} == {"eng_Latn->jpn_Jpan", "jpn_Jpan->eng_Latn"}


class _LengthTokenizer:
    src_lang = "zho_Hans"

    def __call__(self, texts, **_kwargs):
        return {"input_ids": [[1] * (len(text.split()) + 2) for text in texts]}


def test_unpc_side_selector_supports_english_and_excludes_aligned_hans_groups(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unpc.en"
    path.write_text(
        "\n".join(
            f"This is a sufficiently long English sentence number {index} for deterministic selection."
            for index in range(8)
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer = _LengthTokenizer()
    selected = select_unpc_side(
        path,
        tokenizer=tokenizer,
        language_tag="eng_Latn",
        count=3,
        seed="fixture",
        used_exact=set(),
        used_near=set(),
        contamination_exact=set(),
        contamination_near=set(),
        excluded_record_ids={"0", "1", "2", "3", "4"},
        candidate_capacity=8,
    )
    assert len(selected) == 3
    assert all(row["source_record_id"] not in {"0", "1", "2", "3", "4"} for row in selected)
    assert all(row["language_tag"] == "eng_Latn" for row in selected)
    assert tokenizer.src_lang == "zho_Hans"
