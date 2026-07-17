from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_training_contract import directed_routes  # noqa: E402
from mvp_60m_data_pipeline import AbilityDataError, write_jsonl  # noqa: E402
from review_mvp_60m_teacher import build_queue, load_config, verify_decisions  # noqa: E402


def _accepted(index: int, source: str, target: str) -> dict[str, object]:
    return {
        "record_id": f"accepted-{index}",
        "src_lang": source,
        "tgt_lang": target,
        "source_text": f"source {index}",
        "target_text": f"target {index}",
    }


def test_review_queue_covers_every_route_and_all_available_filtered() -> None:
    config = copy.deepcopy(load_config(ROOT / "configs/mvp_60m_teacher_review.yaml"))
    config["sampling"]["accepted_per_route"] = 2
    config["sampling"]["filtered_per_route"] = 2
    accepted = []
    filtered = []
    for route_index, (source, target) in enumerate(directed_routes()):
        for offset in range(2):
            accepted.append(_accepted(route_index * 10 + offset, source, target))
        filtered.append({
            "job_id": f"filtered-{route_index}", "route": f"{source}->{target}",
            "src_lang": source, "tgt_lang": target,
            "source_text": "source", "normalized_output": "target",
            "rejection_reasons": ["truncated"],
        })
    queue = build_queue(accepted, filtered, config)
    assert len(queue) == 60
    assert sum(row["kind"] == "accepted" for row in queue) == 40
    assert sum(row["kind"] == "filtered" for row in queue) == 20
    assert len({row["review_id"] for row in queue}) == 60


def test_incomplete_manual_decisions_are_rejected(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/mvp_60m_teacher_review.yaml")
    td04 = tmp_path / "td04"
    td04.mkdir()
    write_jsonl(td04 / "manual-review-queue.jsonl", [{"review_id": "one", "route": "eng_Latn->jpn_Jpan", "kind": "accepted"}])
    write_jsonl(td04 / "manual-review-decisions.jsonl", [])
    with pytest.raises(AbilityDataError, match="decisions incomplete"):
        verify_decisions(tmp_path, config)
