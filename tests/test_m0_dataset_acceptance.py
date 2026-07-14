from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from m0_dataset_acceptance import (  # noqa: E402
    M0AcceptanceError,
    _select_accepted,
    compare_builds,
    load_sampling_config,
)
from model_training_contract import (  # noqa: E402
    ContractError,
    config_sha256,
    directed_routes,
    validate_route,
)


SAMPLING_PATH = ROOT / "configs" / "mvp_direction_sampling.yaml"
ROUTE_FIXTURE_PATH = ROOT / "tests" / "fixtures" / "model_data" / "m0-routes.json"
EVIDENCE_PATH = ROOT / "artifacts" / "model-training" / "m0-dataset-acceptance.json"


def test_route_fixture_covers_all_routes_and_rejects_counterexamples() -> None:
    fixture = json.loads(ROUTE_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["schema_version"] == 1
    assert {tuple(route) for route in fixture["routes"]} == set(directed_routes())
    assert len(fixture["routes"]) == 18
    for record in fixture["invalid_routes"]:
        with pytest.raises(ContractError):
            validate_route(*record["route"])


def test_sampling_config_is_uniform_bounded_and_complete() -> None:
    sampling = load_sampling_config(SAMPLING_PATH)
    assert len(sampling["routes"]) == 18
    assert all(record["weight"] == 1.0 for record in sampling["routes"])
    assert all(record["maximum_repeats_per_epoch"] == 1 for record in sampling["routes"])
    assert sampling["strategy"]["low_resource_oversampling"] == "prohibited"
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert evidence["status"] == "complete"
    assert evidence["scope"]["directed_routes"] == 18
    assert evidence["scope"]["teacher_synthetic"] == 0
    assert evidence["identities"]["direction_sampling_config_sha256"] == config_sha256(sampling)
    assert evidence["reproducibility"]["byte_identical"] is True


def _sample(index: int, source: str, target: str) -> dict[str, str]:
    return {
        "sample_id": f"sample-{index}",
        "src_lang": "eng_Latn",
        "tgt_lang": "zho_Hant",
        "source_text": source,
        "target_text": target,
        "split": "train",
    }


def test_review_selection_is_stable_and_covers_boundaries() -> None:
    records = [
        _sample(1, "a", "短"),
        _sample(2, "a much longer source sentence", "很長的繁體中文句子"),
        _sample(3, "ratio boundary has many characters", "短"),
        _sample(4, "mixed script", "播放 spotify 音樂"),
        _sample(5, "ordinary source", "普通繁體句子"),
    ]
    first = _select_accepted(records, "eng_Latn--zho_Hant", 5)
    second = _select_accepted(list(reversed(records)), "eng_Latn--zho_Hant", 5)
    assert first == second
    assert {record["selection_tag"] for record in first} >= {
        "short-boundary",
        "long-boundary",
        "ratio-boundary",
        "traditional-mixed-script",
    }


def test_reproducibility_comparison_reports_and_blocks_differences(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    from m0_dataset_acceptance import REPRO_PATHS

    for relative in REPRO_PATHS:
        (first / relative).parent.mkdir(parents=True, exist_ok=True)
        (second / relative).parent.mkdir(parents=True, exist_ok=True)
        (first / relative).write_bytes(relative.encode())
        (second / relative).write_bytes(relative.encode())
    report = compare_builds(first, second)
    assert report["identical"] is True
    assert all(record["identical"] for record in report["files"])

    (second / REPRO_PATHS[0]).write_bytes(b"different")
    with pytest.raises(M0AcceptanceError, match="independent builds differ"):
        compare_builds(first, second)
