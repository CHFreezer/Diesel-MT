from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hymt2_distillation import (  # noqa: E402
    ROUTES,
    DistillationError,
    deterministic_route_sample,
    load_prompt_config,
)
from hymt2_distillation_data import (  # noqa: E402
    _load_route_state,
    _load_route_checkpoints,
    _write_sample_checkpoint,
    _write_route_state,
    generation_contract,
    load_distillation_config,
    prepare_review_queue,
    resolve_work_root,
    validate_manual_attestation,
    validate_distillation_config,
)
from model_training_contract import directed_routes  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "hymt2_distillation.yaml"
D1_CONFIG_PATH = ROOT / "configs" / "hymt2_distillation_d1.yaml"
PROMPT_PATH = ROOT / "configs" / "hymt2_teacher_prompt_decode.yaml"


@pytest.fixture
def distillation_config() -> dict[str, object]:
    return load_distillation_config(CONFIG_PATH)


@pytest.fixture
def prompt_config() -> dict[str, object]:
    return load_prompt_config(PROMPT_PATH)


def _sample(source: str, target: str, index: int) -> dict[str, str]:
    return {
        "sample_id": f"sample-{source}-{target}-{index}",
        "sample_group_id": f"group-{source}-{target}-{index}",
        "source_id": "fixture",
        "source_version": "1",
        "license": "CC-BY-4.0",
        "source_text": f"source {index}",
        "target_text": f"target {index}",
        "src_lang": source,
        "tgt_lang": target,
        "split": "train",
    }


def test_distillation_contract_is_train_only_bounded_and_complete(
    distillation_config: dict[str, object],
) -> None:
    assert distillation_config["sampling"]["records_per_route"] == 128
    assert distillation_config["sampling"]["total_records"] == 2_304
    assert distillation_config["input"]["split"] == "train"
    assert distillation_config["input"]["dev_access"] == "prohibited"
    assert distillation_config["input"]["test_access"] == "prohibited"
    assert distillation_config["prompt_decode"]["selected_profile"] == "greedy-v1"


def test_d1_contract_is_mvp_sized_and_reuses_the_d0_prefix() -> None:
    config = load_distillation_config(D1_CONFIG_PATH)
    assert config["sampling"] == {
        "unit": "directed_sample",
        "records_per_route": 2_224,
        "total_records": 40_032,
        "selection_seed": "diesel-mt-td08-d0-v1",
        "order": "frozen-route-order-then-selection-hash",
        "replacement": False,
    }
    assert config["acceptance_gates"]["minimum_accepted_per_route"] == 2_000
    assert config["reuse"]["required_records_per_route"] == 128
    assert config["reuse"]["require_selected_prefix"] is True


def test_deterministic_sampling_keeps_the_small_run_as_an_exact_prefix() -> None:
    records = [
        _sample(source, target, index)
        for source, target in directed_routes()
        for index in range(6)
    ]
    small = deterministic_route_sample(records, per_route=2, seed="shared-seed")
    large = deterministic_route_sample(records, per_route=5, seed="shared-seed")
    for route_index in range(len(ROUTES)):
        small_ids = [
            record["sample_id"] for record in small[route_index * 2 : (route_index + 1) * 2]
        ]
        large_prefix_ids = [
            record["sample_id"] for record in large[route_index * 5 : route_index * 5 + 2]
        ]
        assert small_ids == large_prefix_ids


def test_distillation_contract_rejects_test_or_hardware_drift(
    distillation_config: dict[str, object],
) -> None:
    changed = copy.deepcopy(distillation_config)
    changed["input"]["split"] = "test"
    with pytest.raises(DistillationError, match="train-only"):
        validate_distillation_config(changed)

    changed = copy.deepcopy(distillation_config)
    changed["runtime"]["gpu_model"] = "specific-device"
    with pytest.raises(DistillationError, match="unknown fields: gpu_model"):
        validate_distillation_config(changed)


def test_generation_contract_is_deterministic_and_covers_routes(
    distillation_config: dict[str, object],
    prompt_config: dict[str, object],
) -> None:
    samples = [
        _sample(source, target, index)
        for source, target in directed_routes()
        for index in range(2)
    ]
    first = generation_contract(distillation_config, prompt_config, samples)
    second = generation_contract(distillation_config, prompt_config, list(samples))
    assert first == second
    assert first["input"]["split"] == "train"
    assert first["input"]["test_access"] == "prohibited"
    assert first["sampling"]["records_by_route"] == {route: 2 for route in ROUTES}


def test_route_checkpoint_accepts_only_a_valid_result_prefix(tmp_path: Path) -> None:
    samples = [_sample("eng_Latn", "jpn_Jpan", index) for index in range(3)]
    path = tmp_path / "state.json"
    results = [{"sample_id": samples[0]["sample_id"], "raw_output": "結果"}]
    _write_route_state(
        path,
        route="eng_Latn->jpn_Jpan",
        contract_sha256="a" * 64,
        samples=samples,
        results=results,
    )
    assert _load_route_state(
        path,
        route="eng_Latn->jpn_Jpan",
        contract_sha256="a" * 64,
        samples=samples,
    ) == results

    changed = copy.deepcopy(samples)
    changed[0]["sample_id"] = "different"
    with pytest.raises(DistillationError, match="identity differs"):
        _load_route_state(
            path,
            route="eng_Latn->jpn_Jpan",
            contract_sha256="a" * 64,
            samples=changed,
        )


def test_atomic_sample_checkpoints_resume_a_contiguous_prefix(tmp_path: Path) -> None:
    samples = [_sample("eng_Latn", "jpn_Jpan", index) for index in range(3)]
    assert _load_route_checkpoints(
        tmp_path,
        route="eng_Latn->jpn_Jpan",
        contract_sha256="b" * 64,
        samples=samples,
    ) == []
    record = {"sample_id": samples[0]["sample_id"], "raw_output": "結果"}
    _write_sample_checkpoint(
        tmp_path,
        route="eng_Latn->jpn_Jpan",
        index=0,
        record=record,
    )
    assert _load_route_checkpoints(
        tmp_path,
        route="eng_Latn->jpn_Jpan",
        contract_sha256="b" * 64,
        samples=samples,
    ) == [record]


def test_review_queue_has_twenty_per_route_and_extra_traditional(
    distillation_config: dict[str, object],
) -> None:
    records: list[dict[str, object]] = []
    for route in ROUTES:
        for index in range(25):
            records.append(
                {
                    "record_id": f"{route}-{index}",
                    "route": route,
                    "sample_id": f"sample-{route}-{index}",
                    "sample_group_id": f"group-{route}-{index}",
                    "source_text": f"source {index}",
                    "reference_text": f"reference {index}",
                    "normalized_output": f"output {index}",
                    "accepted": True,
                    "rejection_reasons": [],
                    "script_counts": {},
                    "chinese_script_evidence": {"simplified": 0, "traditional": index % 3},
                }
            )
        records.append(
            {
                "record_id": f"{route}-rejected",
                "route": route,
                "sample_id": f"sample-{route}-rejected",
                "sample_group_id": f"group-{route}-rejected",
                "source_text": "source",
                "reference_text": "reference",
                "normalized_output": "",
                "accepted": False,
                "rejection_reasons": ["empty_output"],
                "script_counts": {},
                "chinese_script_evidence": {"simplified": 0, "traditional": 0},
            }
        )
    queue = prepare_review_queue(records, distillation_config)
    accepted = [record for record in queue if record["selection_tag"] == "accepted"]
    rejected = [record for record in queue if record["selection_tag"] == "rejected"]
    extra = [record for record in queue if record["selection_tag"] == "traditional-extra"]
    assert len(accepted) == 18 * 20
    assert len(rejected) == 18
    assert len(extra) == 3 * 5
    assert all(record["route"].endswith("->zho_Hant") for record in extra)


def test_work_root_override_must_be_absolute(
    distillation_config: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIESEL_MT_DISTILLATION_WORK_ROOT", str(tmp_path))
    assert resolve_work_root(ROOT, distillation_config) == tmp_path.resolve()
    monkeypatch.setenv("DIESEL_MT_DISTILLATION_WORK_ROOT", "relative/path")
    with pytest.raises(DistillationError, match="must be absolute"):
        resolve_work_root(ROOT, distillation_config)


def test_manual_review_can_only_restore_source_copy_false_positives(
    distillation_config: dict[str, object], tmp_path: Path
) -> None:
    queue_path = tmp_path / "review.jsonl"
    queue = [
        {
            "review_id": "review-source-copy",
            "record_id": "record-source-copy",
            "route": "jpn_Jpan->zho_Hant",
            "selection_tag": "rejected",
            "automated_accepted": False,
            "automated_rejection_reasons": ["source_copy"],
        }
    ]
    queue_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in queue),
        encoding="utf-8",
    )
    counts = {route: 0 for route in ROUTES}
    rejected_counts = dict(counts)
    rejected_counts["jpn_Jpan->zho_Hant"] = 1
    attestation = {
        "schema_version": 1,
        "status": "complete",
        "queue": {
            "path": distillation_config["outputs"]["manual_review_queue"],
            "records": 1,
            "bytes": queue_path.stat().st_size,
            "sha256": hashlib.sha256(queue_path.read_bytes()).hexdigest(),
        },
        "reviewer": "fixture",
        "method": "fixture",
        "decisions": {
            "accepted_reviewed_by_route": counts,
            "rejected_reviewed_by_route": rejected_counts,
            "traditional_extra_reviewed_by_route": counts,
            "manual_rejections": [],
            "rejected_rule_mismatches": [
                {
                    "review_id": "review-source-copy",
                    "rule": "source_copy",
                    "reason": "A shared Han-script proper noun is a valid unchanged translation.",
                }
            ],
            "systemic_blocker": False,
        },
        "notes": [],
    }
    attestation_path = tmp_path / "attestation.yaml"
    attestation_path.write_text(
        yaml.safe_dump(attestation, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    validated = validate_manual_attestation(
        attestation_path,
        queue_path,
        queue,
        distillation_config,
    )
    assert validated["decisions"]["rejected_rule_mismatches"][0]["rule"] == "source_copy"

    attestation["decisions"]["rejected_rule_mismatches"][0]["rule"] = "empty_output"
    attestation_path.write_text(
        yaml.safe_dump(attestation, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(DistillationError, match="source_copy-only"):
        validate_manual_attestation(
            attestation_path,
            queue_path,
            queue,
            distillation_config,
        )


def test_td08_frozen_evidence_closes_d0_without_starting_td09() -> None:
    evidence = json.loads(
        (ROOT / "artifacts" / "model-training" / "td08-distilled-data.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["status"] == "complete"
    assert evidence["scope"] == {
        "accepted": 2263,
        "dev_records": 0,
        "filtered": 41,
        "input": 2304,
        "routes": 18,
        "teacher_synthetic": 2263,
        "test_records": 0,
    }
    assert evidence["manual_review"] == {
        "manual_acceptances": 4,
        "manual_rejections": 39,
        "queue_records": 381,
        "systemic_blocker": False,
    }
    assert evidence["replay"] == {
        "exact_normalized": True,
        "exact_raw": True,
        "records": 36,
    }
    assert evidence["quality"]["gate_failures"] == []
    assert set(evidence["quality"]["routes"]) == set(ROUTES)
    assert all(
        route["accepted"] >= 100
        and route["accepted_rate"] >= 0.90
        and route["script_compliance_rate"] >= 0.98
        and route["retry_rate"] <= 0.05
        for route in evidence["quality"]["routes"].values()
    )
    assert evidence["test_accessed"] is False
    assert evidence["dev_accessed"] is False
    assert evidence["td09_started"] is False


def test_td08_d1_evidence_is_mvp_sized_and_closes_td08() -> None:
    evidence = json.loads(
        (ROOT / "artifacts" / "model-training" / "td08-d1-distilled-data.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["status"] == "complete"
    assert evidence["release"] == "d1-hymt2-distillation-mvp"
    assert evidence["corpus_maturity"] == "mvp"
    assert evidence["scope"] == {
        "accepted": 39_941,
        "dev_records": 0,
        "filtered": 91,
        "input": 40_032,
        "reused_d0_inputs": 2_304,
        "routes": 18,
        "teacher_synthetic": 39_941,
        "test_records": 0,
    }
    assert evidence["manual_review"] == {
        "manual_acceptances": 31,
        "manual_rejections": 52,
        "queue_records": 444,
        "systemic_blocker": False,
    }
    assert evidence["replay"] == {
        "exact_normalized": True,
        "exact_raw": True,
        "records": 36,
    }
    assert evidence["quality"]["gate_failures"] == []
    assert set(evidence["quality"]["routes"]) == set(ROUTES)
    assert all(
        route["accepted"] >= 2_000
        and route["accepted_rate"] >= 0.90
        and route["script_compliance_rate"] >= 0.98
        and route["retry_rate"] <= 0.05
        for route in evidence["quality"]["routes"].values()
    )
    assert evidence["downstream_consumer"] == "TD-15"
    assert evidence["td08_completed"] is True
    assert evidence["td09_started"] is False
    assert evidence["test_accessed"] is False
    assert evidence["dev_accessed"] is False
