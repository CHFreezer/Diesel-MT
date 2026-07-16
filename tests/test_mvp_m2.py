from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import mvp_m2  # noqa: E402
from freeze_tokenizer_artifact import sha256_file  # noqa: E402
from mvp_m2 import (  # noqa: E402
    ARM_MANIFEST,
    CANDIDATE_MANIFEST,
    M2ContractError,
    run_selected_test_once,
    select_m2_candidate,
)
from mvp_training import ROUTE_ORDER, git_identity  # noqa: E402


AB_CONFIG = ROOT / "configs" / "mvp_distillation_ab.yaml"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _summary(*, chrf: float, route_overrides: dict[str, float] | None = None) -> dict:
    route_overrides = route_overrides or {}
    route20 = {
        route: {
            "samples": 10,
            "loss": 2.0,
            "sacrebleu": 5.0,
            "chrf": route_overrides.get(route, chrf),
            "script_compliance_rate": 1.0,
            "empty_output_rate": 0.0,
            "source_copy_rate": 0.0,
            "target_control_rate": 1.0,
        }
        for route in ROUTE_ORDER
    }
    return {
        "schema_version": 1,
        "status": "passed",
        "split": "dev",
        "test_access_explicitly_authorized": False,
        "records": 200,
        "identities": {"checkpoint_state_sha256": "a" * 64},
        "aggregates": {
            "overall": {
                "loss": 2.0,
                "sacrebleu": 5.0,
                "chrf": chrf,
                "script_compliance_rate": 1.0,
                "empty_output_rate": 0.0,
                "source_copy_rate": 0.0,
                "target_control_rate": 1.0,
            },
            "route20": route20,
        },
    }


def _candidate(
    root: Path,
    *,
    arm: str,
    step: int,
    summary: dict,
) -> dict:
    name = f"step-{step:08d}"
    candidate = root / arm / "candidates" / name
    payload = candidate / "config.json"
    payload.parent.mkdir(parents=True)
    payload.write_text("{}\n", encoding="utf-8", newline="\n")
    state_sha256 = summary["identities"]["checkpoint_state_sha256"]
    candidate_manifest = {
        "schema_version": 1,
        "status": "complete",
        "arm": arm,
        "optimizer_step": step,
        "training_config": {
            "path": f"configs/{arm}.yaml",
            "file_sha256": "1" * 64,
            "canonical_sha256": "2" * 64,
        },
        "checkpoint_identity_sha256": "3" * 64,
        "code": {
            "scripts/mvp_m2.py": "6" * 64,
            "scripts/run_mvp_m2.py": "7" * 64,
        },
        "source_checkpoint": {
            "path": str(root / arm / "exact" / name),
            "manifest_sha256": "4" * 64,
        },
        "state_dict_sha256": state_sha256,
        "model_alignment": {},
        "tokenizer_manifest_sha256": "5" * 64,
        "files": [
            {
                "path": "config.json",
                "bytes": payload.stat().st_size,
                "sha256": sha256_file(payload),
            }
        ],
    }
    _write_json(candidate / CANDIDATE_MANIFEST, candidate_manifest)

    evaluation = root / arm / "evaluations" / name
    _write_json(evaluation / "summary.json", summary)
    evaluation_manifest = {
        "schema_version": 1,
        "status": "complete",
        "checkpoint_state_sha256": state_sha256,
    }
    _write_json(evaluation / "manifest.json", evaluation_manifest)
    return {
        "optimizer_step": step,
        "candidate_path": str(candidate.resolve()),
        "candidate_manifest_sha256": sha256_file(candidate / CANDIDATE_MANIFEST),
        "checkpoint_state_sha256": state_sha256,
        "dev_evaluation_path": str(evaluation.resolve()),
        "dev_summary_sha256": sha256_file(evaluation / "summary.json"),
        "dev_manifest_sha256": sha256_file(evaluation / "manifest.json"),
        "dev_records": 200,
    }


def _arm_manifest(root: Path, *, arm: str, candidates: list[dict]) -> Path:
    path = root / arm / "evaluations" / ARM_MANIFEST
    _write_json(
        path,
        {
            "schema_version": 1,
            "status": "complete",
            "task": "TD-16",
            "arm": arm,
            "ab_config_sha256": sha256_file(AB_CONFIG),
            "training_config": {},
            "checkpoint_identity_sha256": "3" * 64,
            "code": {
                "scripts/mvp_m2.py": sha256_file(ROOT / "scripts/mvp_m2.py"),
                "scripts/run_mvp_m2.py": sha256_file(ROOT / "scripts/run_mvp_m2.py"),
            },
            "git": git_identity(ROOT),
            "expected_optimizer_steps": [candidate["optimizer_step"] for candidate in candidates],
            "missing_optimizer_steps": [],
            "candidates": candidates,
            "checkpoint_retention": {},
            "test_access": "forbidden",
        },
    )
    return path


def _selection(
    tmp_path: Path,
    *,
    distilled_summary: dict | None = None,
) -> tuple[Path, dict]:
    human = _candidate(
        tmp_path,
        arm="human-only",
        step=1000,
        summary=_summary(chrf=10.0),
    )
    distilled = _candidate(
        tmp_path,
        arm="distilled",
        step=1000,
        summary=distilled_summary or _summary(chrf=11.0),
    )
    human_manifest = _arm_manifest(tmp_path, arm="human-only", candidates=[human])
    distilled_manifest = _arm_manifest(tmp_path, arm="distilled", candidates=[distilled])
    output = tmp_path / "selection.json"
    result = select_m2_candidate(
        ab_config_path=AB_CONFIG,
        human_arm_manifest_path=human_manifest,
        distilled_arm_manifest_path=distilled_manifest,
        output_path=output,
    )
    return output, result


def test_dev_selection_picks_distilled_only_when_every_frozen_gate_passes(
    tmp_path: Path,
) -> None:
    _, result = _selection(tmp_path)
    assert result["selected"]["arm"] == "distilled"
    assert all(result["comparison"]["gates"].values())
    assert result["test_access"] == {
        "authorized_after_selection": True,
        "runs_allowed": 1,
        "runs_consumed": 0,
    }


def test_any_route_regression_falls_back_to_human_only(tmp_path: Path) -> None:
    route = ROUTE_ORDER[0]
    _, result = _selection(
        tmp_path,
        distilled_summary=_summary(chrf=11.0, route_overrides={route: 6.0}),
    )
    assert result["selected"]["arm"] == "human-only"
    assert not result["comparison"]["gates"][
        "maximum_any_route_chrf_degradation"
    ]


def test_selection_rejects_test_or_tampered_dev_evidence(tmp_path: Path) -> None:
    human = _candidate(
        tmp_path,
        arm="human-only",
        step=1000,
        summary=_summary(chrf=10.0),
    )
    distilled = _candidate(
        tmp_path,
        arm="distilled",
        step=1000,
        summary=_summary(chrf=11.0),
    )
    human_manifest = _arm_manifest(tmp_path, arm="human-only", candidates=[human])
    distilled_manifest = _arm_manifest(tmp_path, arm="distilled", candidates=[distilled])
    summary_path = Path(human["dev_evaluation_path"]) / "summary.json"
    summary_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(M2ContractError, match="summary SHA-256 changed"):
        select_m2_candidate(
            ab_config_path=AB_CONFIG,
            human_arm_manifest_path=human_manifest,
            distilled_arm_manifest_path=distilled_manifest,
            output_path=tmp_path / "selection.json",
        )


def test_formal_test_authorization_is_consumed_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path, selection = _selection(tmp_path)
    output = tmp_path / "formal-test"
    receipt = tmp_path / "formal-test-receipt.json"
    report = tmp_path / "formal-test-report.json"
    calls = []

    def fake_evaluate(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        summary = {
            "schema_version": 1,
            "status": "passed",
            "split": "test",
            "test_access_explicitly_authorized": True,
            "identities": {
                "checkpoint_state_sha256": selection["selected"][
                    "checkpoint_state_sha256"
                ]
            },
        }
        return summary, []

    def fake_publish(path, summary, samples):  # type: ignore[no-untyped-def]
        del samples
        path.mkdir(parents=True)
        _write_json(path / "summary.json", summary)
        manifest = {
            "status": "complete",
            "checkpoint_state_sha256": summary["identities"][
                "checkpoint_state_sha256"
            ],
        }
        _write_json(path / "manifest.json", manifest)
        return manifest

    monkeypatch.setattr(mvp_m2, "evaluate_checkpoint", fake_evaluate)
    monkeypatch.setattr(mvp_m2, "publish_evaluation", fake_publish)
    result = run_selected_test_once(
        repository_root=ROOT,
        selection_path=selection_path,
        output_directory=output,
        receipt_path=receipt,
        report_path=report,
    )
    assert result["split"] == "test"
    assert calls[0]["split"] == "test" and calls[0]["allow_test"] is True
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "complete"
    with pytest.raises(M2ContractError, match="already exists"):
        run_selected_test_once(
            repository_root=ROOT,
            selection_path=selection_path,
            output_directory=output,
            receipt_path=receipt,
            report_path=report,
        )
