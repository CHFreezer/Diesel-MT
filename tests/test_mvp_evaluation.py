from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_training_contract import MODEL_TO_PRODUCT  # noqa: E402
from mvp_evaluation import (  # noqa: E402
    EvaluationContractError,
    aggregate_results,
    corpus_metrics,
    evaluate_checkpoint,
    load_evaluation_config,
    publish_evaluation,
    target_script_compliant,
    validate_evaluation_config,
)
from mvp_training import ROUTE_ORDER  # noqa: E402


@pytest.fixture(scope="module")
def evaluation_config() -> dict:
    return load_evaluation_config(ROOT / "configs/mvp_evaluation.yaml")


def _text(language: str) -> str:
    return {
        "eng_Latn": "hello world",
        "jpn_Jpan": "こんにちは世界",
        "kor_Hang": "안녕하세요 세계",
        "zho_Hans": "简体中文",
        "zho_Hant": "繁體中文",
    }[language]


def _rows(per_route: int = 1) -> list[dict]:
    rows = []
    for route in ROUTE_ORDER:
        source, target = route.split("->")
        direction = f"{MODEL_TO_PRODUCT[source]}->{MODEL_TO_PRODUCT[target]}"
        for index in range(per_route):
            reference = _text(target)
            rows.append(
                {
                    "sample_id": f"{route}:{index}",
                    "route": route,
                    "product_direction": direction,
                    "prediction": reference,
                    "reference": reference,
                    "loss_sum": 2.0,
                    "target_loss_tokens": 2,
                    "script_compliant": True,
                    "empty_output": False,
                    "source_copy": False,
                    "target_control": True,
                    "source_truncated": False,
                    "target_truncated": False,
                    "length_ratio": 1.0,
                }
            )
    return rows


def test_identical_corpus_has_perfect_metrics(evaluation_config: dict) -> None:
    result = corpus_metrics(["a b", "繁體中文"], ["a b", "繁體中文"], evaluation_config)
    assert result["sacrebleu"] == pytest.approx(100.0)
    assert result["chrf"] == pytest.approx(100.0)
    assert "version:2.6.0" in result["sacrebleu_signature"]


def test_target_script_compliance_distinguishes_all_tags() -> None:
    assert target_script_compliant("English text", "eng_Latn")
    assert not target_script_compliant("한국어 문장", "eng_Latn")
    assert target_script_compliant("日本語です", "jpn_Jpan")
    assert target_script_compliant("한국어입니다", "kor_Hang")
    assert target_script_compliant("简体中文", "zho_Hans")
    assert not target_script_compliant("繁體中文", "zho_Hans")
    assert target_script_compliant("繁體中文", "zho_Hant")
    assert not target_script_compliant("简体中文", "zho_Hant")


def test_aggregate_preserves_20_routes_12_products_and_two_conversions(
    evaluation_config: dict,
) -> None:
    result = aggregate_results(_rows(per_route=2), evaluation_config)
    assert len(result["route20"]) == 20
    assert len(result["product_directions12"]) == 12
    assert set(result["chinese_conversions2"]) == {
        "zho_Hans->zho_Hant",
        "zho_Hant->zho_Hans",
    }
    chinese_to_english = result["product_directions12"]["Chinese->English"]
    assert chinese_to_english["samples"] == 4
    assert chinese_to_english["tag_routes"] == [
        "zho_Hans->eng_Latn",
        "zho_Hant->eng_Latn",
    ]
    assert chinese_to_english["tag_route_weights"] == {
        "zho_Hans->eng_Latn": 0.5,
        "zho_Hant->eng_Latn": 0.5,
    }


def test_aggregate_rejects_empty_or_incomplete_route_matrix(evaluation_config: dict) -> None:
    with pytest.raises(EvaluationContractError, match="empty"):
        aggregate_results([], evaluation_config)
    with pytest.raises(EvaluationContractError, match="missing routes"):
        aggregate_results(_rows()[:-1], evaluation_config)


def test_config_rejects_metric_or_test_boundary_drift(evaluation_config: dict) -> None:
    changed = deepcopy(evaluation_config)
    changed["metrics"]["bleu_tokenize"] = "none"
    with pytest.raises(EvaluationContractError, match="metric semantics"):
        validate_evaluation_config(changed)
    changed = deepcopy(evaluation_config)
    changed["runtime"]["test_requires_explicit_authorization"] = False
    with pytest.raises(EvaluationContractError, match="explicit authorization"):
        validate_evaluation_config(changed)


def test_test_split_requires_explicit_authorization(tmp_path: Path) -> None:
    with pytest.raises(EvaluationContractError, match="explicit --allow-test"):
        evaluate_checkpoint(
            repository_root=ROOT,
            evaluation_config_path=ROOT / "configs/mvp_evaluation.yaml",
            checkpoint=tmp_path / "missing",
            split="test",
            allow_test=False,
        )


def test_atomic_publication_and_manifest_last(tmp_path: Path, evaluation_config: dict) -> None:
    rows = _rows()
    aggregates = aggregate_results(rows, evaluation_config)
    summary = {
        "split": "dev",
        "records": len(rows),
        "identities": {
            "checkpoint_state_sha256": "a" * 64,
            "evaluation_config_file_sha256": "b" * 64,
            "data_sha256": "c" * 64,
        },
        "aggregates": aggregates,
    }
    output = tmp_path / "published"
    manifest = publish_evaluation(output, summary, rows)
    assert manifest["status"] == "complete"
    assert (output / "manifest.json").is_file()
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["records"] == 20
    with pytest.raises(EvaluationContractError, match="refusing to overwrite"):
        publish_evaluation(output, summary, rows)
