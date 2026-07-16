from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mvp_distillation_ab import (  # noqa: E402
    DistillationABError,
    compare_recipe_configs,
    load_ab_config,
)
from mvp_training import load_training_config  # noqa: E402


def _contracts() -> tuple[dict, dict, dict]:
    human = load_training_config(ROOT / "configs/mvp_training_m2_human.yaml")
    distilled = load_training_config(ROOT / "configs/mvp_training_m2_distilled.yaml")
    import yaml
    profile = yaml.safe_load((ROOT / "configs/mvp_training_m2_profile.yaml").read_text(encoding="utf-8"))
    return human, distilled, profile


def test_recipes_differ_only_by_arm_input_identity() -> None:
    human, distilled, profile = _contracts()
    assert all(compare_recipe_configs(human, distilled, profile).values())


def test_optimizer_or_exposure_drift_is_rejected() -> None:
    human, distilled, profile = _contracts()
    changed = deepcopy(distilled)
    changed["optimization"]["max_optimizer_steps"] += 1
    with pytest.raises(DistillationABError, match="differ outside"):
        compare_recipe_configs(human, changed, profile)
    changed = deepcopy(distilled)
    changed["data"]["route_weights"]["eng_Latn->jpn_Jpan"] = 2.0
    with pytest.raises(DistillationABError, match="differ outside"):
        compare_recipe_configs(human, changed, profile)


def test_standalone_d1_or_small_route_contract_is_rejected(tmp_path: Path) -> None:
    import yaml
    config = load_ab_config(ROOT / "configs/mvp_distillation_ab.yaml")
    changed = deepcopy(config)
    changed["distilled"]["required_identity_name"] = "d1-v1"
    path = tmp_path / "standalone.yaml"
    path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    with pytest.raises(DistillationABError, match="standalone"):
        load_ab_config(path)
    changed = deepcopy(config)
    changed["distilled"]["minimum_accepted_per_route"] = 1999
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    with pytest.raises(DistillationABError, match="2,000"):
        load_ab_config(path)


def test_dev_rules_are_frozen_before_results_and_keep_chinese_separate() -> None:
    config = load_ab_config(ROOT / "configs/mvp_distillation_ab.yaml")
    selection = config["dev_selection"]
    assert selection["within_arm_priority"][-1] == "optimizer_step_asc"
    assert selection["distilled_vs_human"]["fallback_if_any_gate_fails"] == "human-only"
    assert selection["distilled_vs_human"]["exact_tie_break"] == "human-only"
    assert selection["distilled_vs_human"]["evaluate_zho_Hans_and_zho_Hant_separately"] is True
    assert config["evaluation"]["test_access"] == "forbidden_until_unique_td16_candidate_is_frozen"


def test_published_cohort_has_20_routes_and_minimum() -> None:
    manifest = json.loads((ROOT / "data/model/corpus/mvp/ab/m2-v1/manifest.json").read_text(encoding="utf-8"))
    assert manifest["intersection"]["routes"] == 20
    assert manifest["intersection"]["minimum_route_records"] >= 2000
    assert manifest["intersection"]["teacher_rejected_or_filtered_included"] == 0
