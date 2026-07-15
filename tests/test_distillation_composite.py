from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from distillation_composite import (  # noqa: E402
    COMPOSITE_IDENTITY,
    CompositeError,
    load_composite_config,
)
from model_training_contract import directed_routes  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "hymt2_distillation_d1_20route_composite.yaml"


def test_composite_config_covers_exactly_twenty_disjoint_routes() -> None:
    config = load_composite_config(CONFIG_PATH)
    assert config["identity"] == COMPOSITE_IDENTITY
    component_routes = [
        route
        for component in config["components"]
        for route in component["routes"]
    ]
    assert len(component_routes) == 20
    assert len(set(component_routes)) == 20
    assert set(component_routes) == {
        f"{source}->{target}" for source, target in directed_routes()
    }
    assert sum(component["accepted_records"] for component in config["components"]) == 44_361


def test_composite_config_rejects_overlapping_component_routes(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    changed = copy.deepcopy(config)
    changed["components"][1]["routes"][0] = changed["components"][0]["routes"][0]
    path = tmp_path / "composite.yaml"
    path.write_text(
        yaml.safe_dump(changed, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(CompositeError, match="overlap"):
        load_composite_config(path)


def test_composite_evidence_binds_both_immutable_components() -> None:
    config = load_composite_config(CONFIG_PATH)
    evidence = json.loads(
        (
            ROOT
            / "artifacts"
            / "model-training"
            / "td08-d1-20route-composite.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["status"] == "complete"
    assert evidence["scope"]["accepted"] == 44_361
    assert evidence["scope"]["routes"] == 20
    assert evidence["test_accessed"] is False
    assert evidence["dev_accessed"] is False
    assert [component["component_id"] for component in evidence["components"]] == [
        component["component_id"] for component in config["components"]
    ]
    assert {
        component["manifest_sha256"] for component in evidence["components"]
    } == {component["manifest_sha256"] for component in config["components"]}
