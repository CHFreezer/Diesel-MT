from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mvp_resource_profile import (  # noqa: E402
    ResourceProfileError,
    load_m2_profile,
    validate_profile_binding,
)
from mvp_training import load_training_config, validate_resource_budget  # noqa: E402


def test_frozen_profile_matches_selected_candidate_and_soak() -> None:
    profile = load_m2_profile(ROOT / "configs/mvp_training_m2_profile.yaml")
    benchmark = json.loads((ROOT / "artifacts/model-training/reports/m2/resources/candidate-benchmark.json").read_text(encoding="utf-8"))
    soak = load_training_config(ROOT / "configs/mvp_training_td14_soak.yaml")
    assert all(validate_profile_binding(profile=profile, benchmark=benchmark, soak_config=soak).values())


def test_profile_rejects_candidate_or_soak_drift() -> None:
    profile = load_m2_profile(ROOT / "configs/mvp_training_m2_profile.yaml")
    benchmark = json.loads((ROOT / "artifacts/model-training/reports/m2/resources/candidate-benchmark.json").read_text(encoding="utf-8"))
    soak = load_training_config(ROOT / "configs/mvp_training_td14_soak.yaml")
    changed = deepcopy(profile)
    changed["resource_profile"]["micro_batch_size"] += 1
    with pytest.raises(ResourceProfileError, match="selected candidate"):
        validate_profile_binding(profile=changed, benchmark=benchmark, soak_config=soak)
    changed_soak = deepcopy(soak)
    changed_soak["resource_profile"]["max_source_length"] = 63
    with pytest.raises(ResourceProfileError, match="soak resource"):
        validate_profile_binding(profile=profile, benchmark=benchmark, soak_config=changed_soak)


def test_capacity_adaptation_is_data_driven_not_device_named() -> None:
    profile = load_m2_profile(ROOT / "configs/mvp_training_m2_profile.yaml")
    resource = profile["resource_profile"]
    assert "name" not in resource
    assert "cuda_devices" not in resource
    runtime = {
        "device_total_bytes": 16 * 1024**3,
        "host_available_bytes": 64 * 1024**3,
    }
    accepted = validate_resource_budget(resource, runtime)
    assert accepted["absolute_budget_bytes"] == 12000 * 1024**2
    smaller = deepcopy(resource)
    smaller["device_memory_budget_mib"] = 6000
    smaller["device_memory_reserve_mib"] = 1024
    smaller_runtime = {"device_total_bytes": 8 * 1024**3, "host_available_bytes": 64 * 1024**3}
    assert validate_resource_budget(smaller, smaller_runtime)["absolute_budget_bytes"] == 6000 * 1024**2


def test_profile_has_no_semantic_drive_letter() -> None:
    profile = load_m2_profile(ROOT / "configs/mvp_training_m2_profile.yaml")
    assert ":\\" not in json.dumps(profile)
