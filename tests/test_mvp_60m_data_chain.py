from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mvp_60m_data_pipeline import AbilityDataError  # noqa: E402
from validate_mvp_60m_data_chain import _require_complete, load_config  # noqa: E402


def test_chain_config_binds_frozen_no_test_contract() -> None:
    config = load_config(ROOT / "configs/mvp_60m_data_chain.yaml")
    assert config["requirements"]["teacher_sampling_weight"] == 0.80
    assert config["requirements"]["human_sampling_weight"] == 0.20
    assert config["requirements"]["formal_test_access"] == "prohibited"
    assert config["requirements"]["formal_devtest_access"] == "prohibited"


def test_chain_rejects_incomplete_or_unproven_test_isolation() -> None:
    with pytest.raises(AbilityDataError, match="not complete"):
        _require_complete({"status": "pending", "formal_test_accessed": False}, "fixture")
    with pytest.raises(AbilityDataError, match="isolation is unproven"):
        _require_complete({"status": "complete", "formal_test_accessed": None}, "fixture")


def test_chain_config_rejects_relaxed_devtest_boundary(tmp_path: Path) -> None:
    import yaml

    config = copy.deepcopy(load_config(ROOT / "configs/mvp_60m_data_chain.yaml"))
    config["requirements"]["formal_devtest_access"] = "allowed"
    path = tmp_path / "chain.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(AbilityDataError, match="devtest access"):
        load_config(path)
