from pathlib import Path

from trm_reranker.config import load_experiment_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_base_plus_arm_merge():
    cfg = load_experiment_config([REPO_ROOT / "configs/base.yaml", REPO_ROOT / "configs/arms/trm.yaml"])
    assert cfg.model.arch == "trm"
    assert cfg.experiment.name == "trm_reranker"
    assert cfg.training.learning_rate == 2e-4
    assert cfg.model.params["H_cycles"] == 2


def test_overrides_win():
    cfg = load_experiment_config(
        [REPO_ROOT / "configs/base.yaml", REPO_ROOT / "configs/arms/trm.yaml"],
        overrides=["experiment.seed=42", "training.learning_rate=1e-5", "model.params.L_cycles=8"],
    )
    assert cfg.experiment.seed == 42
    assert cfg.training.learning_rate == 1e-5
    assert cfg.model.params["L_cycles"] == 8


def test_all_arm_configs_parse():
    for arm in sorted((REPO_ROOT / "configs/arms").glob("*.yaml")):
        cfg = load_experiment_config([REPO_ROOT / "configs/base.yaml", arm])
        assert cfg.model.arch
        assert cfg.experiment.name != "experiment"


def test_unknown_key_rejected():
    import pytest

    with pytest.raises(ValueError):
        load_experiment_config([REPO_ROOT / "configs/base.yaml"], overrides=["training.learning_rte=1e-5"])
