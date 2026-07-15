from pathlib import Path

from trm_reranker.config import load_experiment_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_base_plus_variant_merge():
    cfg = load_experiment_config([REPO_ROOT / "configs/base.yaml", REPO_ROOT / "configs/variants/trm.yaml"])
    assert cfg.model.arch == "trm"
    assert cfg.experiment.name == "trm_reranker"
    assert cfg.training.learning_rate == 2e-4
    assert cfg.model.params["H_cycles"] == 2


def test_overrides_win():
    cfg = load_experiment_config(
        [REPO_ROOT / "configs/base.yaml", REPO_ROOT / "configs/variants/trm.yaml"],
        overrides=["experiment.seed=42", "training.learning_rate=1e-5", "model.params.L_cycles=8"],
    )
    assert cfg.experiment.seed == 42
    assert cfg.training.learning_rate == 1e-5
    assert cfg.model.params["L_cycles"] == 8


def test_all_variant_configs_parse():
    for variant in sorted((REPO_ROOT / "configs/variants").glob("*.yaml")):
        cfg = load_experiment_config([REPO_ROOT / "configs/base.yaml", variant])
        assert cfg.model.arch
        assert cfg.experiment.name != "experiment"


def test_unknown_key_rejected():
    import pytest

    with pytest.raises(ValueError):
        load_experiment_config([REPO_ROOT / "configs/base.yaml"], overrides=["training.learning_rte=1e-5"])


def test_unknown_section_rejected_in_override():
    import pytest

    with pytest.raises(ValueError, match="unknown section"):
        load_experiment_config([REPO_ROOT / "configs/base.yaml"], overrides=["trainig.learning_rate=1e-5"])


def test_unknown_section_rejected_in_file(tmp_path):
    import pytest

    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("trainig:\n  learning_rate: 1e-5\n")
    with pytest.raises(ValueError, match="unknown top-level sections"):
        load_experiment_config([REPO_ROOT / "configs/base.yaml", bad_config])


def test_override_coercion_preserves_strings():
    cfg = load_experiment_config(
        [REPO_ROOT / "configs/base.yaml", REPO_ROOT / "configs/variants/trm.yaml"],
        overrides=[
            "experiment.run_id=seed007",
            "training.learning_rate=1e-5",  # YAML 1.1 parses this as str; must become float
            "training.precision=bf16-mixed",
        ],
    )
    assert cfg.experiment.run_id == "seed007"
    assert cfg.training.learning_rate == 1e-5
    assert cfg.training.precision == "bf16-mixed"


def test_local_example_config_merges():
    cfg = load_experiment_config(
        [
            REPO_ROOT / "configs/base.yaml",
            REPO_ROOT / "configs/variants/trm.yaml",
            REPO_ROOT / "configs/local.example.yaml",
        ]
    )
    assert cfg.experiment.output_root
    assert cfg.data.run_data_manifest_path
