import numpy as np

from trm_reranker.evaluation.significance import paired_bootstrap


def test_bootstrap_no_difference():
    rng = np.random.default_rng(0)
    a = rng.random(500)
    b = a + rng.normal(0, 1e-6, size=500)
    pvalue, ci_low, ci_high = paired_bootstrap(a, b, n_bootstrap=2000, seed=1)
    assert pvalue > 0.05
    assert ci_low <= 0 <= ci_high or abs(ci_low) < 1e-4


def test_bootstrap_clear_difference():
    rng = np.random.default_rng(0)
    a = rng.random(500)
    b = a - 0.2
    pvalue, ci_low, _ci_high = paired_bootstrap(a, b, n_bootstrap=2000, seed=1)
    assert pvalue < 0.01
    assert ci_low > 0
