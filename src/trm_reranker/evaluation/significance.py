"""Per-query paired significance tests for E2.

Implements the two tests expected at IR venues (Smucker et al. 2007):
paired t-test and paired bootstrap over per-query metric values. Inputs are
the per-query CSV files written by ``evaluate_reranker(per_query_path=...)``.
"""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PairedTestResult:
    metric: str
    n_queries: int
    mean_a: float
    mean_b: float
    mean_diff: float  # a - b
    t_statistic: Optional[float]
    t_pvalue: Optional[float]
    bootstrap_pvalue: float
    diff_ci_low: float
    diff_ci_high: float
    n_bootstrap: int

    def to_dict(self) -> Dict:
        return asdict(self)


def load_per_query_metrics(path: Path, column: str) -> Dict[int, float]:
    values: Dict[int, float] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise KeyError(f"Column {column!r} not found in {path}. Available: {reader.fieldnames}")
        for row in reader:
            values[int(row["qid"])] = float(row[column])
    return values


def align_by_query(values_a: Dict[int, float], values_b: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    common_qids = sorted(set(values_a) & set(values_b))
    if not common_qids:
        raise ValueError("No common query ids between the two runs")
    a = np.array([values_a[qid] for qid in common_qids], dtype=np.float64)
    b = np.array([values_b[qid] for qid in common_qids], dtype=np.float64)
    return a, b, common_qids


def paired_t_test(a: np.ndarray, b: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    try:
        from scipy import stats
    except ImportError:
        return None, None
    result = stats.ttest_rel(a, b)
    return float(result.statistic), float(result.pvalue)


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 13,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Two-sided paired bootstrap p-value and CI for mean(a - b)."""
    rng = np.random.default_rng(seed)
    diff = a - b
    n = len(diff)
    observed = diff.mean()
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = diff[indices].mean(axis=1)
    # Shift to the null (zero mean) and count means at least as extreme as observed.
    shifted = boot_means - observed
    pvalue = float((np.abs(shifted) >= abs(observed)).mean())
    alpha = (1.0 - ci) / 2.0
    ci_low, ci_high = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return pvalue, float(ci_low), float(ci_high)


def compare_runs(
    per_query_path_a: Path,
    per_query_path_b: Path,
    metric: str = "trm_mrr@10",
    n_bootstrap: int = 10_000,
    seed: int = 13,
) -> PairedTestResult:
    values_a = load_per_query_metrics(per_query_path_a, metric)
    values_b = load_per_query_metrics(per_query_path_b, metric)
    a, b, qids = align_by_query(values_a, values_b)
    t_stat, t_pvalue = paired_t_test(a, b)
    boot_pvalue, ci_low, ci_high = paired_bootstrap(a, b, n_bootstrap=n_bootstrap, seed=seed)
    return PairedTestResult(
        metric=metric,
        n_queries=len(qids),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=float((a - b).mean()),
        t_statistic=t_stat,
        t_pvalue=t_pvalue,
        bootstrap_pvalue=boot_pvalue,
        diff_ci_low=ci_low,
        diff_ci_high=ci_high,
        n_bootstrap=n_bootstrap,
    )
