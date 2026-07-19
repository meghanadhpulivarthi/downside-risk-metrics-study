"""
Tests for the partial-moment core. Every expected number is hand-computed in the
comment above it, so the test fails if the *definition* drifts — not merely if
the code throws (Rule 9). Run: `uv run python tests/test_partial_moments.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

import partial_moments as pm

# Toy series used across tests. target = 0 unless stated.
RETURNS = [0.10, -0.05, 0.02, -0.08]


def test_lpm_degree_one():
    # shortfalls below 0: [0, 0.05, 0, 0.08]; mean = 0.13/4 = 0.0325
    assert np.isclose(pm.lpm(RETURNS, target=0.0, n=1), 0.0325)


def test_upm_degree_one():
    # excess above 0: [0.10, 0, 0.02, 0]; mean = 0.12/4 = 0.03
    assert np.isclose(pm.upm(RETURNS, target=0.0, q=1), 0.03)


def test_lpm_upm_degree_two():
    # squared shortfalls: [0, 0.0025, 0, 0.0064]; mean = 0.0089/4 = 0.002225
    assert np.isclose(pm.lpm(RETURNS, target=0.0, n=2), 0.002225)
    # squared excess: [0.01, 0, 0.0004, 0]; mean = 0.0104/4 = 0.0026
    assert np.isclose(pm.upm(RETURNS, target=0.0, q=2), 0.0026)


def test_explanatory_ratio():
    # UPM/LPM at (q=1,n=1) = 0.03 / 0.0325 = 0.9230769...
    assert np.isclose(pm.explanatory_metric(RETURNS, target=0.0, q=1, n=1), 0.03 / 0.0325)


def test_double_benchmark_elementwise_max():
    # elementwise max of ([0.01,0.02],[0.005,0.03]) = [0.01, 0.03]
    target = pm.double_nonstationary_benchmark([0.01, 0.02], [0.005, 0.03])
    assert np.allclose(target, [0.01, 0.03])


def test_scalar_riskfree_broadcasts():
    # scalar rf broadcasts: max([0.01,-0.02], 0.0) = [0.01, 0.0]
    target = pm.double_nonstationary_benchmark([0.01, -0.02], 0.0)
    assert np.allclose(target, [0.01, 0.0])


def test_lag1_autocorr_perfect():
    # linear series -> lag-1 correlation is exactly 1.0
    assert np.isclose(pm.lag1_autocorr([1, 2, 3, 4, 5], rho_method="correlation"), 1.0)


def test_predictive_zeroed_by_full_autocorr():
    # ρ=1 => (1-|ρ|)=0 => predictive metric is 0 regardless of the ratio (Eq 7 logic).
    # Use RETURNS (has downside, so LPM>0 and the ratio is well-defined).
    predictive = pm.predictive_metric(RETURNS, target=0.0, q=1, n=1, rho=1.0)
    assert np.isclose(predictive, 0.0)


def test_predictive_equals_ratio_when_rho_zero():
    # ρ=0 => predictive == explanatory ratio (no discount)
    ratio = pm.explanatory_metric(RETURNS, target=0.0, q=1, n=1)
    predictive = pm.predictive_metric(RETURNS, target=0.0, q=1, n=1, rho=0.0)
    assert np.isclose(predictive, ratio)


def test_nan_fails_loud():
    # NaN must raise, never silently drop (Rule 12)
    try:
        pm.lpm([0.1, np.nan, 0.2], target=0.0, n=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError on NaN input")


def test_zero_lpm_fails_loud():
    # all returns above target => LPM=0 => ratio undefined => must raise
    try:
        pm.explanatory_metric([0.1, 0.2, 0.3], target=0.0, q=1, n=1)
    except ZeroDivisionError:
        return
    raise AssertionError("expected ZeroDivisionError when LPM=0")


def test_covariance_vs_correlation_differ():
    # the two rho_methods should generally give different numbers (the RQ2 ambiguity)
    series = [0.10, -0.05, 0.02, -0.08, 0.03, -0.01]
    corr = pm.lag1_autocorr(series, rho_method="correlation")
    cov = pm.lag1_autocorr(series, rho_method="covariance")
    assert not np.isclose(corr, cov)


if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    passed = 0
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} tests passed")
