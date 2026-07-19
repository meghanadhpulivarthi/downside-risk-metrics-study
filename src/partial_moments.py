"""
Partial-moment metrics from Viole & Nawrocki (2016), Eqs 1-7.

Implemented directly from the PDF (references/), NOT the prose summary, so the
audit stands on the paper's own definitions. Where the paper is ambiguous, the
ambiguity is exposed as an explicit switch (never silently resolved) — because
those ambiguities are exactly what RQ2 audits. See context/gotchas.md.

Equations:
  (1) LPM(n,h,i) = (1/T) Σ_t max(0, h - R_it)^n
  (2) UPM(q,l,i) = (1/T) Σ_t max(0, R_it - l)^q      [paper assumes h = l]
  (4) EXPLANATORY = UPM(q,y,x) / LPM(n,y,x)           [y = double benchmark]
  (5) ρ(x) = Cov(x_t, x_{t-1})   -- but see rho_method note below
  (7) PREDICTIVE = (UPM/LPM) * (1 - |ρ(x)|)

Convention: returns/targets are 1-D array-likes aligned in time. The target may
be a scalar or a same-length array (the double non-stationary benchmark).
Inputs must be finite — NaNs raise rather than being silently dropped, so the
caller cleans upstream and coverage gaps stay visible (Rule 12, fail loud).
"""

import numpy as np

# Degree presets — read as q/n ratios with n fixed at 1 (see context/gotchas.md).
# Confirm against the paper's Tables 1-6 before trusting for exact replication.
INVESTOR_DEGREES = {
    "risk_averse":     {"q": 0.25, "n": 1.0},
    "prospect_theory": {"q": 0.44, "n": 1.0},  # ~ 1 / 2.25 Kahneman–Tversky loss aversion
    "risk_neutral":    {"q": 1.0,  "n": 1.0},
    "risk_seeking":    {"q": 2.0,  "n": 1.0},
}


def _as_finite_array(values, name):
    """Coerce to a 1-D float array and fail loud on NaN/inf (no silent drops)."""
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{name}: empty input")
    if not np.all(np.isfinite(array)):
        n_bad = int(np.sum(~np.isfinite(array)))
        raise ValueError(
            f"{name}: {n_bad} non-finite value(s). Clean upstream — "
            f"partial moments must not silently drop observations."
        )
    return array


def _align_target(target, length):
    """Return target as a length-`length` array (scalar broadcast or checked)."""
    target_array = np.asarray(target, dtype=float).ravel()
    if target_array.size == 1:
        return np.full(length, target_array.item())
    if target_array.size != length:
        raise ValueError(
            f"target length {target_array.size} != returns length {length}; "
            f"the double non-stationary benchmark must align in time with returns."
        )
    return target_array


def lpm(returns, target, n):
    """Eq 1 — Lower Partial Moment. Below-target shortfall, raised to degree n."""
    returns_array = _as_finite_array(returns, "returns")
    target_array = _align_target(_as_finite_array(target, "target"), returns_array.size)
    shortfall = np.maximum(0.0, target_array - returns_array)
    return float(np.mean(shortfall ** n))


def upm(returns, target, q):
    """Eq 2 — Upper Partial Moment. Above-target excess, raised to degree q."""
    returns_array = _as_finite_array(returns, "returns")
    target_array = _align_target(_as_finite_array(target, "target"), returns_array.size)
    excess = np.maximum(0.0, returns_array - target_array)
    return float(np.mean(excess ** q))


def double_nonstationary_benchmark(benchmark_returns, riskfree_returns):
    """
    Target at each t = max(benchmark_t, riskfree_t), elementwise (p.905).
    'The greater of the two observations at time t will be the target.'
    """
    benchmark_array = _as_finite_array(benchmark_returns, "benchmark_returns")
    riskfree_array = _align_target(
        _as_finite_array(riskfree_returns, "riskfree_returns"), benchmark_array.size
    )
    return np.maximum(benchmark_array, riskfree_array)


def explanatory_metric(returns, target, q, n):
    """Eq 4 — UPM(q,y,x) / LPM(n,y,x). The historical up/down asymmetry ratio."""
    lower = lpm(returns, target, n)
    upper = upm(returns, target, q)
    if lower == 0.0:
        # No below-target observations => ratio undefined (not zero risk).
        # Fail loud: caller decides how to rank an asset with no downside history.
        raise ZeroDivisionError(
            "LPM = 0 (no below-target returns in window); explanatory ratio "
            "undefined. Handle at the ranking layer, do not treat as 0 or inf silently."
        )
    return upper / lower


def lag1_autocorr(returns, rho_method="correlation"):
    """
    Eq 5/6 serial-dependence term ρ(x).

    The paper WRITES Cov(x_t, x_{t-1}) (Eq 5) but its logic — entry at ρ=0, full
    exit at |ρ|=1, the (1-|ρ|) discount — only holds for a bounded [-1,1]
    CORRELATION. This contradiction is audited in RQ2, so the choice is explicit:
      - "correlation": Pearson corr of (x_t, x_{t-1})  [coheres with the (1-|ρ|) form]
      - "covariance" : literal Eq 5 Cov(x_t, x_{t-1})  [as written; unbounded]
    """
    returns_array = _as_finite_array(returns, "returns")
    if returns_array.size < 3:
        raise ValueError("lag-1 autocorr needs >= 3 observations")
    current = returns_array[1:]
    lagged = returns_array[:-1]
    if rho_method == "correlation":
        return float(np.corrcoef(current, lagged)[0, 1])
    if rho_method == "covariance":
        # ddof=0 to match the (1/T) partial-moment convention in Eqs 1-2.
        return float(np.cov(current, lagged, ddof=0)[0, 1])
    raise ValueError(f"rho_method must be 'correlation' or 'covariance', got {rho_method!r}")


def predictive_metric(returns, target, q, n, rho=None, rho_method="correlation"):
    """
    Eq 7 — (UPM(q,y,x)/LPM(n,y,x)) * (1 - |ρ(x)|).

    rho: pass a precomputed ρ (e.g. estimated over the whole explanatory period
    per footnote 10) to keep replication faithful; if None, ρ is computed from
    `returns` here via `rho_method`.
    """
    ratio = explanatory_metric(returns, target, q, n)
    if rho is None:
        rho = lag1_autocorr(returns, rho_method=rho_method)
    return ratio * (1.0 - abs(rho))
