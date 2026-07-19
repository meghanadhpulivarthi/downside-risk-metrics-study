"""
Left-tail risk measures (Atilgan, Bali, Demirtas & Gunaydin 2020) + the
death-aware variants this project contributes.

Naive left-tail risk is estimated on surviving assets and its forecasting is
tested on assets that lived through the outcome window — so the assets that
actually died (the true left tail) never enter. The death-aware design (a) selects
the universe point-in-time and (b) scores assets that die in the outcome window at
their true terminal loss. See writeup/literature_review_post2016.md.
"""

import numpy as np


def expected_shortfall(returns, q=0.05):
    """ES_q: mean of returns at or below the q-quantile, sign-flipped so higher =
    more left-tail risk. The Atilgan et al. (2020) headline measure."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 20:
        return np.nan
    threshold = np.quantile(returns, q)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return np.nan
    return float(-tail.mean())


def value_at_risk(returns, q=0.05):
    """VaR_q: sign-flipped q-quantile of returns (higher = more left-tail risk)."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 20:
        return np.nan
    return float(-np.quantile(returns, q))


def realized_volatility(returns):
    """Plain return volatility over the window — the symmetric-risk baseline foil."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 20:
        return np.nan
    return float(np.std(returns))


def forward_drawdown(price_window, window_end, death_gap_days=14):
    """
    Realized max drawdown over a forward price window. DEATH-AWARE: if the asset's
    data terminates well before `window_end` (it delisted/stopped trading mid-window),
    the terminal loss dominates and drawdown is set to 1.0 (-100%). Otherwise it is
    the standard peak-to-trough drawdown (which already captures crash-to-dust).
    """
    import pandas as pd

    clean = price_window.dropna()
    if len(clean) < 2:
        return np.nan
    running_max = clean.cummax()
    drawdown = float(-(clean / running_max - 1.0).min())
    last_obs = clean.index.max()
    if last_obs < pd.Timestamp(window_end) - pd.Timedelta(days=death_gap_days):
        return 1.0  # delisted mid-window => terminal -100% loss
    return drawdown


def died_in_window(price_window, window_end, death_gap_days=14, dust_fraction=0.01):
    """Flag an asset as dead in the window if its data terminates early OR it ends
    below `dust_fraction` of its in-window peak (crashed to dust)."""
    import pandas as pd

    clean = price_window.dropna()
    if len(clean) < 2:
        return False
    terminated = clean.index.max() < pd.Timestamp(window_end) - pd.Timedelta(days=death_gap_days)
    to_dust = clean.iloc[-1] < dust_fraction * clean.max()
    return bool(terminated or to_dust)
