"""
metric.py — THE AGENT-EDITED FILE (open-ended downside-risk metric).

Defines build_scores(panel) -> per-row score (higher = more predicted downside).
May be ANY Python: nonlinear, ratios, thresholds, regime conditioning, mixtures,
or a small model fit ON THE TRAIN SPLIT ONLY. Not restricted to linear.

Contract:
  - Input `panel` has the feature columns + 'split' + 'formation_date' + 'coin'.
    `fwd_drawdown` is present ONLY for train rows (withheld elsewhere) — never use it
    for val/test rows, never reference the 'test' split.
  - Return an array/Series aligned to panel's rows.
Set DESCRIPTION to a short, comma-free summary of the idea.
"""

import numpy as np

DESCRIPTION = "iter48 base plus small runup-times-smallcap bubble interaction tilt"


def build_scores(panel):
    feats = ["var5", "es10", "lpm2", "downside_dev"]
    weights = {"var5": 1.0, "es10": 1.0, "lpm2": 1.0, "downside_dev": 2.0}
    consensus = np.zeros(len(panel))
    total_weight = 0.0
    for feat in feats:
        consensus += weights[feat] * panel.groupby("formation_date")[feat].rank(pct=True).values
        total_weight += weights[feat]
    consensus /= total_weight

    runup_rank = panel.groupby("formation_date")["cum_ret"].rank(pct=True).values
    smallcap_rank = (-panel["log_mcap"]).groupby(panel["formation_date"]).rank(pct=True).values
    young_rank = (-panel["age_days"]).groupby(panel["formation_date"]).rank(pct=True).values

    boosted = consensus * (1.0 + 0.55 * runup_rank)
    young_smallcap = young_rank * smallcap_rank
    runup_smallcap = runup_rank * smallcap_rank
    return (0.73 * boosted + 0.09 * smallcap_rank + 0.07 * young_rank
            + 0.08 * young_smallcap + 0.03 * runup_smallcap)
