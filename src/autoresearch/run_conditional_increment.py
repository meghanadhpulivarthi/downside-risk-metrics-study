"""
When does tail magnitude help? — conditional analysis (Appendix B)

The VaR increment beyond volatility is largest in fat-tailed crypto (§4.2). Here we test a
sharper conditional claim WITHIN asset classes: is the magnitude increment concentrated in
high-stress formation dates? For each date we record (i) the aggregate volatility of the
cross-section (median trailing vol), (ii) the VaR partial increment rho(VaR, dd | vol), and
(iii) the volatility skill rho(vol, dd). We pool dates by asset class (crypto: 24 dates;
equity: 120 dates), split each at its median aggregate volatility into LOW- vs HIGH-stress
regimes, and compare the mean VaR increment across regimes with a block-bootstrap CI.

Hypothesis: the increment is larger when aggregate volatility is high (tail magnitude carries
more information in stress).

Run: uv run --project <repo> python -u src/autoresearch/run_conditional_increment.py
"""

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import evaluate as ev
import left_tail as lt
from run_multi_testbed import compute_metrics, TRAILING_DAYS, FORWARD_DAYS
from run_multi_testbed_v2 import block_bootstrap_diff
from run_mechanism import build_beds, _spearman, _partial_spearman, MIN_PAIRS

OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"
GROUPS = {"crypto": ["crypto_2016_calm", "crypto_2017_bull", "crypto_2018_crash"],
          "equity": ["equity_1994_99_calm", "equity_2005_09_gfc"]}


def per_date_records(close, market_ret, formation_dates, universe_fn):
    """List of {agg_vol, var_partial, vol_skill} per usable formation date."""
    out = []
    for fdate in formation_dates:
        rows = []
        for asset in universe_fn(fdate):
            if asset not in close.columns:
                continue
            r = close.loc[fdate - pd.Timedelta(days=TRAILING_DAYS):fdate, asset].dropna().pct_change().dropna()
            if len(r) < 60:
                continue
            m = market_ret.reindex(r.index).fillna(0.0).values
            feats = compute_metrics(r.values, m)
            if feats is None:
                continue
            fwd = close.loc[fdate:fdate + pd.Timedelta(days=FORWARD_DAYS), asset]
            dd = lt.forward_drawdown(fwd, fdate + pd.Timedelta(days=FORWARD_DAYS))
            if np.isnan(dd):
                continue
            rows.append({"vol": feats["volatility"], "var5": feats["var5"], "dd": dd})
        g = pd.DataFrame(rows)
        if len(g) < MIN_PAIRS:
            print(f"    skip {fdate.date()}: {len(g)} assets")
            continue
        vol, var5, dd = g["vol"].values, g["var5"].values, g["dd"].values
        r_var_dd, r_var_vol, r_vol_dd = _spearman(var5, dd), _spearman(var5, vol), _spearman(vol, dd)
        out.append({"agg_vol": float(np.median(vol)),
                    "var_partial": _partial_spearman(r_var_dd, r_var_vol, r_vol_dd),
                    "vol_skill": r_vol_dd})
    return out


def summarize(records):
    df = pd.DataFrame(records).dropna(subset=["var_partial", "vol_skill", "agg_vol"])
    med = df["agg_vol"].median()
    low = df[df["agg_vol"] <= med]
    high = df[df["agg_vol"] > med]
    def agg(s):
        mean, lo, hi = block_bootstrap_diff(s.values)
        return {"mean": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)], "n": int(len(s))}
    # difference in VaR increment between high- and low-stress dates
    # pad to equal handling via block bootstrap on the two subsamples separately
    return {"n_dates": int(len(df)), "median_agg_vol": round(float(med), 4),
            "low_stress": {"var_partial": agg(low["var_partial"]), "vol_skill": agg(low["vol_skill"])},
            "high_stress": {"var_partial": agg(high["var_partial"]), "vol_skill": agg(high["vol_skill"])}}


def main():
    now = datetime.datetime.now()
    print("=" * 74)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print("=" * 74)
    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_conditional_increment"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    beds = build_beds()
    per_bed_records = {}
    for name, (close, mkt, dates, univ) in beds.items():
        print(f"[{name}] collecting per-date records ...")
        per_bed_records[name] = per_date_records(close, mkt, dates, univ)

    results = {}
    for group, members in GROUPS.items():
        pooled = [rec for name in members for rec in per_bed_records[name]]
        results[group] = summarize(pooled)

    print("\n" + "=" * 74)
    print("VaR partial increment rho(VaR, dd | vol), low- vs high-aggregate-volatility dates")
    print("=" * 74)
    for group, res in results.items():
        lo = res["low_stress"]["var_partial"]
        hi = res["high_stress"]["var_partial"]
        vlo = res["low_stress"]["vol_skill"]
        vhi = res["high_stress"]["vol_skill"]
        print(f"\n[{group}]  n_dates={res['n_dates']}  (median agg vol={res['median_agg_vol']})")
        print(f"    VaR increment : low-stress {lo['mean']:+.3f} CI{lo['ci']} (n={lo['n']})   "
              f"high-stress {hi['mean']:+.3f} CI{hi['ci']} (n={hi['n']})")
        print(f"    vol skill     : low-stress {vlo['mean']:+.3f}                    "
              f"high-stress {vhi['mean']:+.3f}")

    with open(output_dir / "conditional_increment.json", "w") as h:
        json.dump(results, h, indent=2)
    print(f"\nResults saved: {output_dir / 'conditional_increment.json'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
