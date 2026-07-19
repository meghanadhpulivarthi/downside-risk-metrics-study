"""
Economic backtest — does the risk ranking matter net of costs? (Appendix B)

The paper's skill measure is a rank correlation, not P&L. This experiment gives the
decision-relevant complement: at each formation date we sort the universe into risk
terciles by a metric, hold the LOW-risk tercile, avoid the HIGH-risk tercile, and measure

  (a) defensive return spread  = mean forward return(low-risk) - mean forward return(high-risk)
  (b) drawdown discrimination  = mean forward drawdown(high-risk) - mean forward drawdown(low-risk)

gross and NET of a turnover-based transaction cost, for two sorting metrics:
  volatility   vs   the pre-registered composite  z(vol)+z(VaR).

A better risk metric flags a high-risk tercile that realizes larger drawdowns and worse
returns. The question is whether the composite improves on volatility once trading costs
on the extra turnover are paid. Forward return uses delisting = -100%.

Run: uv run --project <repo> python -u src/autoresearch/run_economic_backtest.py
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
from run_mechanism import build_beds, MIN_PAIRS

# Config — edit these directly
N_GROUPS = 3          # terciles (robust for the smaller crypto universes)
COST_BPS = 20.0       # transaction cost in bps applied to portfolio turnover per rebalance
DEATH_GAP_DAYS = 14   # delisting detection gap
OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"


def zscore(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.full_like(x, np.nan)
    return (x - np.nanmean(x)) / sd


def forward_return(price_window, window_end):
    """Simple forward return; delisting mid-window => total loss (-1.0)."""
    clean = price_window.dropna()
    if len(clean) < 2:
        return np.nan
    if clean.index.max() < pd.Timestamp(window_end) - pd.Timedelta(days=DEATH_GAP_DAYS):
        return -1.0
    return float(clean.iloc[-1] / clean.iloc[0] - 1.0)


def backtest_bed(close, market_ret, formation_dates, universe_fn):
    """Per-date tercile spreads for volatility and composite, with turnover."""
    metrics = ["volatility", "composite"]
    rec = {m: {"spread": [], "dd_disc": [], "turnover": [],
               "ret_low": [], "ret_high": [], "dd_low": [], "dd_high": []} for m in metrics}
    prev_low = {m: set() for m in metrics}

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
            end = fdate + pd.Timedelta(days=FORWARD_DAYS)
            fwd = close.loc[fdate:end, asset]
            fr = forward_return(fwd, end)
            dd = lt.forward_drawdown(fwd, end)
            if np.isnan(fr) or np.isnan(dd):
                continue
            rows.append({"asset": asset, "vol": feats["volatility"], "var5": feats["var5"],
                         "fr": fr, "dd": dd})
        g = pd.DataFrame(rows)
        if len(g) < MIN_PAIRS:
            print(f"    skip {fdate.date()}: {len(g)} assets (<{MIN_PAIRS})")
            continue
        g["composite"] = zscore(g["vol"].values) + zscore(g["var5"].values)

        for metric in metrics:
            col = "vol" if metric == "volatility" else "composite"
            ranks = g[col].rank(method="first")
            grp = np.ceil(ranks / len(g) * N_GROUPS).astype(int)  # 1=low risk ... N=high risk
            low = g[grp == 1]
            high = g[grp == N_GROUPS]
            if len(low) < 3 or len(high) < 3:
                print(f"    {fdate.date()} {metric}: thin group (low={len(low)}, high={len(high)}) — skip date")
                continue
            low_set = set(low["asset"])
            turnover = 1.0 if not prev_low[metric] else len(low_set ^ prev_low[metric]) / max(len(low_set), 1)
            prev_low[metric] = low_set
            rec[metric]["ret_low"].append(low["fr"].mean())
            rec[metric]["ret_high"].append(high["fr"].mean())
            rec[metric]["dd_low"].append(low["dd"].mean())
            rec[metric]["dd_high"].append(high["dd"].mean())
            rec[metric]["spread"].append(low["fr"].mean() - high["fr"].mean())
            rec[metric]["dd_disc"].append(high["dd"].mean() - low["dd"].mean())
            rec[metric]["turnover"].append(turnover)

    out = {}
    for metric in metrics:
        spread = np.array(rec[metric]["spread"], dtype=float)
        turn = np.array(rec[metric]["turnover"], dtype=float)
        cost = turn * COST_BPS / 1e4
        net = spread - cost  # cost charged once against the long (held) leg's contribution
        n = len(spread)
        out[metric] = {
            "n_periods": n,
            "gross_spread_mean": round(float(np.mean(spread)), 4) if n else None,
            "net_spread_mean": round(float(np.mean(net)), 4) if n else None,
            "spread_ir": round(float(np.mean(spread) / np.std(spread)), 3) if n and np.std(spread) > 0 else None,
            "dd_discrimination": round(float(np.mean(rec[metric]["dd_disc"])), 4) if n else None,
            "mean_turnover": round(float(np.mean(turn)), 3) if n else None,
            "ret_low_mean": round(float(np.mean(rec[metric]["ret_low"])), 4) if n else None,
            "ret_high_mean": round(float(np.mean(rec[metric]["ret_high"])), 4) if n else None,
        }
    return out


def main():
    now = datetime.datetime.now()
    print("=" * 74)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : terciles N={N_GROUPS}, cost={COST_BPS}bps, hold={FORWARD_DAYS}d")
    print("=" * 74)
    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_economic_backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    beds = build_beds()
    results = {}
    for name, (close, mkt, dates, univ) in beds.items():
        print(f"\n[{name}]")
        results[name] = backtest_bed(close, mkt, dates, univ)

    print("\n" + "=" * 74)
    print("Low-risk minus high-risk tercile: forward-return spread & drawdown discrimination")
    print("=" * 74)
    print(f"{'bed':22s} {'metric':11s} {'net spread':>11s} {'IR':>6s} {'dd_disc':>9s} {'turnover':>9s}")
    for name, res in results.items():
        for metric in ["volatility", "composite"]:
            r = res[metric]
            print(f"{name:22s} {metric:11s} {r['net_spread_mean']:>11} {str(r['spread_ir']):>6} "
                  f"{r['dd_discrimination']:>9} {r['mean_turnover']:>9}")

    print("\nDoes the composite beat volatility economically (net spread & dd discrimination)?")
    for name, res in results.items():
        v, c = res["volatility"], res["composite"]
        better = (c["net_spread_mean"] > v["net_spread_mean"]) and (c["dd_discrimination"] > v["dd_discrimination"])
        print(f"  {name:22s}: composite {'BETTER' if better else 'not better'} "
              f"(net spread {v['net_spread_mean']:+.4f}->{c['net_spread_mean']:+.4f}, "
              f"dd_disc {v['dd_discrimination']:+.4f}->{c['dd_discrimination']:+.4f})")

    with open(output_dir / "economic_backtest.json", "w") as h:
        json.dump(results, h, indent=2)
    print(f"\nResults saved: {output_dir / 'economic_backtest.json'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
