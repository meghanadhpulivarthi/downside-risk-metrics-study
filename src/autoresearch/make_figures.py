"""
Render the paper's figures from the saved experiment JSONs (no recomputation).

Sources (latest timestamped folder is auto-selected):
  outputs/*_mechanism/mechanism.json                 -> Fig1 (forest), Fig2 (chain), Fig6
  outputs/*_mechanism_depth/mechanism_depth.json     -> Fig3 (mediation), Fig4 (horizon), Fig6
  outputs/*_directed_sweep/sweep_result.json         -> Fig5 (best-of-N)

Writes PNGs to writeup/figures/.

Run: uv run --project <repo> python -u src/autoresearch/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"
FIGDIR = PROJECT_ROOT / "writeup" / "figures"

BEDS = ["crypto_2016_calm", "crypto_2017_bull", "crypto_2018_crash",
        "equity_1994_99_calm", "equity_2005_09_gfc"]
BED_LABEL = {"crypto_2016_calm": "crypto '16", "crypto_2017_bull": "crypto '17",
             "crypto_2018_crash": "crypto '18", "equity_1994_99_calm": "equity 94-99",
             "equity_2005_09_gfc": "equity 05-09"}
IS_CRYPTO = {b: b.startswith("crypto") for b in BEDS}

# metric groups for the forest plot (top = magnitude, bottom = shape)
MAGNITUDE = [("var5", "VaR"), ("downside_dev", "downside dev"), ("es5", "ES")]
SHAPE = [("down_semibeta", "semibeta"), ("down_beta", "downside beta"),
         ("gda_voldown", "GDA vol-down"), ("ltd_crash", "lower-tail dep"),
         ("hill_tail", "Hill tail"), ("vn_ratio", "VN UPM/LPM")]


def latest(suffix, fname):
    dirs = sorted(p for p in OUTPUTS.glob(f"*_{suffix}") if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No outputs/*_{suffix} folder found for {fname}")
    path = dirs[-1] / fname
    print(f"  using {path}")
    return json.loads(path.read_text())


def xerr(mean, ci):
    """Non-negative [low, high] error distances from the point estimate."""
    return [max(0.0, mean - ci[0])], [max(0.0, ci[1] - mean)]


# ---------------- Fig 1: partial-correlation forest (magnitude vs shape) ----------------
def fig_forest(mech):
    metrics = MAGNITUDE + SHAPE
    fig, axes = plt.subplots(1, 5, figsize=(13, 4.6), sharey=True)
    ylabels = [lab for _, lab in metrics]
    ypos = list(range(len(metrics)))[::-1]  # first metric at top
    split_y = (ypos[len(MAGNITUDE) - 1] + ypos[len(MAGNITUDE)]) / 2  # between groups

    for ax, bed in zip(axes, BEDS):
        part = mech[bed]["partial"]
        for (key, _), y in zip(metrics, ypos):
            rec = part.get(key)
            if rec is None or not np.isfinite(rec["mean"]):
                continue
            color = "#1f77b4" if (key, _) in MAGNITUDE else "#d62728"
            lo, hi = xerr(rec["mean"], rec["ci"])
            ax.errorbar(rec["mean"], y, xerr=[lo, hi], fmt="o", ms=4,
                        color=color, ecolor=color, capsize=2, lw=1.2)
        ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.axhline(split_y, color="gray", lw=0.6, ls=":")
        ax.set_title(BED_LABEL[bed], fontsize=10)
        ax.set_xlim(-0.4, 0.4)
        ax.tick_params(labelsize=8)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(ylabels, fontsize=8)
    fig.suptitle("Partial Spearman rho(metric, forward drawdown | volatility)  "
                 "— blue = tail magnitude, red = tail shape", fontsize=11)
    fig.text(0.5, 0.005, "incremental rank correlation beyond volatility (95% block-bootstrap CI)",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save(fig, "fig1_partial_forest.png")


# ---------------- Fig 2: persistence chain ----------------
def fig_chain(mech):
    labels = [BED_LABEL[b] for b in BEDS]
    tv_fv = [mech[b]["chain"]["chain_tvol_fvol"]["mean"] for b in BEDS]
    fv_dd = [mech[b]["chain"]["chain_fvol_dd"]["mean"] for b in BEDS]
    base = [mech[b]["raw"]["volatility"]["mean"] for b in BEDS]
    x = np.arange(len(BEDS)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - w, tv_fv, w, label="trailing vol -> forward vol", color="#1f77b4")
    ax.bar(x, fv_dd, w, label="forward vol -> drawdown", color="#2ca02c")
    ax.bar(x + w, base, w, label="trailing vol -> drawdown (baseline)", color="#7f7f7f")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mean per-date Spearman"); ax.set_ylim(0, 1)
    ax.set_title("The persistence chain: why volatility forecasts drawdown")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig2_persistence_chain.png")


# ---------------- Fig 3: mediation is a same-window artifact ----------------
def fig_mediation(med):
    """med = mediation_nonoverlap.json with 'overlap' and 'nonoverlap' designs."""
    labels = [BED_LABEL[b] for b in BEDS]
    ov = [med["overlap"][b]["trailing_given_forward"]["mean"] for b in BEDS]
    no = [med["nonoverlap"][b]["trailing_given_forward"]["mean"] for b in BEDS]
    e_ov = np.array([xerr(med["overlap"][b]["trailing_given_forward"]["mean"],
                          med["overlap"][b]["trailing_given_forward"]["ci"]) for b in BEDS]).squeeze(-1).T
    e_no = np.array([xerr(med["nonoverlap"][b]["trailing_given_forward"]["mean"],
                          med["nonoverlap"][b]["trailing_given_forward"]["ci"]) for b in BEDS]).squeeze(-1).T
    x = np.arange(len(BEDS)); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - w / 2, ov, w, yerr=e_ov, capsize=3,
           label="same-window (mechanical): vol & drawdown on [t, t+90]", color="#c7c7c7")
    ax.bar(x + w / 2, no, w, yerr=e_no, capsize=3,
           label="predictive (disjoint): vol on [t, t+30], drawdown on [t+30, t+120]", color="#ff7f0e")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("partial ρ(trailing vol, drawdown | forward vol)")
    ax.set_title("The equity 'full mediation' is a same-window artifact, not a sufficient statistic")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig3_mediation.png")


# ---------------- Fig 4: horizon co-movement ----------------
def fig_horizon(depth):
    horizons = sorted((int(h) for h in depth[BEDS[0]]["horizon"]), key=int)
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.4), sharey=True)
    for ax, bed in zip(axes, BEDS):
        hz = depth[bed]["horizon"]
        persist = [hz[str(h)]["persistence"]["mean"] for h in horizons]
        skill = [hz[str(h)]["skill"]["mean"] for h in horizons]
        ax.plot(horizons, persist, "o-", color="#1f77b4", ms=3, label="vol persistence")
        ax.plot(horizons, skill, "s-", color="#d62728", ms=3, label="drawdown skill")
        ax.set_title(BED_LABEL[bed], fontsize=10)
        ax.set_xlabel("horizon (days)", fontsize=8)
        ax.set_ylim(0, 1); ax.tick_params(labelsize=8)
    axes[0].set_ylabel("mean Spearman")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Drawdown skill tracks volatility persistence across horizons", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig4_horizon_decay.png")


# ---------------- Fig 5: best-of-N (data snooping) ----------------
def fig_best_of_n(sweep):
    pts = sweep["best_of_N"]
    N = [p["N"] for p in pts]
    val = [p["val"] for p in pts]
    test = [p["test"] for p in pts]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(N, val, "o-", color="#1f77b4", label="best validation score")
    ax.plot(N, test, "s-", color="#d62728", label="its locked-test score")
    ax.axhline(sweep["vol_test"], color="k", ls="--", lw=1,
               label=f"volatility baseline (test {sweep['vol_test']:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("search intensity N (configs tried)")
    ax.set_ylabel("mean per-date Spearman")
    ax.set_title("Automated search inflates validation, not the locked test")
    ax.legend(fontsize=8, loc="center right")
    ax.text(0.03, 0.05,
            f"val-selected winners beat vol on test only {sweep['val_winners_beating_vol_on_test']*100:.0f}%"
            f"\nval-test corr = {sweep['val_test_corr']:.2f}",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", fc="#f0f0f0", ec="gray"))
    fig.tight_layout()
    _save(fig, "fig5_best_of_N.png")


# ---------------- Fig 6: increment scales with tail fatness ----------------
def fig_increment(mech, depth):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for bed in BEDS:
        x = depth[bed]["heterogeneity"]["blowup_rate"]
        y = mech[bed]["partial"]["var5"]["mean"]
        kurt = depth[bed]["heterogeneity"]["mean_trailing_kurt"]
        color = "#1f77b4" if IS_CRYPTO[bed] else "#d62728"
        ax.scatter(x, y, s=60, color=color, zorder=3)
        ax.annotate(f"{BED_LABEL[bed]}\n(kurt {kurt:g})", (x, y),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("blow-up rate  (fraction of asset-dates with drawdown >= 80%)")
    ax.set_ylabel("VaR incremental signal:  partial rho(VaR, dd | vol)")
    ax.set_title("The magnitude increment is larger in fat-tailed, blow-up-prone beds")
    ax.text(0.97, 0.05, "blue = crypto,  red = equity  (n=5, descriptive)",
            transform=ax.transAxes, ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    _save(fig, "fig6_increment_vs_tailfat.png")


# ---------------- Fig 7: economic drawdown discrimination ----------------
def fig_economic(econ):
    """Grouped bars: realized drawdown discrimination (high- minus low-risk tercile),
    volatility vs composite, per bed."""
    labels = [BED_LABEL[b] for b in BEDS]
    vol = [econ[b]["volatility"]["dd_discrimination"] for b in BEDS]
    comp = [econ[b]["composite"]["dd_discrimination"] for b in BEDS]
    x = np.arange(len(BEDS)); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - w / 2, vol, w, label="sorted by volatility", color="#7f7f7f")
    ax.bar(x + w / 2, comp, w, label="sorted by z(vol)+z(VaR) composite", color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("drawdown discrimination\n(high- minus low-risk tercile, forward DD)")
    ax.set_title("Volatility already separates realized drawdown; the composite adds little")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig7_economic.png")


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    print("Loading experiment JSONs ...")
    mech = latest("mechanism", "mechanism.json")
    depth = latest("mechanism_depth", "mechanism_depth.json")
    med = latest("mediation_nonoverlap", "mediation_nonoverlap.json")
    sweep = latest("directed_sweep", "sweep_result.json")
    econ = latest("economic_backtest", "economic_backtest.json")
    print("Rendering figures ...")
    fig_forest(mech)
    fig_chain(mech)
    fig_mediation(med)
    fig_horizon(depth)
    fig_best_of_n(sweep)
    fig_increment(mech, depth)
    fig_economic(econ)
    print(f"Done -> {FIGDIR}")


if __name__ == "__main__":
    main()
