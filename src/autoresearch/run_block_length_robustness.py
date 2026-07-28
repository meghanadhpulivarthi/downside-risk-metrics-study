"""
Block-length robustness for the VaR partial-correlation increment (referee M7).

The paper's paired block-bootstrap CIs use length-3 blocks over the per-date series.
A reviewer asked whether the crypto conclusions survive a different block length, since
the crypto beds have only 7-9 formation dates (~3 effective blocks at length 3). Here we
recompute the 95% CI of the mean VaR partial rho under block lengths {1,2,3,4} and a
stationary (geometric-block) bootstrap with expected length 3, reusing the SAME per-date
series machinery as run_robustness_fdr.py so the point estimates match Table 2.

Run: uv run python -u src/autoresearch/run_block_length_robustness.py
"""

# Config -- edit these directly
METRIC = "var5"            # the flagship magnitude increment the abstract leans on
BLOCK_LENGTHS = [1, 2, 3, 4]
STATIONARY_MEAN_BLOCK = 3  # expected block length for the stationary bootstrap
N_BOOT = 3000
SEED = 0

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import evaluate as ev
from run_robustness_fdr import per_date_series, build_beds

OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"


def fixed_block_ci(series, block, n_boot=N_BOOT, seed=SEED):
    """95% CI of the mean via a moving-block bootstrap with fixed block length."""
    d = np.asarray(series, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        # loud: a bed too thin to bootstrap should not be silently returned as a number
        print(f"    block={block}: n={n} < 3, CI not estimable")
        return None
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = max(1, n - block + 1)
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max, n_blocks)
        means[i] = np.concatenate([d[s:s + block] for s in starts])[:n].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stationary_block_ci(series, mean_block=STATIONARY_MEAN_BLOCK, n_boot=N_BOOT, seed=SEED):
    """95% CI via the Politis-Romano stationary bootstrap (geometric block lengths)."""
    d = np.asarray(series, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        print(f"    stationary: n={n} < 3, CI not estimable")
        return None
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block  # per-step restart probability -> expected block length = mean_block
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.empty(n, dtype=int)
        pos = rng.integers(0, n)
        for t in range(n):
            idx[t] = pos
            if rng.random() < p:
                pos = rng.integers(0, n)
            else:
                pos = (pos + 1) % n
        means[i] = d[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    now = datetime.datetime.now()
    print("=" * 74)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : metric={METRIC}; blocks={BLOCK_LENGTHS}; "
          f"stationary E[block]={STATIONARY_MEAN_BLOCK}; n_boot={N_BOOT}")
    print("=" * 74)
    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_block_length_robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    beds = build_beds()
    print(f"Loaded {len(beds)} beds")
    results = {}
    for name, (close, mkt, dates, univ) in beds.items():
        print(f"\n[{name}] collecting per-date VaR partial series ...")
        partial, _ = per_date_series(close, mkt, dates, univ)
        series = [x for x in partial[METRIC] if np.isfinite(x)]
        n = len(series)
        mean = float(np.mean(series)) if n else float("nan")
        print(f"    n_dates={n}  mean partial rho={mean:+.3f}")
        cell = {"n_dates": n, "mean": round(mean, 4), "ci_by_block": {}}
        for block in BLOCK_LENGTHS:
            ci = fixed_block_ci(series, block)
            if ci is not None:
                cell["ci_by_block"][str(block)] = [round(ci[0], 4), round(ci[1], 4)]
                print(f"    block={block}: 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        ci_stat = stationary_block_ci(series)
        if ci_stat is not None:
            cell["ci_stationary"] = [round(ci_stat[0], 4), round(ci_stat[1], 4)]
            print(f"    stationary : 95% CI [{ci_stat[0]:+.3f}, {ci_stat[1]:+.3f}]")
        results[name] = cell

    config = {"metric": METRIC, "block_lengths": BLOCK_LENGTHS,
              "stationary_mean_block": STATIONARY_MEAN_BLOCK, "n_boot": N_BOOT, "seed": SEED}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(output_dir / "block_length_robustness.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nConfig saved : {output_dir / 'config.json'}")
    print(f"Results saved: {output_dir / 'block_length_robustness.json'}")


if __name__ == "__main__":
    main()
