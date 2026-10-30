"""Paired moving-block bootstrap for frozen out-of-sample predictions.

Dates, rather than individual rows, are resampled to preserve cross-sectional
dependence. Five-day moving blocks additionally reduce the effect of overlap in
five-day forward labels. Seeds remain paired throughout every replicate.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score, brier_score_loss, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def metric_values(y, p):
    yh = (p >= .5).astype(int)
    return {"mcc": matthews_corrcoef(y, yh),
            "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
            "brier": brier_score_loss(y, p),
            "accuracy": accuracy_score(y, yh)}


def moving_block_dates(dates, block, rng):
    n = len(dates)
    starts = rng.integers(0, max(1, n - block + 1), size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]


def prepare(pred, market, seed, methods):
    g = pred[(pred.market == market) & (pred.seed == seed)]
    base = g[g.method == methods[0]][["date", "code", "y"]].copy()
    for method in methods:
        p = g[g.method == method][["date", "code", "p"]].rename(columns={"p": method})
        base = base.merge(p, on=["date", "code"], how="inner", validate="one_to_one")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()
    pred = pd.read_csv(OUT / "predictions_h5.csv", parse_dates=["date"])
    rng = np.random.default_rng(args.seed)
    rows = []
    for market in ["CN", "TH"]:
        for comparator in ["static", "adwinER", "periodicER", "phDTARC"]:
            methods = ["adwinDTARC", comparator]
            paired = {s: prepare(pred, market, s, methods) for s in sorted(pred.seed.unique())}
            dates = {s: np.sort(d.date.unique()) for s, d in paired.items()}
            lookups = {}
            for s, d in paired.items():
                codes = d.date.map({x: i for i, x in enumerate(dates[s])}).values
                lookups[s] = {i: np.where(codes == i)[0] for i in range(len(dates[s]))}
            observed = []
            for s, d in paired.items():
                a, b = metric_values(d.y.values, d[methods[0]].values), metric_values(d.y.values, d[methods[1]].values)
                observed.append({m: a[m] - b[m] for m in a})
            obs = pd.DataFrame(observed).mean().to_dict()
            boot = {m: [] for m in obs}
            for _ in range(args.replicates):
                per_seed = {m: [] for m in obs}
                for s, d in paired.items():
                    sampled_positions = moving_block_dates(dates[s], args.block, rng)
                    # Repeated positions represent repeated bootstrap blocks.
                    idx = np.concatenate([lookups[s][i] for i in sampled_positions])
                    y = d.y.values[idx]
                    a = metric_values(y, d[methods[0]].values[idx])
                    b = metric_values(y, d[methods[1]].values[idx])
                    for m in obs:
                        per_seed[m].append(a[m] - b[m])
                for m in obs:
                    boot[m].append(np.nanmean(per_seed[m]))
            for metric, vals in boot.items():
                vals = np.asarray(vals)
                lo, hi = np.nanpercentile(vals, [2.5, 97.5])
                p_two = min(1.0, 2 * min((np.sum(vals <= 0) + 1) / (len(vals) + 1),
                                         (np.sum(vals >= 0) + 1) / (len(vals) + 1)))
                rows.append({"market": market, "comparison": f"adwinDTARC - {comparator}",
                             "metric": metric, "mean_difference": obs[metric],
                             "ci_2.5": lo, "ci_97.5": hi, "bootstrap_p": p_two,
                             "replicates": args.replicates, "block_days": args.block})
            print("bootstrap", market, comparator, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "paired_block_bootstrap_h5.csv", index=False)
    print(out.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
