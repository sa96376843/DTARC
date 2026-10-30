"""Frozen-protocol ablation, horizon, and sensitivity experiments.

The primary h=5 test configuration is loaded from best_config.json and never
retuned on 2026 data. Sensitivity runs are diagnostics, not model selection.
"""
from pathlib import Path
import argparse, json
import pandas as pd
from experiment import (ROOT, DATA, OUT, initial_fit, run_online,
                        static_predictions, logistic_predictions, evaluate)

METHODS = ["static", "periodic", "periodicER", "adwinER", "phER",
           "adwinDTARC", "phDTARC"]


def load_all(h):
    info = json.loads((DATA / "dataset_summary.json").read_text())
    data = {m: pd.read_csv(DATA / f"{m.lower()}_processed.csv", parse_dates=["date"])
            for m in ["CN", "TH"]}
    cfg = json.loads((OUT / "best_config.json").read_text())["best"]
    return info["features"], data, cfg


def base_cache(data, features, h, seeds):
    return {(m, s): initial_fit(d[d.split == "train"].dropna(subset=[f"y_{h}"]),
                               features, h, s)
            for m, d in data.items() for s in seeds}


def save(pred, events, runtime, stem):
    pred = pd.concat(pred, ignore_index=True)
    pred.to_csv(OUT / f"predictions_{stem}.csv", index=False)
    evaluate(pred).to_csv(OUT / f"metrics_{stem}.csv", index=False)
    pd.DataFrame(runtime).to_csv(OUT / f"runtime_{stem}.csv", index=False)
    if events:
        pd.concat(events, ignore_index=True).to_csv(OUT / f"drift_events_{stem}.csv", index=False)
    print(evaluate(pred).groupby(["market", "method"])[
        ["accuracy", "macro_f1", "mcc", "auc", "brier"]].mean().round(4))


def run_ablation(seeds):
    h = 5
    features, data, cfg = load_all(h)
    bases = base_cache(data, features, h, seeds)
    preds, events, runtime = [], [], []
    variants = ["full", "no_anchor", "no_diversity", "no_typing", "no_proximal"]
    for market, d in data.items():
        for seed in seeds:
            base = bases[market, seed]
            for variant in variants:
                name = "adwinDTARC" if variant == "full" else f"DTARC_{variant}"
                p, e, r = run_online("adwinDTARC", d, base, features, h, seed,
                                     cfg, "test", variant)
                p["method"] = name
                preds.append(p)
                if len(e):
                    events.append(e.assign(market=market, seed=seed, method=name, h=h))
                runtime.append(r | {"market": market, "seed": seed,
                                    "method": name, "h": h, "variant": variant})
            print("ablation", market, seed, flush=True)
    save(preds, events, runtime, "ablation_h5")


def run_h1(seeds):
    h = 1
    features, data, cfg = load_all(h)
    bases = base_cache(data, features, h, seeds)
    preds, events, runtime = [], [], []
    for market, d in data.items():
        for seed in seeds:
            base = bases[market, seed]
            for method in METHODS:
                if method == "static":
                    p, e, r = static_predictions(method, d, base, features, h, seed, "test")
                else:
                    p, e, r = run_online(method, d, base, features, h, seed, cfg, "test")
                preds.append(p)
                if len(e):
                    events.append(e.assign(market=market, seed=seed, method=method, h=h))
                runtime.append(r | {"market": market, "seed": seed, "method": method, "h": h})
            preds.append(logistic_predictions(d, features, h, seed, "test"))
            print("horizon", market, seed, flush=True)
    save(preds, events, runtime, "h1")


def run_sensitivity(seeds):
    h = 5
    features, data, frozen = load_all(h)
    bases = base_cache(data, features, h, seeds)
    settings = []
    for factor, values, method in [
        ("buffer", [128, 512, 1024], "adwinDTARC"),
        ("replay", [16, 64, 128], "adwinDTARC"),
        ("online_lr", [0.00003, 0.0001, 0.0003], "adwinDTARC"),
        ("ph_threshold", [2.5, 5.0, 10.0], "phDTARC")]:
        for value in values:
            settings.append((factor, value, method, frozen | {factor: value}))
    preds, events, runtime = [], [], []
    for market, d in data.items():
        for seed in seeds:
            base = bases[market, seed]
            for factor, value, method, cfg in settings:
                p, e, r = run_online(method, d, base, features, h, seed, cfg, "test")
                p["factor"], p["value"] = factor, value
                preds.append(p)
                if len(e):
                    events.append(e.assign(market=market, seed=seed, method=method,
                                           h=h, factor=factor, value=value))
                runtime.append(r | {"market": market, "seed": seed, "method": method,
                                    "h": h, "factor": factor, "value": value})
            print("sensitivity", market, seed, flush=True)
    pred = pd.concat(preds, ignore_index=True)
    pred.to_csv(OUT / "predictions_sensitivity_h5.csv", index=False)
    metrics = evaluate(pred)
    # evaluate() intentionally has a fixed grouping contract; add factors back
    # by computing each run directly to avoid accidentally pooling settings.
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, brier_score_loss
    rows = []
    for keys, g in pred.groupby(["market", "split", "h", "method", "seed", "factor", "value"]):
        y, p = g.y.values, g.p.values
        yh = (p >= .5).astype(int)
        rows.append(dict(zip(["market", "split", "h", "method", "seed", "factor", "value"], keys)) | {
            "n": len(g), "accuracy": accuracy_score(y, yh),
            "macro_f1": f1_score(y, yh, average="macro"),
            "mcc": matthews_corrcoef(y, yh), "auc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p)})
    pd.DataFrame(rows).to_csv(OUT / "metrics_sensitivity_h5.csv", index=False)
    pd.DataFrame(runtime).to_csv(OUT / "runtime_sensitivity_h5.csv", index=False)
    if events:
        pd.concat(events, ignore_index=True).to_csv(OUT / "drift_events_sensitivity_h5.csv", index=False)
    print(pd.DataFrame(rows).groupby(["market", "factor", "value"])[
        ["mcc", "auc", "brier"]].mean().round(4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ablation", "h1", "sensitivity"], required=True)
    ap.add_argument("--seeds", default="11,22,33,44,55")
    args = ap.parse_args()
    seeds = list(map(int, args.seeds.split(",")))
    {"ablation": run_ablation, "h1": run_h1, "sensitivity": run_sensitivity}[args.mode](seeds)


if __name__ == "__main__":
    main()
