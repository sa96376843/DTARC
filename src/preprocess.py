"""Create leakage-safe price-volume features and forward-return labels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / (down + 1e-12))


def per_stock(g):
    g = g.sort_values("date").copy()
    c, v = g["close"], g["volume"].clip(lower=1)
    ret = c.pct_change()
    g["ret_1"] = ret
    for k in [2, 3, 5, 10, 20]:
        g[f"mom_{k}"] = c.pct_change(k)
    for k in [5, 10, 20]:
        g[f"vol_{k}"] = ret.rolling(k).std()
        g[f"sma_ratio_{k}"] = c / c.rolling(k).mean() - 1
        g[f"volume_z_{k}"] = (np.log(v) - np.log(v).rolling(k).mean()) / (np.log(v).rolling(k).std() + 1e-8)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    g["macd_norm"] = macd / c
    g["macd_signal_norm"] = macd.ewm(span=9, adjust=False).mean() / c
    g["rsi_14"] = rsi(c) / 100
    g["range_hl"] = (g["high"] - g["low"]) / c
    g["close_position"] = (c - g["low"]) / (g["high"] - g["low"] + 1e-12)
    g["volume_change"] = np.log(v).diff()
    g["amihud_20"] = (ret.abs() / (g["amount"] if "amount" in g else c*v)).rolling(20).mean()
    # Cross-sectional, same-day context is known at prediction time.
    g["market_ret_1"] = ret.groupby(g["date"]).transform("mean")
    g["market_mom_5"] = g["mom_5"].groupby(g["date"]).transform("mean")
    g["relative_ret_5"] = g["mom_5"] - g["market_mom_5"]
    for h in [1, 5]:
        g[f"fwd_ret_{h}"] = c.shift(-h) / c - 1
        g[f"y_{h}"] = (g[f"fwd_ret_{h}"] > 0).astype(float)
        g.loc[g[f"fwd_ret_{h}"].isna(), f"y_{h}"] = np.nan
    return g


def main():
    cn = pd.read_csv(DATA / "cn_raw_baostock.csv", parse_dates=["date"])
    th = pd.read_csv(DATA / "th_raw_siamchart.csv", parse_dates=["date"])
    frames = []
    for market, raw in [("CN", cn), ("TH", th)]:
        raw = raw[(raw.date >= "2023-11-13") & (raw.date <= "2026-09-30")].copy()
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in raw:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw["volume"] = raw.groupby("code")["volume"].transform(lambda x: x.ffill().bfill())
        d = pd.concat([per_stock(g) for _, g in raw.groupby("code")], ignore_index=True)
        if "code" not in d:
            raise RuntimeError("Missing security identifier")
        d["market"] = market
        d = d[d.date >= "2023-12-11"].copy()
        d["split"] = np.select([d.date <= "2024-12-31", d.date <= "2025-12-31"], ["train", "validation"], default="test")
        d.to_csv(DATA / f"{market.lower()}_processed.csv", index=False)
        frames.append(d)
    all_d = pd.concat(frames, ignore_index=True)
    features = [c for c in all_d.columns if c.startswith(("ret_", "mom_", "vol_", "sma_", "volume_", "macd_", "rsi_", "range_", "close_position", "amihud", "market_", "relative_"))]
    features = [c for c in features if c not in ["volume"]]
    summary = {}
    for market, d in all_d.groupby("market"):
        summary[market] = {
            "stocks": int(d.code.nunique()), "start": str(d.date.min().date()), "end": str(d.date.max().date()),
            "observations": int(len(d)), "features": len(features),
            "splits": {k: int(v) for k, v in d.split.value_counts().items()},
            "up_rate_h5": float(d.y_5.mean()),
        }
    (DATA / "dataset_summary.json").write_text(json.dumps({"features": features, "markets": summary}, indent=2))
    print(json.dumps({"features": features, "markets": summary}, indent=2))

if __name__ == "__main__":
    main()
