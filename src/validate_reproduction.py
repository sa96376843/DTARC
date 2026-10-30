"""Independently validate bundled panels, predictions, metrics, and figure files."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "results"

REQUIRED = [
    "cn_raw_sina_qianfuquan.csv", "cn_raw_sina_houfuquan.csv", "cn_raw_baostock.csv",
    "th_raw_siamchart.csv", "cn_processed.csv", "th_processed.csv",
    "cn_processed_exec.csv", "th_processed_exec.csv", "cn_processed_loo.csv", "th_processed_loo.csv",
    "dataset_summary.json",
]
RESULT_REQUIRED = [
    "best_config.json", "predictions_h5.csv", "metrics_h5.csv", "runtime_h5.csv",
    "metrics_ablation_h5.csv", "metrics_h1.csv", "metrics_matched_replay.csv",
    "metrics_execution_aligned.csv", "metrics_sina_qianfuquan.csv", "metrics_sina_houfuquan.csv",
    "metrics_loo.csv", "metrics_mechanism_h5.csv", "metrics_protocol_degradation.csv",
    "paired_block_bootstrap_h5.csv", "paired_block_bootstrap_sensitivity_h5.csv",
    "economic_metrics_h5.csv", "economic_curves_h5.csv", "economic_buy_hold_h5.csv",
]


def check_files():
    missing = [f"data/{x}" for x in REQUIRED if not (DATA / x).is_file()]
    missing += [f"results/{x}" for x in RESULT_REQUIRED if not (OUT / x).is_file()]
    if missing:
        raise FileNotFoundError("Missing required reproduction files: " + ", ".join(missing))
    return len(REQUIRED) + len(RESULT_REQUIRED)


def check_panels(info):
    report = {}
    expected_features = info["features"]
    if len(expected_features) != 25:
        raise AssertionError(f"Expected 25 predictors, found {len(expected_features)}")
    for market in ["CN", "TH"]:
        d = pd.read_csv(DATA / f"{market.lower()}_processed.csv", parse_dates=["date"])
        if d.duplicated(["code", "date"]).any():
            raise AssertionError(f"Duplicate code-date rows in {market}")
        if list(d.split.unique()) and not set(d.split).issubset({"train", "validation", "test"}):
            raise AssertionError(f"Unexpected split labels in {market}")
        summary = info["markets"][market]
        observed = {
            "rows": int(len(d)), "stocks": int(d.code.nunique()),
            "start": str(d.date.min().date()), "end": str(d.date.max().date()),
            "splits": {k: int(v) for k, v in d.split.value_counts().items()},
            "up_rate_h5": float(d.y_5.mean()),
        }
        if observed["stocks"] != summary["stocks"] or observed["start"] != summary["start"] or observed["end"] != summary["end"]:
            raise AssertionError(f"Panel summary mismatch for {market}: {observed}")
        if observed["splits"] != summary["splits"]:
            raise AssertionError(f"Split-count mismatch for {market}: {observed['splits']} vs {summary['splits']}")
        if not np.isclose(observed["up_rate_h5"], summary["up_rate_h5"], atol=1e-12):
            raise AssertionError(f"Up-rate mismatch for {market}")
        report[market] = observed
    return report


def recompute_metrics(pred):
    rows = []
    keys = ["market", "split", "h", "method", "seed"]
    for key, g in pred.groupby(keys):
        y, p = g.y.to_numpy(), g.p.to_numpy()
        yh = (p >= .5).astype(int)
        rows.append(dict(zip(keys, key)) | {
            "n": len(g), "accuracy": accuracy_score(y, yh),
            "macro_f1": f1_score(y, yh, average="macro"),
            "mcc": matthews_corrcoef(y, yh),
            "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
            "brier": brier_score_loss(y, p),
        })
    return pd.DataFrame(rows)


def check_primary_predictions():
    pred = pd.read_csv(OUT / "predictions_h5.csv")
    keys = ["market", "split", "h", "method", "seed", "date", "code"]
    if pred.duplicated(keys).any():
        raise AssertionError("Duplicate primary prediction rows")
    if pred[["y", "p"]].isna().any().any():
        raise AssertionError("Missing labels or probabilities in primary predictions")
    if not pred.p.between(0, 1).all():
        raise AssertionError("Predicted probabilities outside [0,1]")
    if set(pred.seed.unique()) != {11, 22, 33, 44, 55}:
        raise AssertionError("Unexpected primary seed set")
    expected_methods = {"static", "logistic", "periodic", "periodicER", "adwinER", "phER", "adwinDTARC", "phDTARC"}
    if set(pred.method.unique()) != expected_methods:
        raise AssertionError("Unexpected primary method set")
    frozen = pd.read_csv(OUT / "metrics_h5.csv")
    recomputed = recompute_metrics(pred)
    keys = ["market", "split", "h", "method", "seed"]
    merged = frozen.merge(recomputed, on=keys, suffixes=("_frozen", "_recomputed"), validate="one_to_one")
    if len(merged) != len(frozen) or len(merged) != 80:
        raise AssertionError(f"Primary metric row mismatch: {len(merged)}")
    max_error = {}
    for metric in ["n", "accuracy", "macro_f1", "mcc", "auc", "brier"]:
        error = np.abs(merged[f"{metric}_frozen"] - merged[f"{metric}_recomputed"])
        max_error[metric] = float(np.nanmax(error))
        if not np.allclose(merged[f"{metric}_frozen"], merged[f"{metric}_recomputed"], rtol=1e-10, atol=1e-12, equal_nan=True):
            raise AssertionError(f"Frozen metric mismatch: {metric}, max error {max_error[metric]}")
    means = pred.groupby(["market", "method"]).size().to_dict()
    return {"prediction_rows": int(len(pred)), "metric_rows": int(len(frozen)), "max_absolute_metric_error": max_error,
            "rows_per_market": {m: int(pred[pred.market == m].shape[0]) for m in ["CN", "TH"]}}


def main():
    checked_files = check_files()
    info = json.loads((DATA / "dataset_summary.json").read_text())
    panels = check_panels(info)
    primary = check_primary_predictions()
    pdf_figures = sorted((ROOT / "figures").glob("fig*.pdf"))
    png_figures = sorted((ROOT / "figures").glob("fig*.png"))
    if len(pdf_figures) != 16 or len(png_figures) != 16:
        raise AssertionError(f"Expected 16 PDF and 16 PNG figures, found {len(pdf_figures)} and {len(png_figures)}")
    source_files = sorted((ROOT / "src").glob("*.py"))
    for path in source_files:
        compile(path.read_text(), str(path), "exec")
    report = {
        "status": "PASS", "required_core_files_checked": checked_files,
        "python_files_syntax_checked": len(source_files), "panels": panels,
        "primary_predictions": primary, "pdf_figures": len(pdf_figures), "png_figures": len(png_figures),
        "note": "Metrics were independently recomputed from frozen predictions and compared with metrics_h5.csv.",
    }
    (ROOT / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
