"""Build consolidated PLOS ONE tables directly from frozen experiment outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
T = ROOT / "manuscript" / "plos_tables"
T.mkdir(parents=True, exist_ok=True)
ORDER = ["static", "logistic", "periodic", "periodicER", "adwinER", "phER", "adwinDTARC", "phDTARC"]


def esc(x):
    return str(x).replace("_", r"\_")


def num(x, n=4):
    return f"{x:.{n}f}"


def pm(q, metric, n=4):
    return f"{q[metric].mean():.{n}f} $\\pm$ {q[metric].std():.{n}f}"


def make_table(filename, title, labels, panels, note, width="-0.45in", right_width="-0.45in", align_default=None, size=r"\footnotesize", stretch="1.12"):
    parts = [r"\begin{table}[H]", r"\centering", f"\\caption{{\\textbf{{{title}}}}}"]
    parts += [f"\\label{{{x}}}" for x in labels]
    parts += [r"\begin{singlespace}", f"\\begin{{adjustwidth}}{{{width}}}{{{right_width}}}", size, r"\setlength{\tabcolsep}{4pt}", f"\\renewcommand{{\\arraystretch}}{{{stretch}}}"]
    for i, (heading, headers, rows, align) in enumerate(panels):
        if heading:
            parts += [f"\\textbf{{Panel {chr(65+i)}. {heading}}}" + r"\\[2pt]"]
        parts += [f"\\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
        parts += [" & ".join(map(str, row)) + r" \\" for row in rows]
        parts += [r"\bottomrule", r"\end{tabular}"]
        if i != len(panels) - 1:
            parts += [r"\par\vspace{7pt}\par"]
    parts += [r"\end{adjustwidth}", r"\end{singlespace}", "\\begin{flushleft}\\footnotesize " + note + r"\end{flushleft}", r"\end{table}", ""]
    (T / filename).write_text("\n".join(parts))
    print("PLOS table", filename)


def metric_row(market, method, q, runtime=None):
    row = [esc(method)] + [pm(q, m) for m in ["accuracy", "macro_f1", "mcc", "auc", "brier"]]
    if runtime is not None:
        row += ["--" if len(runtime) == 0 else f"{runtime.updates.mean():.2f}", "--" if len(runtime) == 0 else f"{runtime.seconds.mean():.3f}"]
    return row


def main():
    # Table 1: sample and frozen protocol.
    info = json.loads((ROOT / "data" / "dataset_summary.json").read_text())
    pred = pd.read_csv(R / "predictions_h5.csv")
    vals = {}
    for m in ["CN", "TH"]:
        s = info["markets"][m]
        n = len(pred[(pred.market == m) & (pred.method == "static") & (pred.seed == 11)])
        vals[m] = {
            "Price source": "Sina Finance, single-source forward adjustment" if m == "CN" else "SiamChart, as published",
            "Stocks": s["stocks"],
            "Common modeling window": f"{s['start']}--{s['end']}",
            "Initial training rows": s["splits"]["train"],
            "Validation rows": s["splits"]["validation"],
            "Test rows": s["splits"]["test"],
            "Scored five-day rows per seed": n,
            "Observed five-day up rate": f"{s['up_rate_h5']:.3f}",
        }
    rows_a = [[k, vals["CN"][k], vals["TH"][k]] for k in vals["CN"]]
    cfg = json.loads((R / "best_config.json").read_text())["best"]
    rows_b = [
        ["Initial MLP", "25--32--16--1; ReLU; dropout 0.10"],
        ["Initial optimization", "120 epochs; Adam $10^{-3}$; weight decay $10^{-4}$"],
        ["Online optimization", "2 epochs; Adam $3\\times10^{-4}$; weight decay $10^{-5}$"],
        ["ADWIN $\\delta$; Page--Hinkley threshold", f"{cfg['adwin_delta']}; {cfg['ph_threshold']}"],
        ["Memory capacity; replay batch", f"{cfg['buffer']}; {cfg['replay']}"],
        ["Proximal coefficient; abrupt threshold", f"{cfg['prox']}; {cfg['abrupt_threshold']}"],
        ["Optimization seeds", "11, 22, 33, 44, 55; tuning seed 42"],
        ["Selection and restart", "Chinese 2025 validation only; 2026 restart from 2023--2024 fit"],
    ]
    make_table("plos_tab01_design.tex", "Study sample, temporal partitions, and frozen experimental protocol.", ["tab:design", "tab:data", "tab:protocol"], [
        ("Observed sample by market.", ["Characteristic", "China", "Thailand"], rows_a, "lll"),
        ("Model selection and optimization settings.", ["Component", "Frozen value"], rows_b, "ll"),
    ], "The common modeling window follows feature warm-up. Initial training purges the final $h$ rows per stock. Validation-period model states are not carried into the 2026 test.", width="-0.45in")

    # Table 2: feature groups and core data dictionary.
    rows_a = [
        ["Returns and momentum", "6", "$r_1$ and momentum over 2, 3, 5, 10, and 20 days"],
        ["Rolling volatility", "3", "Standard deviations over 5, 10, and 20 days"],
        ["Moving-average ratios", "3", "Price/SMA ratios over 5, 10, and 20 days"],
        ["Volume state", "4", "Log-volume $z$-scores over 5, 10, and 20 days; log-volume change"],
        ["Oscillators", "3", "Price-normalized MACD and signal; scaled RSI-14"],
        ["Price position", "2", "High--low range and close position within daily range"],
        ["Liquidity", "1", "Twenty-day Amihud-style absolute return per amount proxy"],
        ["Market context", "3", "Market return, five-day momentum, and relative momentum"],
    ]
    dictionary = [
        ("date", "date", "Trading date used for ordering and splitting"),
        ("code", "string", "Market-specific security identifier"),
        ("open/high/low/close", "numeric", "Daily OHLC prices from the named source"),
        ("volume", "numeric", "Reported daily trading volume"),
        ("amount", "numeric", "Chinese traded amount; unavailable for Thailand"),
        ("ret_1, mom_k", "numeric", "One-day return and $k$-day price momentum"),
        ("vol_k", "numeric", "Rolling standard deviation of daily returns"),
        ("sma_ratio_k", "numeric", "Close/SMA minus one over $k$ days"),
        ("volume_z_k", "numeric", "Rolling $z$-score of log volume"),
        ("macd_norm, macd_signal_norm", "numeric", "Price-normalized MACD and signal"),
        ("rsi_14", "numeric", "Fourteen-period relative strength index divided by 100"),
        ("range_hl, close_position", "numeric", "Daily price range and close location"),
        ("amihud_20", "numeric", "Twenty-day absolute-return/amount proxy"),
        ("market_*", "numeric", "Same-date cross-sectional market context"),
        ("fwd_ret_h, y_h", "numeric/binary", "Forward return and positive-return indicator"),
        ("split", "category", "Train, validation, or test partition"),
        ("p", "probability", "Pre-update predicted probability"),
        ("method, seed", "category", "Method and optimization-seed identifiers"),
    ]
    rows_b = [[f"\\texttt{{{esc(a)}}}", b, c] for a, b, c in dictionary]
    make_table("plos_tab02_features.tex", "Predictor feature map and core processed-data dictionary.", ["tab:features", "tab:dictionary"], [
        ("Feature groups used by every neural and linear predictor.", ["Group", "Count", "Variables"], rows_a, "llp{4.7in}"),
        ("Core fields in processed panels and frozen prediction files.", ["Field", "Type", "Meaning"], rows_b, "llp{4.2in}"),
    ], "The predictor contains $6+3+3+4+3+2+1+3=25$ features. Rolling features use training medians for imputation; standardization is fitted only on purged initial training data.", width="-0.45in")

    # Table 3: integrity and deliberate protocol failures.
    rows_a = []
    for panel in ["stitched", "single_source"]:
        s = pd.read_csv(R / f"boundary_summary_{panel}.csv").iloc[0]
        rows_a.append([esc(panel), f"{100*s.median_abs_boundary_return:.2f}\\%", f"{100*s.max_abs_boundary_return:.2f}\\%", f"{s.median_return_psi:.3f}", f"{s.max_return_psi:.3f}", f"{s.median_volume_log_ratio:.3f}"])
    pr = pd.read_csv(R / "metrics_protocol_degradation.csv")
    rows_b = [[esc(q["method"]), num(q.accuracy), num(q.macro_f1), num(q.mcc), num(q.auc), num(q.brier)] for _, q in pr.iterrows()]
    make_table("plos_tab03_audits.tex", "Data-provenance audit and deliberate evaluation-protocol degradation.", ["tab:audits", "tab:boundary", "tab:degradation"], [
        ("Twenty-observation pre/post-boundary diagnostics for the Chinese panels.", ["Panel", "Median $|$boundary $r|$", "Max $|$boundary $r|$", "Median PSI", "Max PSI", "Volume log ratio"], rows_a, "lrrrrr"),
        ("Chinese validation outcomes under correct and deliberately invalid protocols (seed 42).", ["Protocol", "Accuracy", "Macro-F1", "MCC", "AUC", "Brier"], rows_b, "lrrrrr"),
    ], "The stitched panel is rejected because cross-source adjustment scales differ. Random row splitting, premature labels, and removal of the purge are diagnostics only and are invalid for deployment.", width="-0.45in")

    # Table 4: complete primary metrics plus computational cost.
    d = pd.read_csv(R / "metrics_h5.csv")
    rt = pd.read_csv(R / "runtime_h5.csv")
    panels = []
    for market, letter in [("CN", "China"), ("TH", "Thailand")]:
        rows = []
        for method in ORDER:
            q = d[(d.market == market) & (d.method == method)]
            r = rt[(rt.market == market) & (rt.method == method)]
            rows.append(metric_row(market, method, q, r))
        panels.append((f"{letter}: five-day prediction and online adaptation.", ["Method", "Accuracy", "Macro-F1", "MCC", "AUC", "Brier", "Updates", "Seconds"], rows, "lrrrrrrr"))
    make_table("plos_tab04_primary.tex", "Primary five-day predictive performance and observed online computational cost.", ["tab:primary", "tab:main", "tab:fullmetrics", "tab:runtime"], panels, "Entries are mean $\\pm$ standard deviation over five seeds. Updates and seconds are means over seeds; initial fitting is excluded. Logistic regression has no comparable online phase. Lower Brier is better; higher values are better for the other predictive metrics.", width="-0.45in")

    # Table 5: 24 primary tests with block-length sensitivity.
    boot = pd.read_csv(R / "paired_block_bootstrap_h5.csv")
    sens = pd.read_csv(R / "paired_block_bootstrap_sensitivity_h5.csv")
    boot = boot[boot.family == "primary"]
    rows = []
    metric_order = ["accuracy", "mcc", "auc", "brier"]
    for market in ["CN", "TH"]:
        for comp in ["adwinER", "static", "periodicER"]:
            for metric in metric_order:
                q = boot[(boot.market == market) & (boot.comparator == comp) & (boot.metric == metric)].iloc[0]
                intervals = []
                for block in [5, 10, 20]:
                    z = sens[(sens.market == market) & (sens.comparator == comp) & (sens.metric == metric) & (sens.block_days == block)].iloc[0]
                    intervals.append(f"[{z['ci_2.5']:.5f}, {z['ci_97.5']:.5f}]")
                rows.append([market, esc(comp), metric.upper() if metric != "brier" else "Brier", f"{q.mean_difference:.5f}", intervals[0], f"{q.holm_p:.4f}", intervals[1], intervals[2]])
    make_table("plos_tab05_inference.tex", "Paired moving-block inference for DTARC and sensitivity to block length.", ["tab:inference", "tab:bootstrap", "tab:block"], [
        ("Prespecified primary comparisons; effects are DTARC minus the named comparator.", ["Market", "Comparator", "Metric", "Difference", "95\\% CI (5 d)", "Holm $p$", "95\\% CI (10 d)", "95\\% CI (20 d)"], rows, "lllrrrrr"),
    ], "Each interval uses 500 date-level paired moving-block bootstrap replicates, preserving same-date stock cross-sections and five-seed pairing. Holm correction covers 24 primary tests. For Brier, negative differences favor DTARC; for other metrics, positive differences favor DTARC.", width="-0.45in")

    # Table 6: matched replay, execution, adjustment, context, and horizon.
    panels = []
    matched = pd.read_csv(R / "metrics_matched_replay.csv")
    primary = pd.read_csv(R / "metrics_h5.csv")
    rows = []
    for market in ["CN", "TH"]:
        for method in ["adwinDTARC", "adwinER", "adwinERstrat", "DTARC_no_provenance"]:
            src = primary if method in ["adwinDTARC", "adwinER"] else matched
            q = src[(src.market == market) & (src.method == method)]
            rows.append([market, esc(method), num(q.mcc.mean()), num(q.auc.mean()), num(q.brier.mean())])
    panels.append(("Matched replay provenance and class-stratification controls.", ["Market", "Method", "MCC", "AUC", "Brier"], rows, "llrrr"))
    ex = pd.read_csv(R / "metrics_execution_aligned.csv")
    rows = []
    for market in ["CN", "TH"]:
        for method in ["static", "periodicER", "adwinER", "adwinDTARC"]:
            q = ex[(ex.market == market) & (ex.method == method)]
            rows.append([market, esc(method), num(q.mcc.mean()), num(q.auc.mean()), num(q.brier.mean())])
    panels.append(("Execution-aligned $t+1\\rightarrow t+6$ targets and two-batch delay.", ["Market", "Method", "MCC", "AUC", "Brier"], rows, "llrrr"))
    rows = []
    for file, panel in [("metrics_sina_qianfuquan.csv", "Forward"), ("metrics_sina_houfuquan.csv", "Backward")]:
        z = pd.read_csv(R / file)
        for method in ["static", "periodicER", "adwinER", "adwinDTARC"]:
            q = z[(z.market == "CN") & (z.method == method)]
            rows.append([panel, esc(method), num(q.mcc.mean()), num(q.auc.mean()), num(q.brier.mean())])
    panels.append(("Chinese single-source adjustment-convention audit.", ["Adjustment", "Method", "MCC", "AUC", "Brier"], rows, "llrrr"))
    loo = pd.read_csv(R / "metrics_loo.csv")
    rows = []
    for market in ["CN", "TH"]:
        for method in ["static", "adwinER", "adwinDTARC"]:
            q = loo[(loo.market == market) & (loo.method == method)]
            rows.append([market, esc(method), num(q.mcc.mean()), num(q.auc.mean()), num(q.brier.mean())])
    panels.append(("Leave-one-out market-context audit.", ["Market", "Method", "MCC", "AUC", "Brier"], rows, "llrrr"))
    h1 = pd.read_csv(R / "metrics_h1.csv")
    rows = []
    for market in ["CN", "TH"]:
        for method in ORDER:
            q = h1[(h1.market == market) & (h1.method == method)]
            rows.append([market, esc(method), num(q.mcc.mean()), num(q.auc.mean()), num(q.brier.mean())])
    panels.append(("One-day horizon results.", ["Market", "Method", "MCC", "AUC", "Brier"], rows, "llrrr"))
    make_table("plos_tab06_robustness.tex", "Robustness across replay controls, execution alignment, price adjustment, market context, and prediction horizon.", ["tab:robustness", "tab:matched", "tab:exec", "tab:datarob", "tab:h1"], panels, "All entries are five-seed means. No test-set result in this table was used to revise the frozen configuration. The one-day panel includes all principal baselines; other panels emphasize the controls needed to isolate the relevant protocol factor.", width="-0.45in", size=r"\scriptsize", stretch="1.03")

    # Table 7: component ablation and mechanism diagnostics.
    ab = pd.read_csv(R / "metrics_ablation_h5.csv")
    rows = []
    for market in ["CN", "TH"]:
        for method in ["adwinDTARC", "no_typing", "no_diversity", "no_anchor", "no_proximal"]:
            source_method = method if method == "adwinDTARC" else "DTARC_" + method
            q = ab[(ab.market == market) & (ab.method == source_method)]
            if q.empty:
                raise ValueError(f"Missing frozen ablation results: {market} {source_method}")
            rows.append([market, esc(method), num(q.mcc.mean()), num(q.auc.mean()), num(q.brier.mean())])
    mech = pd.read_csv(R / "metrics_mechanism_h5.csv")
    rtm = pd.read_csv(R / "runtime_mechanism_h5.csv")
    rows_b = []
    for factor in ["adwin_delta", "abrupt_threshold", "prox"]:
        for value in sorted(mech[mech.factor == factor].value.unique()):
            row = [esc(factor), f"{value:g}"]
            for market in ["CN", "TH"]:
                q = mech[(mech.market == market) & (mech.factor == factor) & (mech.value == value)]
                u = rtm[(rtm.market == market) & (rtm.factor == factor) & (rtm.value == value)]
                row += [num(q.mcc.mean()), num(q.auc.mean()), f"{u.updates.mean():.1f}"]
            rows_b.append(row)
    make_table("plos_tab07_mechanisms.tex", "Component ablation and detector, typing-threshold, and proximal-scale diagnostics.", ["tab:mechanisms", "tab:ablation", "tab:mechanism"], [
        ("Five-day component ablation.", ["Market", "Variant", "MCC", "AUC", "Brier"], rows, "llrrr"),
        ("Mechanism coverage and discrimination under the frozen protocol.", ["Factor", "Value", "CN MCC", "CN AUC", "CN updates", "TH MCC", "TH AUC", "TH updates"], rows_b, "llrrrrrr"),
    ], "Entries are five-seed means. The full method in Panel A is adwinDTARC. Update counts are means over seeds. These analyses explain mechanism coverage and are not used to select a favorable test configuration.", width="-0.45in")

    # Table 8: parameter sensitivity.
    se = pd.read_csv(R / "metrics_sensitivity_h5.csv")
    rows = []
    for factor in ["buffer", "replay", "online_lr", "ph_threshold"]:
        for value in sorted(se[se.factor == factor].value.unique()):
            row = [esc(factor), f"{value:g}"]
            for market in ["CN", "TH"]:
                q = se[(se.market == market) & (se.factor == factor) & (se.value == value)]
                row += [num(q.mcc.mean()), num(q.auc.mean())]
            rows.append(row)
    make_table("plos_tab08_sensitivity.tex", "Frozen-protocol sensitivity to memory, replay, online learning rate, and Page--Hinkley threshold.", ["tab:sensitivity"], [
        ("Parameter diagnostics; values are five-seed means.", ["Factor", "Value", "CN MCC", "CN AUC", "TH MCC", "TH AUC"], rows, "llrrrr"),
    ], "The selected point is not exceptional across these diagnostics. Sensitivity runs were not used to retune the frozen test configuration.", width="-0.45in")

    # Table 9: economic outcomes, both styles and benchmarks.
    em = pd.read_csv(R / "economic_metrics_h5.csv")
    bh = pd.read_csv(R / "economic_buy_hold_h5.csv")
    panels = []
    for style, heading in [("long_short", "Next-close long--short portfolios."), ("long_only", "Top-three long-only portfolios.")]:
        rows = []
        for market in ["CN", "TH"]:
            for method in ORDER + ["equal_weight"]:
                actual_style = "benchmark" if method == "equal_weight" else style
                q = em[(em.market == market) & (em.method == method) & (em.cost_bps == 10) & (em["style"] == actual_style)]
                rows.append([market, esc(method), f"{100*q.annual_return.mean():.2f}\\%", f"{100*q.annual_excess_return.mean():.2f}\\%", f"{q.sharpe.mean():.3f}", f"{q.information_ratio.mean():.3f}" if method != "equal_weight" else "--", f"{q.market_beta.mean():.3f}", f"{100*q.max_drawdown.mean():.2f}\\%"])
            if style == "long_short":
                q = bh[(bh.market == market) & (bh.cost_bps == 10)]
                rows.append([market, r"buy\_hold", f"{100*q.annual_return.mean():.2f}\\%", "--", f"{q.sharpe.mean():.3f}", "--", "1.000", f"{100*q.max_drawdown.mean():.2f}\\%"])
        panels.append((heading, ["Market", "Method", "Return", "Excess", "Sharpe", "IR", "Beta", "Max DD"], rows, "llrrrrrr"))
    make_table("plos_tab09_economic.tex", "Annualized economic outcomes after 10 bps, including market benchmarks and both portfolio styles.", ["tab:economic", "tab:longonly"], panels, "Returns are annualized from non-overlapping five-day periods with next-close execution. Excess return and information ratio (IR) use the same-cost equal-weight universe benchmark. Buy-and-hold uses fixed initial equal shares and therefore has no same-cost excess-return row. Maximum drawdown (Max DD) is reported as a positive magnitude.", width="-0.45in")

    # Table 10: reproducibility mapping.
    artifacts = [
        ("Single-source processed panels", r"data/*\_processed.csv", r"src/prepare\_robustness\_data.py"),
        ("Validation configuration", r"results/best\_config.json", r"src/experiment.py --mode tune"),
        ("Primary predictions", r"results/predictions\_h5.csv", r"src/experiment.py --mode final"),
        ("Horizon and component ablations", r"results/metrics\_h1.csv; metrics\_ablation\_h5.csv", r"src/extended\_experiments.py"),
        ("Matched and execution audits", r"results/metrics\_*aligned*.csv; metrics\_matched\_replay.csv", r"src/robustness\_experiments.py"),
        ("Boundary and protocol audits", r"results/boundary\_*.csv; metrics\_protocol\_degradation.csv", r"src/data\_boundary\_audit.py; src/robustness\_experiments.py"),
        ("Paired inference and block sensitivity", r"results/paired\_block\_bootstrap\_*.csv", r"src/statistical\_analysis.py"),
        ("Portfolio outcomes and benchmarks", r"results/economic\_*.csv", r"src/economic\_evaluation.py"),
        ("Figures and source tables", r"figures/; manuscript/plos\_tables/", r"src/make\_figures.py; src/make\_plos\_tables.py"),
        ("Bibliographic metadata", r"literature/crossref\_verified.json", r"src/make\_bibliography.py"),
    ]
    rows = [[a, f"\\texttt{{{b}}}", f"\\texttt{{{c}}}"] for a, b, c in artifacts]
    make_table("plos_tab10_artifacts.tex", "Mapping from scientific claims to executable artifacts.", ["tab:artifacts"], [
        ("Claim-to-artifact reproducibility map.", ["Claim or output", "Source artifact", "Generation script"], rows, "p{1.75in}p{2.65in}p{2.45in}"),
    ], "The package records package versions, execution order, source URLs, and SHA-256 hashes. Raw downloads are network-dependent; all reported outputs can be regenerated from the included processed panels. No consolidated table is manually edited.", width="-0.45in")


if __name__ == "__main__":
    main()
