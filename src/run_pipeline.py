"""Reproduce DTARC artifacts from the bundled data and frozen configuration.

Run from any working directory. Paths are resolved relative to the project root.
The default stage is non-destructive validation. Other stages overwrite their
corresponding generated artifacts, so preserve the frozen package if needed.
"""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command):
    printable = " ".join(map(str, command))
    print(f"\n>>> {printable}", flush=True)
    subprocess.run([str(x) for x in command], cwd=ROOT, check=True)


def py(script, *args):
    run([sys.executable, ROOT / "src" / script, *args])


def data_stage():
    py("preprocess.py")
    py("prepare_robustness_data.py")
    py("data_boundary_audit.py", "--input", "cn_raw_baostock.csv", "--suffix", "stitched")
    py("data_boundary_audit.py", "--input", "cn_raw_sina_qianfuquan.csv", "--suffix", "single_source")


def experiment_stage():
    py("experiment.py", "--mode", "tune", "--horizon", "5")
    py("experiment.py", "--mode", "final", "--horizon", "5", "--seeds", "11,22,33,44,55")
    for mode in ["ablation", "h1", "sensitivity"]:
        py("extended_experiments.py", "--mode", mode)
    for mode in ["adjustment", "loo", "exec", "matched", "mechanism", "protocol"]:
        py("robustness_experiments.py", "--mode", mode)
    py("statistical_analysis.py", "--mode", "bootstrap", "--replicates", "500", "--blocks", "5,10,20")
    py("statistical_analysis.py", "--mode", "ablation")
    py("economic_evaluation.py", "--horizon", "5")
    py("economic_evaluation.py", "--horizon", "5", "--input", "results/predictions_execution_aligned.csv", "--suffix", "execution_aligned")


def artifact_stage():
    py("make_figures.py")
    py("make_plos_tables.py")
    py("build_plos_manuscript.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["validate", "data", "experiments", "artifacts", "all"], default="validate")
    args = parser.parse_args()
    if args.stage == "validate":
        py("validate_reproduction.py")
    elif args.stage == "data":
        data_stage()
    elif args.stage == "experiments":
        experiment_stage()
    elif args.stage == "artifacts":
        artifact_stage()
    else:
        data_stage()
        experiment_stage()
        artifact_stage()
        py("validate_reproduction.py")


if __name__ == "__main__":
    main()
