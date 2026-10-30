"""Assemble a single-file PLOS ONE manuscript from the verified study sections."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "manuscript"
S = M / "sections"


def read(name):
    return (S / name).read_text()


def convert_common(text):
    text = re.sub(r"\\citep\{([^}]+)\}", r"\\cite{\1}", text)
    text = re.sub(r"\\citet\{([^}]+)\}", r"\\cite{\1}", text)
    text = re.sub(r"\\(sub|subsub)?section\{", lambda m: "\\" + (m.group(1) or "") + "section*{", text)
    text = text.replace("Appendix Table~", "Table~")
    text = text.replace("Appendix Figure~", "Fig~")
    text = text.replace("Figure~\\ref", "Fig~\\ref")
    text = text.replace("Equation~\\eqref", "Eq~\\eqref")
    text = text.replace("Eq.~\\eqref", "Eq~\\eqref")
    return text


def remove_figures_keep_captions(text):
    # PLOS initial-submission PDFs contain captions but not embedded graphics.
    text = re.sub(r"\\begin\{figure\}\[[^]]*\]", r"\\begin{figure}[!h]", text)
    text = re.sub(r"^\s*\\centering\s*$", "", text, flags=re.M)
    text = re.sub(r"^\s*\\includegraphics(?:\[[^]]*\])?\{[^}]*\}\s*$", "", text, flags=re.M)
    return text


def strip_table(text, label):
    pattern = r"\\begin\{table\}.*?\\label\{" + re.escape(label) + r"\}.*?\\end\{table\}"
    return re.sub(pattern, "", text, flags=re.S)


intro = convert_common(read("01_introduction.tex"))
related = convert_common(read("02_related_work.tex"))
related = related.replace("\\section*{Related work}\\label{sec:related}", "\\subsection*{Related work}")
related = related.replace("\\subsection*{", "\\subsubsection*{").replace("\\subsubsection*{Related work}", "\\subsection*{Related work}")

method = convert_common(read("03_problem_method.tex"))
method = method.replace("\\section*{Problem formulation and DTARC}\\label{sec:method}", "\\section*{Materials and methods}\\label{sec:method}\n\\subsection*{Problem formulation and DTARC}")
method = method.replace("\\subsection*{", "\\subsubsection*{").replace("\\subsubsection*{Problem formulation and DTARC}", "\\subsection*{Problem formulation and DTARC}")
design = convert_common(read("04_experimental_design.tex"))
design = design.replace("\\section*{Experimental design}\\label{sec:design}", "\\subsection*{Experimental design}\\label{sec:design}")
design = design.replace("\\subsection*{", "\\subsubsection*{").replace("\\subsubsection*{Experimental design}", "\\subsection*{Experimental design}")

results = convert_common(read("05_results.tex"))
discussion = convert_common(read("06_discussion.tex"))
conclusion = convert_common(read("07_conclusion.tex"))

replacement = {
    "tab01_data": "plos_tab01_design",
    "tab02_protocol": "",
    "tab03_main_h5": "plos_tab04_primary",
    "tab04_h1": "",
    "tab05_ablation": "plos_tab07_mechanisms",
    "tab06_runtime": "",
    "tab07_economic": "plos_tab09_economic",
    "tab08_bootstrap": "plos_tab05_inference",
    "tab09_sensitivity": "",
    "tab10_full_metrics": "",
    "tab11_longonly": "",
    "tab12_matched_replay": "plos_tab06_robustness",
    "tab13_execution_aligned": "",
    "tab14_data_robustness": "",
    "tab15_protocol_degradation": "",
    "tab16_boundary_audit": "plos_tab03_audits",
    "tab17_block_sensitivity": "",
    "tab18_mechanism": "",
}
body = "\n".join([intro, related, method, design, results, discussion, conclusion])
for old, new in replacement.items():
    body = body.replace("\\input{tables/" + old + "}", ("\\input{plos_tables/" + new + "}") if new else "")
body = strip_table(body, "tab:features")
needle = "\\subsection*{Feature map}\n"
feature_table = "\\input{plos_tables/plos_tab02_features}\n"
# Insert the merged feature/dictionary table immediately after the feature-map paragraph.
feature_para_end = "Market return and five-day market momentum are same-date cross-sectional means; relative momentum subtracts market momentum from stock momentum.\n"
body = body.replace(feature_para_end, feature_para_end + "\n" + feature_table, 1)
# The sensitivity table belongs in the Results read order, not in a separate appendix.
body = body.replace("These runs do not select a new configuration; their purpose is to expose whether the frozen point is exceptional. It is not: performance varies modestly, Thai MCC remains negative, and no setting supports a general superiority claim.\n", "These runs do not select a new configuration; their purpose is to expose whether the frozen point is exceptional. It is not: performance varies modestly, Thai MCC remains negative, and no setting supports a general superiority claim.\n\n\\input{plos_tables/plos_tab08_sensitivity}\n", 1)
body = remove_figures_keep_captions(body)

preamble = r"""% PLOS ONE manuscript prepared with the official PLOS LaTeX template, version 3.8 (April 2026).
\documentclass[10pt,letterpaper]{article}
\usepackage[top=0.85in,left=2.75in,footskip=0.75in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{changepage}
\usepackage{textcomp,marvosym}
\usepackage{cite}
\usepackage{nameref,hyperref}
\usepackage[right]{lineno}
\usepackage[nopatch=eqnum]{microtype}
\ifdefined\XeTeXversion\else\DisableLigatures[f]{encoding = *, family = * }\fi
\usepackage[table]{xcolor}
\usepackage{array}
\newcolumntype{+}{!{\vrule width 2pt}}
\newlength\savedwidth
\newcommand\thickcline[1]{%
  \noalign{\global\savedwidth\arrayrulewidth\global\arrayrulewidth 2pt}%
  \cline{#1}%
  \noalign{\vskip\arrayrulewidth}%
  \noalign{\global\arrayrulewidth\savedwidth}%
}
\newcommand\thickhline{\noalign{\global\savedwidth\arrayrulewidth\global\arrayrulewidth 2pt}%
\hline
\noalign{\global\arrayrulewidth\savedwidth}}
\usepackage{booktabs}
\usepackage{float}
\usepackage{longtable}
\usepackage{multirow}
\usepackage{threeparttable}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{enumitem}
\usepackage{setspace}
\doublespacing
\raggedright
\setlength{\parindent}{0.5cm}
\textwidth 5.25in
\textheight 8.75in
\usepackage[aboveskip=1pt,labelfont=bf,labelsep=period,justification=raggedright,singlelinecheck=off]{caption}
\renewcommand{\figurename}{Fig}
\bibliographystyle{plos2025}
\makeatletter
\renewcommand{\@biblabel}[1]{\quad#1.}
\makeatother
\usepackage{lastpage,fancyhdr,graphicx}
\usepackage{epstopdf}
\pagestyle{fancy}
\fancyhf{}
\rfoot{\thepage/\pageref{LastPage}}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrule}{\hrule height 2pt \vspace{2mm}}
\fancyheadoffset[L]{2.25in}
\fancyfootoffset[L]{2.25in}
\lfoot{PLOS ONE submission manuscript}
\hypersetup{pdftitle={When does drift-type-aware replay help? A leakage-safe two-market audit under delayed feedback},pdfsubject={Continual learning and concept drift in non-stationary financial markets},colorlinks=true,linkcolor=black,citecolor=black,urlcolor=black}
\setlist[itemize]{itemsep=.25em,leftmargin=1.5em}
\setlist[enumerate]{itemsep=.25em,leftmargin=1.7em}
\emergencystretch=1.5em

\begin{document}
\begin{flushleft}
{\Large\textbf{When does drift-type-aware replay help? A leakage-safe two-market audit under delayed feedback}}

\smallskip
\textbf{Short title:} Boundary conditions for drift-type-aware replay

\smallskip
Given Name Family Name\textsuperscript{1*}

\smallskip
\textbf{1} Department, Institution, City, Country

\smallskip
* author@example.com

\smallskip
\textbf{Author information must be completed before submission.}
\end{flushleft}

\section*{Abstract}
Experience replay is intended to support adaptation under concept drift, but delayed labels, inconsistent price adjustment, and unmatched baselines can obscure its value. This study asks when drift-type-aware replay improves on the same detector with uniform replay. The mechanism maps error-change severity to recent, diverse, anchor, and uniform replay quotas and a proximal update. A prediction-first protocol enforces label maturation, temporal purging, and frozen configurations. We evaluate 24 Chinese and Thai stocks over 2023--2026, tune only on Chinese 2025 data, and restart 2026 tests from initial models. Five seeds, matched ablations, execution-aligned labels, provenance audits, moving-block inference, and portfolio benchmarks support the boundary analysis. In China, drift-type-aware replay obtains a Matthews correlation coefficient of 0.0440 and an area under the receiver operating characteristic curve of 0.5371, compared with 0.0441 and 0.5378 for a static network and 0.0417 and 0.5373 for uniform detector-triggered replay. Thai discrimination remains below chance. Logistic regression is stronger in China and better calibrated in both markets. Class-stratified uniform replay matches or exceeds the structured method, while execution, adjustment, and market-context audits establish no robust advantage. At 10 basis points, long--short annualized returns are negative in both markets. Drift-type-aware replay is thus a transparent adaptation mechanism, not a universally superior forecaster. Sparse detector coverage, weak signal, calibration trade-offs, and market context bound its usefulness.

\clearpage
\newgeometry{top=0.85in,left=1in,right=1in,footskip=0.75in}
\linenumbers
"""

repro = r"""
\section*{Reproducibility and artifact availability}\label{app:reproducibility}
The complete pipeline uses Python with pandas, NumPy, scikit-learn, PyTorch, and Matplotlib. Table~\ref{tab:artifacts} maps the scientific claims to frozen artifacts and generation scripts. The reproducibility package records package versions, execution order, source URLs, and SHA-256 hashes. Raw data downloads are network-dependent; all reported numerical outputs can be regenerated from the included processed data without contacting the sources. No result table is manually edited.

\input{plos_tables/plos_tab10_artifacts}

\section*{Data and code availability}
The reproducibility package contains the Sina and SiamChart download scripts, processed panels, frozen predictions, experiment outputs, and figure/table code. The primary Chinese panel is single-source and forward adjusted; backward-adjusted and rejected stitched panels are included solely for audits. Before publication, the authors should deposit the package in a public repository and replace this statement with the repository name and persistent identifier.

\section*{Ethics statement}
This study uses only publicly accessible aggregate market data and involves neither human participants nor animals. Ethics approval, consent to participate, and consent to publish are not applicable.

\section*{Competing interests}
The authors have no relevant financial or non-financial interests to disclose.

\section*{Author contributions}
The author(s) designed the study, verified the data and experiments, interpreted the results, and approved the manuscript. This statement must be completed with the final author list and the CRediT taxonomy before submission.

\section*{Use of generative AI}
Kimi (Moonshot AI) assisted with literature-metadata checks, code debugging, figure generation, and language editing. It was not treated as experimental evidence. The author(s) checked the saved data, executed outputs, citations, and final text and take full responsibility for them.

\section*{Acknowledgments}
The authors have no additional acknowledgments.

\nolinenumbers
\bibliography{references}

\section*{Supporting information}
\paragraph*{S1 File.}\label{S1_File}
\textbf{Reproducibility package.} Contains processed data, frozen predictions, experiment outputs, source code, and file hashes.

\paragraph*{S2 File.}\label{S2_Figures}
\textbf{Separate figure files.} Contains Figs 1--16 in publication-suitable formats; figure captions remain in the main manuscript.

\end{document}
"""

# Expand all table inputs so the deliverable is one cohesive .tex file.
def expand_tables(text):
    def repl(match):
        name = match.group(1)
        return (M / "plos_tables" / (name + ".tex")).read_text()
    return re.sub(r"\\input\{plos_tables/([^}]+)\}", repl, text)

out = preamble + body + repro
out = expand_tables(out)
(M / "main_plos.tex").write_text(out)
print("Wrote", M / "main_plos.tex", "characters", len(out), "remaining inputs", out.count("\\input{"))
