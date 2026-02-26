#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")


def esc(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--confusion_dir", type=Path, required=True)
    ap.add_argument("--out_tex", type=Path, required=True)
    args = ap.parse_args()
    out_dir = args.out_tex.parent.resolve()

    df = pd.read_csv(args.csv).copy()
    df = df.sort_values(["dataset", "model"]).reset_index(drop=True)

    # numeric formatting
    for c in [
        "accuracy",
        "precision_fake",
        "recall_fake_sensitivity",
        "specificity_real",
        "f1_fake",
        "balanced_accuracy",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    datasets = list(df["dataset"].dropna().unique())
    models = list(df["model"].dropna().unique())

    lines = []
    add = lines.append

    add(r"\documentclass[11pt]{article}")
    add(r"\usepackage[margin=1in]{geometry}")
    add(r"\usepackage{booktabs}")
    add(r"\usepackage{longtable}")
    add(r"\usepackage{array}")
    add(r"\usepackage{graphicx}")
    add(r"\usepackage{float}")
    add(r"\usepackage{hyperref}")
    add(r"\usepackage{caption}")
    add(r"\captionsetup{font=small}")
    add(r"\title{Out-of-Domain Deepfake Audio Evaluation Report}")
    add(r"\author{Generated from experiment artifacts}")
    add(r"\date{\today}")
    add(r"\begin{document}")
    add(r"\maketitle")

    add(r"\section{Scope}")
    add(
        "This report summarizes out-of-domain (OOD) evaluation results from "
        r"\texttt{final\_ood\_comparison.csv} and the generated confusion matrices."
    )
    add(
        "All numeric values in this report are taken directly from the experiment output CSV; "
        "no values are manually edited."
    )

    add(r"\section{Split Construction and Leakage Control}")
    add(
        r"Splits were generated with \texttt{ood\_eval/build\_balanced\_ood\_splits.py}. "
        "The split policy used validation data for model-specific sources to avoid train-test leakage:"
    )
    add(r"\begin{itemize}")
    add(
        r"\item \textbf{FMA real/fake}: sampled from Deezer validation split "
        r"(\texttt{dataset\_medium\_split\_filtered.npy}, key \texttt{validation})."
    )
    add(
        r"\item \textbf{FMA fake encoder fairness}: fake samples were stratified across "
        r"\texttt{encodec3, encodec6, encodec24, griffin256, griffin512, lac14, lac2, lac7, musika}."
    )
    add(
        r"\item \textbf{SONICS real/fake}: sampled from \texttt{SONICS/valid.csv}. "
        r"Entries overlapping \texttt{SONICS/train.csv} are excluded."
    )
    add(r"\item \textbf{FakeMusicCaps fake}: sampled from the FakeMusicCaps pool.")
    add(r"\end{itemize}")

    add(r"\section{Evaluation Orchestration}")
    add(r"\begin{itemize}")
    add(
        r"\item \textbf{Deezer model}: evaluated with "
        r"\texttt{deepfake-detector/scripts/eval\_manifest.py}, aligned to the original "
        r"\texttt{eval.py} pipeline (configuration loading, preprocessing, architecture, weights)."
    )
    add(
        r"\item \textbf{SONICS models}: evaluated with "
        r"\texttt{sonics/benchmark\_pretrained.py} using the same CSV manifests."
    )
    add(
        r"\item \textbf{Metric consolidation}: confusion terms (TP, TN, FP, FN) and derived metrics "
        r"were consolidated in \texttt{ood\_eval/final\_ood\_comparison.csv}."
    )
    add(r"\end{itemize}")

    add(r"\section{Evaluated Datasets and Models}")
    add("Datasets used in this report:")
    add(r"\begin{itemize}")
    for d in datasets:
        n = int(df[df["dataset"] == d]["n"].max())
        add(rf"\item \texttt{{{esc(d)}}}: {n} samples")
    add(r"\end{itemize}")
    add("Models used in this report:")
    add(r"\begin{itemize}")
    for m in models:
        add(rf"\item \texttt{{{esc(m)}}}")
    add(r"\end{itemize}")

    add(r"\section{Results Table}")
    add(r"\small")
    add(r"\begin{longtable}{p{3.5cm}p{3.0cm}rrrrrrrrrr}")
    add(r"\toprule")
    add(
        r"Model & Dataset & N & TP & TN & FP & FN & Acc & Prec\_fake & Rec\_fake & Spec\_real & BalAcc \\"
    )
    add(r"\midrule")
    add(r"\endfirsthead")
    add(r"\toprule")
    add(
        r"Model & Dataset & N & TP & TN & FP & FN & Acc & Prec\_fake & Rec\_fake & Spec\_real & BalAcc \\"
    )
    add(r"\midrule")
    add(r"\endhead")
    for _, r in df.iterrows():
        acc = "NaN" if pd.isna(r["accuracy"]) else f'{r["accuracy"]:.4f}'
        p = "NaN" if pd.isna(r["precision_fake"]) else f'{r["precision_fake"]:.4f}'
        rec = "NaN" if pd.isna(r["recall_fake_sensitivity"]) else f'{r["recall_fake_sensitivity"]:.4f}'
        spec = "NaN" if pd.isna(r["specificity_real"]) else f'{r["specificity_real"]:.4f}'
        bal = "NaN" if pd.isna(r["balanced_accuracy"]) else f'{r["balanced_accuracy"]:.4f}'
        add(
            rf"\texttt{{{esc(r['model'])}}} & "
            rf"\texttt{{{esc(r['dataset'])}}} & "
            rf"{int(r['n'])} & {int(r['TP'])} & {int(r['TN'])} & {int(r['FP'])} & {int(r['FN'])} & "
            rf"{acc} & {p} & {rec} & {spec} & {bal} \\"
        )
    add(r"\bottomrule")
    add(r"\end{longtable}")
    add(r"\normalsize")

    add(r"\section{Confusion Matrices}")
    add(
        r"Each subsection below contains the confusion matrix image for every model-dataset combination. "
        r"Image files are loaded from \texttt{ood\_eval/confusion/}."
    )

    for d in datasets:
        add(rf"\subsection{{Dataset: \texttt{{{esc(d)}}}}}")
        ddf = df[df["dataset"] == d]
        for _, r in ddf.iterrows():
            m = str(r["model"])
            fig_name = f"{slug(d)}__{slug(m)}.png"
            fig_path = args.confusion_dir / fig_name
            acc_txt = "NaN" if pd.isna(r["accuracy"]) else f"{r['accuracy']:.4f}"
            add(r"\begin{figure}[H]")
            add(r"\centering")
            if fig_path.exists():
                try:
                    rel = fig_path.resolve().relative_to(out_dir).as_posix()
                except Exception:
                    rel = fig_path.resolve().as_posix()
                add(rf"\includegraphics[width=0.58\textwidth]{{{rel}}}")
            else:
                add(r"\fbox{\parbox{0.9\textwidth}{Confusion matrix image not found.}}")
            add(
                rf"\caption{{Dataset=\texttt{{{esc(d)}}}, Model=\texttt{{{esc(m)}}}. "
                rf"TP={int(r['TP'])}, TN={int(r['TN'])}, FP={int(r['FP'])}, FN={int(r['FN'])}, "
                rf"Accuracy={acc_txt}.}}"
            )
            add(r"\end{figure}")

    add(r"\section{Interpretation Notes}")
    add(r"\begin{itemize}")
    add(
        r"\item On fake-only datasets, specificity (\texttt{spec\_real}) is undefined because there are no real samples."
    )
    add(
        r"\item On real-only datasets, fake recall is undefined because there are no fake samples."
    )
    add(
        r"\item Balanced accuracy is reported only when both class recalls are defined."
    )
    add(r"\end{itemize}")

    add(r"\end{document}")

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {args.out_tex}")


if __name__ == "__main__":
    main()
