#!/usr/bin/env python3
"""
Compute confusion matrix and core metrics from prediction CSVs.

Supports:
- Deezer eval_manifest output: target,pred_label,pred_prob
- SONICS benchmark output: target,y_pred (probability)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--model_name", type=str, default="")
    p.add_argument("--dataset_name", type=str, default="")
    return p.parse_args()


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else np.nan


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    if "target" in df.columns:
        y_true = df["target"].values.astype(int)
    elif "y_true" in df.columns:
        y_true = df["y_true"].values.astype(int)
    else:
        raise ValueError("Could not find target labels (target or y_true)")

    if "pred_label" in df.columns:
        y_pred = df["pred_label"].values.astype(int)
    elif "y_pred" in df.columns:
        y_pred = (df["y_pred"].values.astype(float) >= args.threshold).astype(int)
    else:
        raise ValueError("Could not find predictions (pred_label or y_pred)")

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = safe_div(tp + tn, tp + tn + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    f1 = safe_div(2 * prec * rec, prec + rec) if np.isfinite(prec) and np.isfinite(rec) else np.nan
    bal_acc = safe_div(rec + spec, 2) if np.isfinite(rec) and np.isfinite(spec) else np.nan

    out = pd.DataFrame(
        [
            {
                "model_name": args.model_name,
                "dataset_name": args.dataset_name,
                "n_samples": int(len(y_true)),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "accuracy": acc,
                "precision_fake": prec,
                "recall_fake_sensitivity": rec,
                "specificity_real": spec,
                "f1_fake": f1,
                "balanced_accuracy": bal_acc,
            }
        ]
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.to_markdown(index=False))
    print(f"[Saved] {args.output_csv}")


if __name__ == "__main__":
    main()

