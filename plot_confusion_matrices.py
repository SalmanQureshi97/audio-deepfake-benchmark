#!/usr/bin/env python3
"""
Plot confusion matrices for each (model, dataset) row in final_ood_comparison.csv.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("ood_eval/confusion_matrices"))
    p.add_argument("--normalized", action="store_true", help="Also save row-normalized matrices")
    return p.parse_args()


def plot_matrix(cm: np.ndarray, title: str, out_path: Path, fmt: str = "d") -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred Real (0)", "Pred Fake (1)"])
    ax.set_yticks([0, 1], labels=["True Real (0)", "True Fake (1)"])
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            txt = f"{int(val)}" if fmt == "d" else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for _, r in df.iterrows():
        model = str(r["model"])
        dataset = str(r["dataset"])
        tn = int(r["TN"])
        fp = int(r["FP"])
        fn = int(r["FN"])
        tp = int(r["TP"])

        cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
        name = f"{slug(dataset)}__{slug(model)}"

        plot_matrix(
            cm,
            title=f"{dataset}\n{model}",
            out_path=args.out_dir / f"{name}.png",
            fmt="d",
        )

        if args.normalized:
            row_sums = cm.sum(axis=1, keepdims=True)
            cmn = np.divide(cm, row_sums, where=row_sums != 0)
            plot_matrix(
                cmn,
                title=f"{dataset} (row-normalized)\n{model}",
                out_path=args.out_dir / f"{name}__normalized.png",
                fmt=".2f",
            )

    print(f"Saved matrices to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

