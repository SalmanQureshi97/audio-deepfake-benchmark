#!/usr/bin/env python3
"""
Build leak-free OOD manifests with explicit per-source counts.

Output CSV columns:
filepath,target,skip_time,source_dataset,split_tag
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List
import random

import numpy as np
import pandas as pd


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
FMA_ENCODERS_DEFAULT = [
    "encodec3",
    "encodec6",
    "encodec24",
    "griffin256",
    "griffin512",
    "lac14",
    "lac2",
    "lac7",
    "musika",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--deezer_split_npy", type=Path, required=True)
    p.add_argument("--fma_real_root", type=Path, required=True)
    p.add_argument("--fma_fake_root", type=Path, default=None)
    p.add_argument("--fakemusiccaps_root", type=Path, required=True)
    p.add_argument("--sonics_root", type=Path, required=True)
    p.add_argument("--sonics_eval_csv", type=Path, default=None)
    p.add_argument("--sonics_train_csv", type=Path, default=None)
    p.add_argument(
        "--deezer_split_name",
        type=str,
        default="valid",
        choices=["train", "validation", "valid", "test"],
        help="Which split to use from Deezer split npy",
    )
    p.add_argument("--out_dir", type=Path, default=Path("ood_eval/manifests"))
    p.add_argument("--n_fma_real", type=int, default=2000)
    p.add_argument("--n_fma_fake", type=int, default=2000)
    p.add_argument("--n_sonics_real", type=int, default=2000)
    p.add_argument("--n_sonics_fake", type=int, default=2000)
    p.add_argument("--n_fakemusiccaps_fake", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fma_encoders",
        type=str,
        default=",".join(FMA_ENCODERS_DEFAULT),
        help="Comma-separated encoders for fair FMA-fake sampling",
    )
    return p.parse_args()


def to_abs(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path)


def sample_n(items: List[dict], n: int, rng: random.Random) -> List[dict]:
    if n > len(items):
        raise ValueError(f"Requested n={n}, but only {len(items)} items available")
    if n == len(items):
        return list(items)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx[:n]]


def sample_fma_fake_fair(
    fake_rows: List[dict], n: int, encoders: List[str], rng: random.Random
) -> List[dict]:
    by_enc = defaultdict(list)
    for r in fake_rows:
        e = r.get("encoder", "")
        if e in encoders:
            by_enc[e].append(r)

    for e in encoders:
        rng.shuffle(by_enc[e])

    k = len(encoders)
    base = n // k
    rem = n % k

    chosen = []
    taken = {e: 0 for e in encoders}

    # Pass 1: equal base allocation
    for e in encoders:
        want = min(base, len(by_enc[e]))
        chosen.extend(by_enc[e][:want])
        taken[e] += want

    # Pass 2: distribute remainder + deficits
    still_need = n - len(chosen)
    if still_need > 0:
        # deterministic round-robin over encoders
        idx = 0
        while still_need > 0:
            e = encoders[idx % k]
            if taken[e] < len(by_enc[e]):
                chosen.append(by_enc[e][taken[e]])
                taken[e] += 1
                still_need -= 1
            idx += 1
            # safety break if nothing left anywhere
            if idx > 10_000_000:
                break
            if all(taken[x] >= len(by_enc[x]) for x in encoders):
                break

    if len(chosen) < n:
        raise ValueError(
            f"Requested n={n} fair FMA-fake samples, but only {len(chosen)} available "
            f"across encoders={encoders}"
        )

    # Print resulting distribution
    out_counts = defaultdict(int)
    for r in chosen:
        out_counts[r["encoder"]] += 1
    print("[FMA-fake fair distribution] " + ", ".join(f"{e}:{out_counts[e]}" for e in encoders))
    return chosen


def cap_n(name: str, requested: int, available: int) -> int:
    n = min(requested, available)
    if n < requested:
        print(
            f"[WARN] {name}: requested {requested}, but only {available} available. Using {n}."
        )
    return n


def write_manifest(rows: Iterable[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["filepath", "target", "skip_time", "source_dataset", "split_tag"]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=cols)
    else:
        df = df[cols]
    df.to_csv(out_csv, index=False)


def index_audio_by_stem(root: Path) -> dict[str, list[Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    out: dict[str, list[Path]] = {}
    n = 0
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
            continue
        n += 1
        out.setdefault(p.stem, []).append(p)
    print(f"[INDEX] {root} -> {n} audio files, {len(out)} unique stems")
    return out


def normalize_deezer_split_name(name: str) -> str:
    return "validation" if name == "valid" else name


def build_fma_rows_from_split(
    deezer_split_npy: Path,
    split_name: str,
    fma_real_root: Path,
    fma_fake_root: Path,
) -> tuple[list[dict], list[dict]]:
    split = np.load(deezer_split_npy, allow_pickle=True).item()
    key = normalize_deezer_split_name(split_name)
    if key not in split:
        raise KeyError(f"Split key '{key}' not found in {deezer_split_npy}. Available: {list(split.keys())}")
    split_names = split[key]

    real_by_stem = index_audio_by_stem(fma_real_root)
    fake_by_stem = index_audio_by_stem(fma_fake_root)

    real_rows = []
    fake_rows = []
    miss_real = 0
    miss_fake = 0
    for name in split_names:
        stem = Path(name).stem
        if stem in real_by_stem:
            # deterministic first path
            p = sorted(real_by_stem[stem])[0]
            real_rows.append(
                {
                    "filepath": str(p.resolve()),
                    "target": 0,
                    "skip_time": 0.0,
                    "source_dataset": "FMA",
                    "split_tag": f"deezer_{key}_real",
                }
            )
        else:
            miss_real += 1

        if stem in fake_by_stem:
            paths = sorted(fake_by_stem[stem])
            for p in paths:
                rel = p.relative_to(fma_fake_root)
                encoder = rel.parts[0] if len(rel.parts) > 0 else "unknown"
                fake_rows.append(
                    {
                        "filepath": str(p.resolve()),
                        "target": 1,
                        "skip_time": 0.0,
                        "source_dataset": "FMA",
                        "split_tag": f"deezer_{key}_fake",
                        "encoder": encoder,
                    }
                )
        else:
            miss_fake += 1

    print(
        f"[FMA-{key}] real usable: {len(real_rows)} (missing: {miss_real}), "
        f"fake usable: {len(fake_rows)} (missing: {miss_fake})"
    )
    return real_rows, fake_rows


def build_fakemusiccaps_fake(root: Path) -> List[dict]:
    rows = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            rows.append(
                {
                    "filepath": str(p.resolve()),
                    "target": 1,
                    "skip_time": 0.0,
                    "source_dataset": "FakeMusicCaps",
                    "split_tag": "all_fake",
                }
            )
    print(f"[FakeMusicCaps] usable fake files: {len(rows)}")
    return rows


def build_sonics_rows(
    sonics_root: Path, sonics_eval_csv: Path, sonics_train_csv: Path | None, split_tag: str
) -> List[dict]:
    test_df = pd.read_csv(sonics_eval_csv, low_memory=False)
    train_paths = set()
    if sonics_train_csv and sonics_train_csv.exists():
        tr = pd.read_csv(sonics_train_csv, low_memory=False)
        train_paths = set(str(x) for x in tr["filepath"].tolist())

    rows = []
    skipped_train_overlap = 0
    missing = 0
    for _, r in test_df.iterrows():
        rel = str(r["filepath"])
        if rel in train_paths:
            skipped_train_overlap += 1
            continue
        full = to_abs(Path(rel), sonics_root)
        if not full.exists():
            missing += 1
            continue
        rows.append(
            {
                "filepath": str(full.resolve()),
                "target": int(r["target"]),
                "skip_time": float(r["skip_time"]) if "skip_time" in r and pd.notna(r["skip_time"]) else 0.0,
                "source_dataset": "SONICS",
                "split_tag": split_tag,
            }
        )
    print(
        f"[SONICS-{split_tag}] usable files: {len(rows)} "
        f"(train-overlap skipped: {skipped_train_overlap}, missing: {missing})"
    )
    return rows


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sonics_eval_csv = args.sonics_eval_csv or (args.sonics_root / "valid.csv")
    sonics_train_csv = args.sonics_train_csv or (args.sonics_root / "train.csv")
    fma_fake_root = args.fma_fake_root or (args.fma_real_root.parent.parent / "fma_rebuilt_medium")

    fma_encoders = [x.strip() for x in args.fma_encoders.split(",") if x.strip()]

    fma_real, fma_fake = build_fma_rows_from_split(
        args.deezer_split_npy,
        args.deezer_split_name,
        args.fma_real_root,
        fma_fake_root,
    )
    fmc_fake = build_fakemusiccaps_fake(args.fakemusiccaps_root)
    sonics = build_sonics_rows(
        args.sonics_root,
        sonics_eval_csv,
        sonics_train_csv,
        split_tag=Path(sonics_eval_csv).stem,
    )
    sonics_real = [x for x in sonics if x["target"] == 0]
    sonics_fake = [x for x in sonics if x["target"] == 1]

    # Per-source sampled sets
    n_fma_real = cap_n("FMA real", args.n_fma_real, len(fma_real))
    n_fma_fake = cap_n("FMA fake", args.n_fma_fake, len(fma_fake))
    n_sonics_real = cap_n("SONICS real", args.n_sonics_real, len(sonics_real))
    n_sonics_fake = cap_n("SONICS fake", args.n_sonics_fake, len(sonics_fake))
    n_fmc_fake = cap_n("FakeMusicCaps fake", args.n_fakemusiccaps_fake, len(fmc_fake))

    fma_real_n = sample_n(fma_real, n_fma_real, rng)
    fma_fake_n = sample_fma_fake_fair(fma_fake, n_fma_fake, fma_encoders, rng)
    sonics_real_n = sample_n(sonics_real, n_sonics_real, rng)
    sonics_fake_n = sample_n(sonics_fake, n_sonics_fake, rng)
    fmc_fake_n = sample_n(fmc_fake, n_fmc_fake, rng)

    fma_real_csv = args.out_dir / f"fma_real_valid_{len(fma_real_n)}.csv"
    fma_fake_csv = args.out_dir / f"fma_fake_valid_{len(fma_fake_n)}.csv"
    sonics_real_csv = args.out_dir / f"sonics_real_valid_{len(sonics_real_n)}.csv"
    sonics_fake_csv = args.out_dir / f"sonics_fake_valid_{len(sonics_fake_n)}.csv"
    fmc_fake_csv = args.out_dir / f"fakemusiccaps_fake_{len(fmc_fake_n)}.csv"

    write_manifest(fma_real_n, fma_real_csv)
    write_manifest(fma_fake_n, fma_fake_csv)
    write_manifest(sonics_real_n, sonics_real_csv)
    write_manifest(sonics_fake_n, sonics_fake_csv)
    write_manifest(fmc_fake_n, fmc_fake_csv)

    # Requested combined pool (can be class-imbalanced by design)
    requested = fma_real_n + sonics_real_n + fma_fake_n + sonics_fake_n + fmc_fake_n
    rng.shuffle(requested)
    write_manifest(requested, args.out_dir / "ood_requested_combined.csv")

    # Balanced combined (for direct model-vs-model comparability)
    all_real = fma_real_n + sonics_real_n
    all_fake = fma_fake_n + sonics_fake_n + fmc_fake_n
    n_bal = min(len(all_real), len(all_fake))
    balanced = sample_n(all_real, n_bal, rng) + sample_n(all_fake, n_bal, rng)
    rng.shuffle(balanced)
    write_manifest(balanced, args.out_dir / "ood_requested_combined_balanced.csv")

    # Compatibility files used by existing Deezer commands
    sonics_bal = sonics_real_n + sonics_fake_n
    rng.shuffle(sonics_bal)
    write_manifest(sonics_bal, args.out_dir / "ood_sonics_test_balanced.csv")

    fma_vs_fmc = fma_real_n + fmc_fake_n
    rng.shuffle(fma_vs_fmc)
    write_manifest(fma_vs_fmc, args.out_dir / "ood_fma_real_vs_fakemusiccaps_fake_balanced.csv")

    # three-source balanced from sampled pools
    n3 = min(len(all_real), len(all_fake))
    three = sample_n(all_real, n3, rng) + sample_n(all_fake, n3, rng)
    rng.shuffle(three)
    write_manifest(three, args.out_dir / "ood_three_source_balanced.csv")

    print(f"[OUT] {fma_real_csv.name} -> {len(fma_real_n)}")
    print(f"[OUT] {fma_fake_csv.name} -> {len(fma_fake_n)}")
    print(f"[OUT] {sonics_real_csv.name} -> {len(sonics_real_n)}")
    print(f"[OUT] {sonics_fake_csv.name} -> {len(sonics_fake_n)}")
    print(f"[OUT] {fmc_fake_csv.name} -> {len(fmc_fake_n)}")
    print(
        f"[OUT] ood_requested_combined.csv -> {len(requested)} "
        f"(real={len(all_real)}, fake={len(all_fake)})"
    )
    print(
        f"[OUT] ood_requested_combined_balanced.csv -> {len(balanced)} "
        f"(real={n_bal}, fake={n_bal})"
    )
    print(f"[OUT] ood_sonics_test_balanced.csv -> {len(sonics_bal)}")
    print(f"[OUT] ood_fma_real_vs_fakemusiccaps_fake_balanced.csv -> {len(fma_vs_fmc)}")
    print(f"[OUT] ood_three_source_balanced.csv -> {len(three)}")


if __name__ == "__main__":
    main()
