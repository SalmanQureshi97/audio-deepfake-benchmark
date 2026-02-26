# OOD Audio Deepfake Benchmark

This repository contains the code used to generate leakage-safe OOD evaluation splits and benchmark results for:

- Deezer Deepfake Detector (`specnn_amplitude`)
- SONICS SpecTTTra pretrained models

Datasets used:

- FMA (real + AE reconstructions)
- FakeMusicCaps (fake)
- SONICS (real + fake)

## What This Repo Includes
Reproducible out-of-domain benchmark pipeline for evaluating Deezer’s deepfake detector and SONICS SpecTTTra models on FMA, FakeMusicCaps, and SONICS with leakage-safe validation splits, standardized manifests, and confusion-matrix reporting.

## What This Folder Includes

- Split generation scripts
- Evaluation orchestration scripts
- Prediction summarization
- Confusion matrix generation
- Final comparison CSV outputs

This repo does **not** host dataset audio or model checkpoints.
This folder does **not** contain dataset audio or model checkpoints.

---

## Assumed Directory Structure

The scripts assume this layout under your workspace root:

```text
<ROOT>/
├── deepfake-detector/
│   ├── data/
│   │   └── dataset_medium_split_filtered.npy
│   ├── weights/
│   │   └── final/
│   │       └── specnn_amplitude/...
│   ├── fma_real_medium/
│   │   └── resampled/...
│   ├── fma_rebuilt_medium/
│   │   ├── encodec3/...
│   │   ├── encodec6/...
│   │   ├── encodec24/...
│   │   ├── griffin256/...
│   │   ├── griffin512/...
│   │   ├── lac14/...
│   │   ├── lac2/...
│   │   ├── lac7/...
│   │   └── musika/...
│   └── scripts/
│       └── eval_manifest.py
├── sonics/
│   └── benchmark_pretrained.py
├── SONICS/
│   ├── train.csv
│   ├── valid.csv
│   └── (audio files referenced by CSV filepath)
├── FakeMusicCaps/
│   ├── audioldm2/...
│   ├── MusicGen_medium/...
│   ├── musicldm/...
│   ├── mustango/...
│   └── stable_audio_open/...
└── ood_eval/
    ├── build_balanced_ood_splits.py
    ├── summarize_predictions.py
    ├── plot_confusion_matrices.py
    ├── final_ood_comparison.csv
    └── confusion/
```

---

## Reproducibility Notes

- FMA samples are selected from Deezer **validation** split (`dataset_medium_split_filtered.npy`).
- SONICS samples are selected from `SONICS/valid.csv` with overlap checks against `SONICS/train.csv`.
- FMA fake split is stratified across AE encoders:
  `encodec3, encodec6, encodec24, griffin256, griffin512, lac14, lac2, lac7, musika`.
- Target label convention in manifests:
  - `0 = real`
  - `1 = fake`

---

## 1) Generate Splits

```bash
python ood_eval/build_balanced_ood_splits.py \
  --deezer_split_npy deepfake-detector/data/dataset_medium_split_filtered.npy \
  --deezer_split_name valid \
  --fma_real_root <ROOT>/deepfake-detector/fma_real_medium/resampled \
  --fma_fake_root <ROOT>/deepfake-detector/fma_rebuilt_medium \
  --fakemusiccaps_root <ROOT>/FakeMusicCaps \
  --sonics_root <ROOT>/SONICS \
  --sonics_eval_csv <ROOT>/SONICS/valid.csv \
  --n_fma_real 2000 \
  --n_fma_fake 2000 \
  --n_sonics_real 2000 \
  --n_sonics_fake 2000 \
  --n_fakemusiccaps_fake 2000 \
  --out_dir ood_eval/manifests \
  --seed 42
```

---

## 2) Run Deezer Evaluation

Example:

```bash
python deepfake-detector/scripts/eval_manifest.py \
  --config specnn_amplitude \
  --weights specnn_amplitude \
  --manifest_csv ood_eval/manifests/fma_fake_valid_2000.csv \
  --output_csv ood_eval/preds/deezer__fma_fake_valid_2000.csv \
  --gpu 0 \
  --repeat 1 \
  --model_n_encoders 10 \
  --eval_sections all \
  --threshold 0.5
```

---

## 3) Run SONICS Evaluation

```bash
python sonics/benchmark_pretrained.py \
  --models awsaf49/sonics-spectttra-alpha-5s,awsaf49/sonics-spectttra-beta-5s,awsaf49/sonics-spectttra-gamma-5s,awsaf49/sonics-spectttra-alpha-120s,awsaf49/sonics-spectttra-beta-120s,awsaf49/sonics-spectttra-gamma-120s \
  --test_csv ood_eval/manifests/fma_fake_valid_2000.csv \
  --output_dir ood_eval/preds/sonics/fma_fake_valid_2000
```

---

## 4) Summarize Predictions

Per prediction file:

```bash
python ood_eval/summarize_predictions.py \
  --input_csv <predictions.csv> \
  --output_csv <metrics.csv> \
  --model_name "<model>" \
  --dataset_name "<dataset>"
```

---

## 5) Plot Confusion Matrices

```bash
python ood_eval/plot_confusion_matrices.py \
  --input_csv ood_eval/final_ood_comparison.csv \
  --out_dir ood_eval/confusion \
  --normalized
```

---

## Outputs

- Manifests: `ood_eval/manifests/*.csv`
- Deezer predictions: `ood_eval/preds/deezer__*.csv`
- SONICS predictions: `ood_eval/preds/sonics/.../test_predictions.csv`
- Final metrics: `ood_eval/final_ood_comparison.csv`
- Confusion matrices: `ood_eval/confusion/*.png`
