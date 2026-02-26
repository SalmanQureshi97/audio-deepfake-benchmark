# OOD Audio Deepfake Benchmark

This repository contains the code used to generate leakage-safe OOD evaluation splits and benchmark results for:

- Deezer Deepfake Detector (`specnn_amplitude`)
- SONICS SpecTTTra pretrained models

Datasets used:

- FMA (real + AE reconstructions)
- FakeMusicCaps (fake)
- SONICS (real + fake)

## What This Repo Includes

- Split generation scripts
- Evaluation orchestration scripts
- Prediction summarization
- Confusion matrix generation
- Final comparison CSV outputs

This repo does **not** host dataset audio or model checkpoints.

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
