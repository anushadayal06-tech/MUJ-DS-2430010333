# Capstone Project — Deepfake Detection using Deep Learning

**Student:** Anusha Dayal  
**Registration No.:** 2430010333  
**Section:** D  
**GitHub:** `anushadayal06-tech`

## Contents

- `src/` — model, dataset, training, adversarial attack, and comparison code
- `data/` — split metadata required by the training scripts
- `results/` — committed evaluation metrics, training logs, and robustness results
- `models/` — generated checkpoints; not committed because of size
- `screenshots/` — reserved for final demo/evidence screenshots

## Implementation

The source supports ResNet18, EfficientNet-B0, and ViT-B/16 backbones. It also includes FGSM/PGD adversarial evaluation and optional PGD adversarial training.

## Evaluation Summary

The committed ResNet18 evaluation reports **92.64% test accuracy**, **92.59% macro F1**, and **0.982 ROC-AUC** on **1,101 test samples**. Detailed metrics and robustness results are stored under `results/`.

## Reproducibility

The repository contains the source implementation and experiment outputs. The raw image dataset and trained checkpoint are intentionally excluded. Before training from scratch, provide the required split CSV at `data/splits.csv` and the corresponding image files referenced by that CSV.

See the repository-level `INSTALLATION.md` for environment and execution instructions.
