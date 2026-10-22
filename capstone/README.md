# Capstone Project — Deepfake Detection using Deep Learning

**Student:** Anusha Dayal  
**Registration No.:** 2430010333  
**Section:** D  
**GitHub:** `anushadayal06-tech`

## Contents

- `src/` — model, dataset, training, adversarial attack, and comparison code
- `data/` — local split metadata when preparing a rerun
- `results/` — evaluation metrics, training logs, and robustness results
- `models/` — generated checkpoints; not committed because of size
- `screenshots/` — reserved for final demo/evidence screenshots

## Evaluation Summary

The committed ResNet18 evaluation reports **92.64% test accuracy**, **92.59% macro F1**, and **0.982 ROC-AUC** on **1,101 test samples**.

## Reproducibility

The raw image dataset, trained checkpoint, and environment-specific split CSV are intentionally excluded. Before training from scratch, prepare a compatible split CSV and update its image paths for the local dataset environment.

See the repository-level `INSTALLATION.md` for environment and execution instructions.