# Deepfake Detection using Deep Learning

## Student Details

| Field | Details |
|---|---|
| **Name** | Anusha Dayal |
| **Registration Number** | 2430010333 |
| **Section** | D |
| **GitHub Username** | anushadayal06-tech |
| **Project Type** | Deep Learning Capstone Project |

## Project Overview

This repository contains the implementation, experiment outputs, literature review, presentation, and supporting documentation for the deepfake detection capstone project.

## Results

The committed ResNet18 evaluation reports **92.64% test accuracy**, **92.59% macro F1**, and **0.982 ROC-AUC** on **1,101 test samples**. FGSM and PGD robustness results are included under `capstone/results/`.

## Supporting Material

- [Literature Review](resources/literature_review.xlsx)
- [Capstone Presentation](presentations/capstone_presentation.pptx)
- [Installation & Execution Guide](INSTALLATION.md)

## Reproducibility

The source implementation and experiment outputs are committed. The raw image dataset and trained model checkpoint are not included. Before rerunning training, provide `capstone/data/splits.csv` and the referenced image files.