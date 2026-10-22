# Deepfake Detection using Deep Learning

## Student Details

| Field | Details |
|---|---|
| **Name** | Anusha Dayal |
| **Registration Number** | 2430010333 |
| **Branch** | Data Science |
| **Section** | D |
| **GitHub Username** | anushadayal06-tech |
| **Project Type** | Deep Learning Capstone Project |

## Project Overview

This repository contains the implementation, experiments, evaluation results, and supporting documentation for the deepfake detection capstone project.

The current implementation uses image-frame classification with pretrained CNN/transformer backbones and includes adversarial robustness evaluation using FGSM and PGD attacks.

## Repository Structure

```text
MUJ-DS-2430010333/
├── README.md
├── INSTALLATION.md
├── assignments/
├── notebooks/
├── code/
├── resources/
├── presentations/
└── capstone/
    ├── README.md
    ├── src/
    │   ├── model.py
    │   ├── dataset.py
    │   ├── train.py
    │   ├── attacks.py
    │   └── compare.py
    ├── data/
    │   └── splits.csv
    └── results/
        ├── resnet18_metrics.json
        ├── resnet18_train_log.csv
        └── resnet18_robustness.csv
```

## Results

The uploaded ResNet18 evaluation reports:

- Test accuracy: **92.64%**
- Macro precision: **92.72%**
- Macro recall: **92.51%**
- Macro F1: **92.59%**
- ROC-AUC: **0.982**
- Test samples: **1,101**

The repository also records FGSM and PGD robustness results across multiple attack strengths.

## Getting Started

See [INSTALLATION.md](INSTALLATION.md) for the environment and execution steps.

## Capstone

The capstone-specific implementation and evidence are organized under [`capstone/`](capstone/). The source code is under [`capstone/src/`](capstone/src/), the split metadata under [`capstone/data/`](capstone/data/), and experiment outputs under [`capstone/results/`](capstone/results/).

## Development Workflow

- Use meaningful commits for substantial updates.
- Develop feature work on branches rather than directly on `main`.
- Use pull requests for review before merging.
- Track major tasks through GitHub Issues.
- Keep documentation synchronized with the implementation.
