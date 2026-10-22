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

This repository contains the implementation, experiments, evaluation results, literature review, presentation, and supporting documentation for the deepfake detection capstone project.

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
│   ├── literature_review.xlsx
│   └── README.md
├── presentations/
│   ├── capstone_presentation.pptx
│   └── README.md
└── capstone/
    ├── README.md
    ├── src/
    │   ├── model.py
    │   ├── dataset.py
    │   ├── train.py
    │   ├── attacks.py
    │   └── compare.py
    └── results/
        ├── resnet18_metrics.json
        ├── resnet18_train_log.csv
        └── resnet18_robustness.csv
```

## Results

The ResNet18 evaluation reports:

- Test accuracy: **92.64%**
- Macro precision: **92.72%**
- Macro recall: **92.51%**
- Macro F1: **92.59%**
- ROC-AUC: **0.982**
- Test samples: **1,101**

The repository also records FGSM and PGD robustness results across multiple attack strengths.

## Supporting Material

- [Literature Review](resources/literature_review.xlsx)
- [Capstone Presentation](presentations/capstone_presentation.pptx)

## Getting Started

See [INSTALLATION.md](INSTALLATION.md) for the environment and execution steps.

## Capstone

The capstone implementation and evidence are organized under [`capstone/`](capstone/). Source code is under [`capstone/src/`](capstone/src/) and experiment outputs are under [`capstone/results/`](capstone/results/).

## Development Workflow

- Use meaningful commits for substantial updates.
- Develop feature work on branches rather than directly on `main`.
- Use pull requests for review before merging.
- Track major tasks through GitHub Issues.
- Keep documentation synchronized with the implementation.
