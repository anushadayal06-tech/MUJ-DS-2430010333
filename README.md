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

The implementation uses pretrained deep-learning backbones for image-frame classification and includes FGSM/PGD adversarial evaluation and optional PGD adversarial training.

## Repository Structure

```text
MUJ-DS-2430010333/
├── README.md
├── INSTALLATION.md
├── assignments/
├── notebooks/
├── code/
├── resources/
│   └── literature_review.xlsx
├── presentations/
│   └── capstone_presentation.pptx
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
    ├── models/
    └── results/
        ├── resnet18_metrics.json
        ├── resnet18_train_log.csv
        └── resnet18_robustness.csv
```

## Reported Results

The committed ResNet18 evaluation reports:

- **Test accuracy:** 92.64%
- **Macro precision:** 92.72%
- **Macro recall:** 92.51%
- **Macro F1:** 92.59%
- **ROC-AUC:** 0.982
- **Test samples:** 1,101

FGSM and PGD robustness results are included under `capstone/results/`.

## Supporting Material

- [Literature Review](resources/literature_review.xlsx)
- [Capstone Presentation](presentations/capstone_presentation.pptx)
- [Installation & Execution Guide](INSTALLATION.md)

## Reproducibility Note

The source implementation and experiment outputs are committed. The raw image dataset and trained model checkpoint are not included in the repository. The split metadata used by the training scripts should be available at `capstone/data/splits.csv` together with the image files referenced by that CSV before rerunning training.

## Development Workflow

Feature work is developed on branches and merged through pull requests. Major project tasks are tracked through GitHub Issues, and documentation is kept alongside the implementation.
