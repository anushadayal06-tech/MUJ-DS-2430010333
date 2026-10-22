# Capstone Project

## Deepfake Detection using Deep Learning

**Student:** Anusha Dayal  
**Registration No.:** 2430010333  
**Section:** D  
**GitHub:** `anushadayal06-tech`

### Project contents

This folder contains the final capstone implementation, experiment outputs, and supporting evidence.

- `src/` — model, dataset, training, adversarial attack, and comparison code
- `data/` — dataset split metadata
- `results/` — evaluation metrics, training logs, and robustness results
- `screenshots/` — interface, prediction, training, and result screenshots when available

### Implementation

The implementation supports ResNet18, EfficientNet-B0, and ViT-B/16 backbones. It also includes FGSM/PGD adversarial evaluation and optional PGD adversarial training.

### Current evaluation

The uploaded ResNet18 evaluation contains test accuracy, precision, recall, F1, specificity, ROC-AUC, a confusion matrix, and 1,101 test samples. Robustness results are recorded separately for FGSM and PGD across multiple epsilon values.

### Documentation checklist

- [ ] Problem statement
- [ ] Objectives
- [ ] Dataset and preprocessing
- [ ] Model architecture
- [ ] Training procedure
- [ ] Evaluation methodology
- [x] Results
- [ ] Limitations
- [ ] Future scope
- [ ] Individual contribution
- [ ] Demo / screenshots
