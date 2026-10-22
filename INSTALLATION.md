# Installation & Execution Guide

## 1. Requirements

- Python 3.9+
- Git
- PyTorch and torchvision
- Pillow
- Matplotlib
- Optional CUDA-compatible GPU

## 2. Clone the Repository

```bash
git clone https://github.com/anushadayal06-tech/MUJ-DS-2430010333.git
cd MUJ-DS-2430010333
```

## 3. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install torch torchvision pillow matplotlib
```

## 5. Project Layout

The implementation is under `capstone/`:

```text
capstone/
├── src/
├── data/
│   └── splits.csv
├── models/
└── results/
```

The raw image dataset and trained checkpoint are not committed. The split CSV must reference image files available in the local dataset environment.

## 6. Training

After placing the required split CSV at `capstone/data/splits.csv` and making the referenced images available:

```bash
cd capstone
python src/train.py --arch resnet18
```

Supported architectures are `resnet18`, `efficientnet_b0`, and `vit_b_16`. PGD adversarial training can be enabled with `--adv`.

## 7. Adversarial Evaluation

After a compatible trained checkpoint is available under `capstone/models/`:

```bash
python src/attacks.py --arch resnet18 --name resnet18
```

## 8. Reproducibility

For a full rerun, use the same dataset/split metadata, preprocessing, model configuration, hyperparameters, hardware setup, and evaluation protocol used for the committed experiment results.
