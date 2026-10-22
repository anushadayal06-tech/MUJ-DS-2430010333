# Installation & Execution Guide

## 1. Requirements

- Python 3.9+
- Git
- PyTorch and torchvision
- Pillow
- Matplotlib
- A working deep-learning environment compatible with the packages used by the project
- Optional GPU/CUDA support

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

Install the packages required by the source code:

```bash
pip install torch torchvision pillow matplotlib
```

## 5. Project Layout

The capstone implementation is under `capstone/`:

```text
capstone/
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── attacks.py
│   └── compare.py
├── data/
│   └── splits.csv
└── results/
```

## 6. Running Training

From the repository root:

```bash
cd capstone
python src/train.py --arch resnet18
```

Other supported architectures are `efficientnet_b0` and `vit_b_16`. PGD adversarial training can be enabled with `--adv`.

## 7. Adversarial Evaluation

After the required trained checkpoint is available under `capstone/models/`, run:

```bash
python src/attacks.py --arch resnet18 --name resnet18
```

The script evaluates FGSM and PGD over the configured epsilon sweep.

## 8. Reproducibility

For a reproducible submission, document the dataset source/access, preprocessing, split construction, training configuration, model architecture, hyperparameters, hardware, and evaluation protocol used to produce the committed results.
