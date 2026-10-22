"""attacks.py - FGSM / PGD (white-box, L-inf) + robustness sweep.
Usage: python src/attacks.py --arch resnet18 --name resnet18
"""
import argparse, csv
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader

from model import get_model, get_device
from dataset import FrameDataset, eval_transform

ROOT = Path(__file__).resolve().parent.parent
SPLITS_CSV, MODELS_DIR, RESULTS_DIR = ROOT / "data" / "splits.csv", ROOT / "models", ROOT / "results"
EPSILONS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]


def fgsm_attack(model, x, y, criterion, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = criterion(model(x), y)
    grad, = torch.autograd.grad(loss, x)
    return torch.clamp(x + eps * grad.sign(), 0, 1).detach()


def pgd_attack(model, x, y, criterion, eps, steps=7, alpha=None):
    alpha = alpha or max(eps / 4, 1e-4)
    orig = x.clone().detach()
    adv = orig.clone()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = criterion(model(adv), y)
        grad, = torch.autograd.grad(loss, adv)
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.clamp(orig + torch.clamp(adv - orig, -eps, eps), 0, 1).detach()
    return adv


def acc_under_attack(model, loader, criterion, device, attack, eps):
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if eps > 0:
            x = attack(model, x, y, criterion, eps)
        with torch.no_grad():
            correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--name", default=None, help="checkpoint/result name, e.g. resnet18_adv")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    name = args.name or args.arch

    device = get_device()
    model = get_model(args.arch, pretrained=False)
    model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pth", map_location=device))
    model.to(device).eval()

    ds = FrameDataset(SPLITS_CSV, "test", transform=eval_transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    rows = []
    for eps in EPSILONS:
        f = acc_under_attack(model, loader, criterion, device, fgsm_attack, eps)
        p = f if eps == 0 else acc_under_attack(model, loader, criterion, device, pgd_attack, eps)
        print(f"[{name}] eps={eps:.3f} FGSM={f:.4f} PGD={p:.4f}")
        rows.append({"epsilon": eps, "fgsm_accuracy": f, "pgd_accuracy": p})

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}_robustness.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["epsilon", "fgsm_accuracy", "pgd_accuracy"])
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
