"""train.py - train any architecture, optionally with PGD adversarial training.
Examples:
  python src/train.py --arch resnet18
  python src/train.py --arch efficientnet_b0
  python src/train.py --arch vit_b_16
  python src/train.py --arch resnet18 --adv --name resnet18_adv
"""
import argparse, csv
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import get_model, get_device, count_params, DEFAULT_LR
from dataset import FrameDataset, train_transform, eval_transform
from attacks import pgd_attack

ROOT = Path(__file__).resolve().parent.parent
SPLITS_CSV, MODELS_DIR, RESULTS_DIR = ROOT / "data" / "splits.csv", ROOT / "models", ROOT / "results"


def run_epoch(model, loader, criterion, opt, device, train, adv=False, eps=0.01, adv_steps=3):
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train and adv:
            model.eval()
            x_adv = pgd_attack(model, x, y, criterion, eps, steps=adv_steps, alpha=eps / 2)
            model.train()
            x_in, y_in = torch.cat([x, x_adv]), torch.cat([y, y])
        else:
            x_in, y_in = x, y
        with torch.set_grad_enabled(train):
            out = model(x_in)
            loss = criterion(out, y_in)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * y_in.size(0)
        correct += (out.argmax(1) == y_in).sum().item()
        total += y_in.size(0)
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--adv", action="store_true", help="PGD adversarial training")
    ap.add_argument("--adv-eps", type=float, default=0.01)
    args = ap.parse_args()
    name = args.name or args.arch
    lr = args.lr or DEFAULT_LR[args.arch]

    device = get_device()
    train_ds = FrameDataset(SPLITS_CSV, "train", transform=train_transform)
    val_ds = FrameDataset(SPLITS_CSV, "val", transform=eval_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = get_model(args.arch, pretrained=True).to(device)
    print(f"{name} | trainable params: {count_params(model):,} | lr={lr} | adv={args.adv} | device={device}")
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    MODELS_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)
    hist = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tl, ta = run_epoch(model, train_loader, criterion, opt, device, True, args.adv, args.adv_eps)
        model.eval()
        vl, va = run_epoch(model, val_loader, criterion, opt, device, False)
        for k, v in zip(hist, [tl, vl, ta, va]):
            hist[k].append(v)
        print(f"Epoch {ep}/{args.epochs} | train {tl:.4f}/{ta:.4f} | val {vl:.4f}/{va:.4f}")
        if va > best:
            best = va
            torch.save(model.state_dict(), MODELS_DIR / f"{name}.pth")
            print("  -> saved best")

    with open(RESULTS_DIR / f"{name}_train_log.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch"] + list(hist))
        for i in range(args.epochs):
            w.writerow([i + 1] + [hist[k][i] for k in hist])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ep = range(1, args.epochs + 1)
    ax[0].plot(ep, hist["train_loss"], label="Train"); ax[0].plot(ep, hist["val_loss"], label="Val")
    ax[0].set_title(f"{name} loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(ep, hist["train_acc"], label="Train"); ax[1].plot(ep, hist["val_acc"], label="Val")
    ax[1].set_title(f"{name} accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(RESULTS_DIR / f"{name}_training_curve.png", dpi=150)
    print(f"Best val acc: {best:.4f}")


if __name__ == "__main__":
    main()
