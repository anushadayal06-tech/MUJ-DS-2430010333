"""compare.py - merges model results into comparison.csv and generates a robustness plot.
Usage: python src/compare.py
"""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"
rows, curves = [], {}
for mf in sorted(RESULTS.glob("*_metrics.json")):
    m = json.loads(mf.read_text()); n = m["model"]
    rc = RESULTS / f"{n}_robustness.csv"
    pgd = fgsm = ""
    if rc.exists():
        data = list(csv.DictReader(open(rc)))
        curves[n] = data
        pick = [r for r in data if abs(float(r["epsilon"]) - 0.01) < 1e-9][0]
        fgsm, pgd = float(pick["fgsm_accuracy"]), float(pick["pgd_accuracy"])
    rows.append({"Model": n, "Accuracy": m["accuracy"], "Precision": m["precision_macro"],
                 "Recall": m["recall_macro"], "F1": m["f1_macro"], "AUC": m["roc_auc"],
                 "Params": m.get("params", ""), "FGSM_acc@eps0.01": fgsm, "PGD_acc@eps0.01": pgd})

if rows:
    with open(RESULTS / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(open(RESULTS / "comparison.csv").read())

if curves:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, atk in enumerate(["fgsm_accuracy", "pgd_accuracy"]):
        for n, d in curves.items():
            ax[i].plot([float(r["epsilon"]) for r in d], [float(r[atk]) for r in d], marker="o", label=n)
        ax[i].set_title(atk.split("_")[0].upper()); ax[i].set_xlabel("Epsilon"); ax[i].set_ylabel("Accuracy")
        ax[i].grid(alpha=.3); ax[i].legend()
    fig.tight_layout(); fig.savefig(RESULTS / "robustness_comparison.png", dpi=150)
    print("Saved results/comparison.csv, results/robustness_comparison.png")
