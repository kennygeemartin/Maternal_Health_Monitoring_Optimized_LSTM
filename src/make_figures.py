"""
make_figures.py
---------------
Generates all manuscript figures from results/histories and predictions:
  fig1_class_distribution.png  - class frequencies of the cleaned primary data
  fig2_learning_curves.png     - train/val loss, proposed LSTM
  fig3_model_comparison.png    - accuracy / macro-F1 / ROC-AUC comparison
  fig4_confusion_matrix.png    - proposed LSTM confusion matrix (test set)
  fig5_roc_curves.png          - one-vs-rest ROC curves, proposed LSTM
  fig6_uci_confusion.png       - secondary (UCI) LSTM confusion matrix
All figures are original vector-style PNGs at 300 dpi (no screenshots).
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
CLASSES = ["low risk", "mid risk", "high risk"]
CMAP = ["#2a9d8f", "#e9c46a", "#e76f51"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 300})

results = json.loads((RES / "results.json").read_text())
histories = json.loads((RES / "histories.json").read_text())
report = json.loads((ROOT / "data/processed/cleaning_report.json").read_text())

# --- Fig 1: class distribution ---------------------------------------------
counts = report["primary"]["class_counts"]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
vals = [counts[c] for c in CLASSES]
bars = ax.bar(["Low", "Mid", "High"], vals, color=CMAP, edgecolor="black",
              linewidth=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 15, str(v), ha="center",
            fontsize=8)
ax.set_ylabel("Number of records")
ax.set_xlabel("Risk category")
fig.tight_layout()
fig.savefig(FIG / "fig1_class_distribution.png")
plt.close(fig)

# --- Fig 2: learning curves of the proposed LSTM ------------------------------
fig, ax = plt.subplots(figsize=(3.6, 2.7))
h = histories["Proposed LSTM"]
ax.plot(h["train_loss"], label="Training loss", color="#264653")
ax.plot(h["val_loss"], label="Validation loss", color="#e76f51")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-entropy loss")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig2_learning_curves.png")
plt.close(fig)

# --- Fig 3: model comparison --------------------------------------------------
order = ["Logistic regression", "kNN", "SVM (RBF)", "ANN baseline",
         "Proposed LSTM"]
labels = ["LR", "kNN", "SVM", "ANN", "LSTM\n(proposed)"]
acc = [results[m]["accuracy"] for m in order]
f1 = [results[m]["macro_f1"] for m in order]
aucv = [results[m]["macro_roc_auc"] for m in order]
x = np.arange(len(order))
w = 0.27
fig, ax = plt.subplots(figsize=(6.6, 2.8))
ax.bar(x - w, acc, w, label="Accuracy", color="#264653")
ax.bar(x, f1, w, label="Macro F1", color="#2a9d8f")
ax.bar(x + w, aucv, w, label="Macro ROC-AUC", color="#e9c46a")
ax.set_xticks(x, labels)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower center",
          bbox_to_anchor=(0.5, 1.0))
for xi, a in zip(x, acc):
    ax.text(xi - w, a + 0.02, f"{a:.3f}", ha="center", fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig3_model_comparison.png")
plt.close(fig)

# --- Fig 4: confusion matrix, proposed LSTM -----------------------------------
cm = np.array(results["Proposed LSTM"]["confusion_matrix"])
fig, ax = plt.subplots(figsize=(3.4, 2.9))
im = ax.imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks(range(3), ["Low", "Mid", "High"])
ax.set_yticks(range(3), ["Low", "Mid", "High"])
ax.set_xlabel("Predicted category")
ax.set_ylabel("Actual category")
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(FIG / "fig4_confusion_matrix.png")
plt.close(fig)

# --- Fig 5: ROC curves, proposed LSTM ------------------------------------------
pred = pd.read_csv(RES / "test_predictions_lstm.csv")
# ensure this file belongs to the proposed model
proba = pred[["p_low", "p_mid", "p_high"]].to_numpy()
y = pred["y_true"].to_numpy()
fig, ax = plt.subplots(figsize=(3.4, 2.9))
for i, c in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve((y == i).astype(int), proba[:, i])
    ax.plot(fpr, tpr, color=CMAP[i],
            label=f"{c.split()[0].capitalize()} (AUC = {auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fig5_roc_curves.png")
plt.close(fig)

# --- Fig 6: UCI secondary confusion matrix (if available) ----------------------
uci_path = RES / "uci_results.json"
if uci_path.exists():
    uci = json.loads(uci_path.read_text())
    cm = np.array(uci["Proposed LSTM (UCI)"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    im = ax.imshow(cm, cmap="Greens")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), ["Low", "Mid", "High"])
    ax.set_yticks(range(3), ["Low", "Mid", "High"])
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_uci_confusion.png")
    plt.close(fig)

print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
