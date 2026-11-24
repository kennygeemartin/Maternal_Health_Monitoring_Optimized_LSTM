"""
train_models.py
---------------
Trains and evaluates all classifiers for the maternal health risk study on the
cleaned primary dataset (data/processed/splits.npz):

  Baselines:  Logistic regression, k-nearest neighbours, SVM (RBF),
              ANN (PyTorch MLP)
  Proposed:   Bidirectional feature-serialised LSTM, configuration selected
              from a documented candidate grid by validation macro-F1
              (standardisation, label smoothing, cosine learning-rate decay)

All models share the same 70:15:15 stratified split (seed 42). Scaling is
fitted on the training partition only. Evaluation on the untouched test
partition: accuracy with 95% bootstrap CI, macro precision / recall / F1,
macro one-vs-rest ROC-AUC, multiclass Brier score, and per-class metrics.

Outputs (../results, ../figures, ../models): results.json, grid_log.json,
histories.json, predictions CSVs, saved weights for the proposed LSTM with
its scaler and model card.
"""
import json
import os
import pathlib
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(os.cpu_count() or 4)

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "figures"
MOD = ROOT / "models"
for d in (RES, FIG, MOD):
    d.mkdir(parents=True, exist_ok=True)

CLASSES = ["low risk", "mid risk", "high risk"]
PROPOSED = "Proposed LSTM"
BASELINES = ["Logistic regression", "kNN", "SVM (RBF)", "ANN baseline"]
DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------
# data helpers
# --------------------------------------------------------------------------
def load_splits():
    z = np.load(PROC / "splits.npz")
    return (z["X_train"], z["y_train"], z["X_val"], z["y_val"],
            z["X_test"], z["y_test"])


def to_seq(X):
    """Serialise each 8-feature record as a sequence of 8 one-dimensional
    steps, in the fixed feature order declared in models/model_card.json."""
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1)


# --------------------------------------------------------------------------
# neural models
# --------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, hidden=128, layers=1, bidirectional=True, dropout=0.3,
                 n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1]))


class MLPClassifier(nn.Module):
    def __init__(self, n_features=8, hidden=(64, 64), dropout=0.2, n_classes=3):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_torch(model, Xtr, ytr, Xva, yva, *, epochs, batch, lr, wd, patience,
                class_weight=None, label_smooth=0.0, seq=True, lr_decay=False):
    Xtr_t = to_seq(Xtr) if seq else torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = to_seq(Xva) if seq else torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.long)

    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch,
                        shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if lr_decay else None)
    crit = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smooth)

    hist = {"train_loss": [], "val_loss": [], "val_f1": []}
    best = {"val_loss": np.inf, "state": None, "epoch": 0}
    wait = 0
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        if sched:
            sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(Xva_t)
            vloss = crit(logits, yva_t).item()
            vf1 = f1_score(yva, logits.argmax(1).numpy(), average="macro")
        hist["train_loss"].append(tot / n)
        hist["val_loss"].append(vloss)
        hist["val_f1"].append(vf1)
        if vloss < best["val_loss"] - 1e-4:
            best = {"val_loss": vloss,
                    "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "epoch": ep}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best["state"])
    return model, hist, best["epoch"]


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def brier_multiclass(proba, y, n_classes=3):
    onehot = np.eye(n_classes)[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def bootstrap_acc_ci(y, pred, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append(accuracy_score(y[idx], pred[idx]))
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


def evaluate(name, proba, y):
    pred = proba.argmax(1)
    ci = bootstrap_acc_ci(y, pred)
    per = {}
    p = precision_score(y, pred, average=None, zero_division=0)
    r = recall_score(y, pred, average=None, zero_division=0)
    f = f1_score(y, pred, average=None, zero_division=0)
    for i, c in enumerate(CLASSES):
        per[c] = {"precision": float(p[i]), "recall": float(r[i]),
                  "f1": float(f[i]), "support": int((y == i).sum())}
    return {
        "model": name,
        "accuracy": float(accuracy_score(y, pred)),
        "accuracy_ci95": ci,
        "macro_precision": float(precision_score(y, pred, average="macro")),
        "macro_recall": float(recall_score(y, pred, average="macro")),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "macro_roc_auc": float(roc_auc_score(y, proba, multi_class="ovr",
                                             average="macro")),
        "brier": brier_multiclass(proba, y),
        "per_class": per,
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def softmax(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def torch_proba(model, X, seq=True):
    model.eval()
    Xt = to_seq(X) if seq else torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return softmax(model(Xt).numpy())


# --------------------------------------------------------------------------
# main experiment
# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    Xtr, ytr, Xva, yva, Xte, yte = load_splits()
    results = {n: {} for n in BASELINES + [PROPOSED]}
    histories = {}
    test_probas = {}

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)

    # ---- 1. Logistic regression --------------------------------------------
    print("training logistic regression ...", flush=True)
    lr_model = LogisticRegression(max_iter=2000, random_state=SEED)
    lr_model.fit(Xtr_s, ytr)
    test_probas["Logistic regression"] = lr_model.predict_proba(Xte_s)

    # ---- 2. k-nearest neighbours -------------------------------------------
    print("training kNN ...", flush=True)
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(Xtr_s, ytr)
    test_probas["kNN"] = knn.predict_proba(Xte_s)

    # ---- 3. SVM (RBF kernel) ------------------------------------------------
    print("training SVM (RBF) ...", flush=True)
    svm = SVC(kernel="rbf", probability=True, random_state=SEED)
    svm.fit(Xtr_s, ytr)
    test_probas["SVM (RBF)"] = svm.predict_proba(Xte_s)

    # ---- 4. ANN baseline (torch MLP) ----------------------------------------
    print("training ANN baseline ...", flush=True)
    ann = MLPClassifier()
    ann, hist_ann, ep_ann = train_torch(
        ann, Xtr_s, ytr, Xva_s, yva,
        epochs=200, batch=256, lr=1e-3, wd=1e-4, patience=20, seq=False,
        lr_decay=True)
    histories["ANN baseline"] = hist_ann
    test_probas["ANN baseline"] = torch_proba(ann, Xte_s, seq=False)
    results["ANN baseline"] = {"best_epoch": ep_ann}

    # ---- 5. Proposed LSTM: configuration grid -------------------------------
    print("training proposed LSTM candidates ...", flush=True)
    counts = np.bincount(ytr, minlength=3)
    cw = torch.tensor(len(ytr) / (3 * counts), dtype=torch.float32)
    grid = [
        {"tag": "G1", "hidden": 64,  "layers": 1, "bidirectional": False,
         "dropout": 0.2, "batch": 256, "lr": 1e-3, "wd": 1e-4,
         "label_smooth": 0.0,  "class_weight": True},
        {"tag": "G2", "hidden": 128, "layers": 1, "bidirectional": True,
         "dropout": 0.3, "batch": 256, "lr": 1e-3, "wd": 1e-4,
         "label_smooth": 0.0,  "class_weight": True},
        {"tag": "G3", "hidden": 128, "layers": 1, "bidirectional": True,
         "dropout": 0.3, "batch": 256, "lr": 1e-3, "wd": 1e-4,
         "label_smooth": 0.05, "class_weight": True},
        {"tag": "G4", "hidden": 128, "layers": 1, "bidirectional": True,
         "dropout": 0.3, "batch": 256, "lr": 3e-4, "wd": 1e-4,
         "label_smooth": 0.05, "class_weight": True},
    ]
    best_val_f1, best_cfg, best_model, best_hist, best_ep = -1, None, None, None, 0
    grid_log = []
    grid_path = RES / "grid_log.json"
    resume_ok = (grid_path.exists() and (MOD / "lstm_proposed.pt").exists())
    if resume_ok:
        prev = json.loads(grid_path.read_text())
        if len(prev) >= len(grid):
            grid_log = prev
            grid = []  # skip re-search; reload best weights below
    for cfg in grid:
        tag = cfg.pop("tag")
        use_cw = cfg.pop("class_weight")
        m = LSTMClassifier(hidden=cfg["hidden"], layers=cfg["layers"],
                           bidirectional=cfg["bidirectional"],
                           dropout=cfg["dropout"])
        m, h, ep = train_torch(
            m, Xtr_s, ytr, Xva_s, yva, epochs=150, batch=cfg["batch"],
            lr=cfg["lr"], wd=cfg["wd"], patience=15,
            class_weight=cw if use_cw else None,
            label_smooth=cfg["label_smooth"], seq=True, lr_decay=True)
        val_f1 = max(h["val_f1"])
        grid_log.append({"tag": tag, **cfg, "class_weight": use_cw,
                         "best_val_loss": min(h["val_loss"]),
                         "best_val_macro_f1": float(val_f1), "best_epoch": ep})
        grid_path.write_text(json.dumps(grid_log, indent=2))
        print(f"  {tag}: val macro-F1 {val_f1:.4f} (best epoch {ep})", flush=True)
        if val_f1 > best_val_f1:
            best_val_f1, best_cfg, best_model, best_hist, best_ep = (
                val_f1, {"tag": tag, **cfg, "class_weight": use_cw}, m, h, ep)
        # persist the current best weights immediately (crash-safe)
        torch.save(best_model.state_dict(), MOD / "lstm_proposed.pt")
        joblib.dump(sc, MOD / "standard_scaler.joblib")
        (RES / "histories.json").write_text(json.dumps(
            histories | {PROPOSED: best_hist}, indent=2))

    if resume_ok and not grid:
        sel = max(grid_log, key=lambda g: g["best_val_macro_f1"])
        best_val_f1 = sel["best_val_macro_f1"]
        best_ep = sel["best_epoch"]
        best_cfg = {k: v for k, v in sel.items()
                    if k not in ("best_val_loss", "best_val_macro_f1",
                                 "best_epoch")}
        best_model = LSTMClassifier(hidden=sel["hidden"], layers=sel["layers"],
                                    bidirectional=sel["bidirectional"],
                                    dropout=sel["dropout"])
        best_model.load_state_dict(torch.load(MOD / "lstm_proposed.pt"))
        prev_hist = json.loads((RES / "histories.json").read_text())
        best_hist = prev_hist.get(PROPOSED) or prev_hist.get("LSTM (improved)")
        print(f"  resumed grid search; selected {sel['tag']} "
              f"(val macro-F1 {best_val_f1:.4f})", flush=True)

    histories[PROPOSED] = best_hist
    test_probas[PROPOSED] = torch_proba(best_model, Xte_s)
    results[PROPOSED] = {"best_epoch": best_ep,
                         "selected_config": best_cfg,
                         "grid": grid_log}

    # ---- evaluate all -------------------------------------------------------
    print("evaluating on held-out test partition ...", flush=True)
    slug = {"Logistic regression": "lr", "kNN": "knn", "SVM (RBF)": "svm",
            "ANN baseline": "ann", PROPOSED: "lstm"}
    for name, proba in test_probas.items():
        m = evaluate(name, proba, yte)
        results[name].update(m)
        pd.DataFrame({"y_true": yte, "y_pred": proba.argmax(1),
                      "p_low": proba[:, 0], "p_mid": proba[:, 1],
                      "p_high": proba[:, 2]}).to_csv(
            RES / f"test_predictions_{slug[name]}.csv", index=False)
        print(f"  {name}: acc {m['accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}  "
              f"AUC {m['macro_roc_auc']:.4f}  Brier {m['brier']:.4f}", flush=True)

    # ---- persist artefacts ---------------------------------------------------
    torch.save(best_model.state_dict(), MOD / "lstm_proposed.pt")
    joblib.dump(sc, MOD / "standard_scaler.joblib")
    model_card = {
        "model": "Proposed LSTM (optimised bidirectional feature-serialised)",
        "feature_order": ["Age", "Body Temperature(F)", "Heart rate(bpm)",
                          "Systolic Blood Pressure(mm Hg)",
                          "Diastolic Blood Pressure(mm Hg)", "BMI(kg/m 2)",
                          "Blood Glucose(HbA1c)",
                          "Blood Glucose(Fasting hour-mg/dl)"],
        "classes": CLASSES,
        "scaling": "StandardScaler fitted on the training partition",
        "config": best_cfg,
        "seed": SEED,
        "framework": f"PyTorch {torch.__version__} (CPU)",
        "note": ("Static records serialised as 8 one-dimensional steps; "
                 "not a temporal/longitudinal model."),
    }
    (MOD / "model_card.json").write_text(json.dumps(model_card, indent=2))

    results["_runtime_seconds"] = time.time() - t0
    (RES / "results.json").write_text(json.dumps(results, indent=2))
    (RES / "histories.json").write_text(json.dumps(histories, indent=2))
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
