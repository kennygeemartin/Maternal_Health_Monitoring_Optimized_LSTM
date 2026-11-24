"""
train_uci.py
------------
Exploratory secondary-data experiment: fits the proposed LSTM configuration on
the cleaned UCI Maternal Health Risk dataset (452 rows after duplicate
removal), using the same protocol (stratified 70:15:15, seed 42,
standardisation fitted on the training partition).

This is a *separately fitted* benchmark, not external validation: the UCI file
has six features with a different schema and unit conventions from the primary
eight-feature model, so the two models are not interchangeable.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from train_models import (CLASSES, LSTMClassifier, evaluate, torch_proba,
                          train_torch)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"

UCI_FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp",
                "HeartRate"]


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "cleaned_uci.csv")
    df.columns = [c.strip() for c in df.columns]
    X = df[UCI_FEATURES].to_numpy(dtype=np.float64)
    y = df["Status"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

    counts = np.bincount(y_tr, minlength=3)
    cw = torch.tensor(len(y_tr) / (3 * counts), dtype=torch.float32)
    sc = StandardScaler().fit(X_tr)

    model = LSTMClassifier(hidden=128, layers=1, bidirectional=True,
                           dropout=0.3)
    model, hist, ep = train_torch(
        model, sc.transform(X_tr), y_tr, sc.transform(X_va), y_va,
        epochs=300, batch=32, lr=1e-3, wd=1e-4, patience=30,
        class_weight=cw, label_smooth=0.05, seq=True, lr_decay=True)

    metrics = evaluate("Proposed LSTM (UCI)", torch_proba(model, sc.transform(X_te)), y_te)
    out = {"Proposed LSTM (UCI)": metrics,
           "n_train": len(y_tr), "n_val": len(y_va), "n_test": len(y_te),
           "best_epoch": ep,
           "note": ("Separately fitted secondary benchmark; six-feature UCI "
                    "schema, not external validation of the primary model.")}
    (RES / "uci_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"},
                     indent=2))


if __name__ == "__main__":
    main()
