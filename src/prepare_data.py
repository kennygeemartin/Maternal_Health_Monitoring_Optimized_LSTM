"""
prepare_data.py
---------------
Reproducible data preparation for the maternal health risk classification study.

Primary source : Hossain, M. M. (2025). Maternal health risk data (Version 1)
                 [Data set]. Mendeley Data. https://doi.org/10.17632/ckhgtgdg4f.1
Secondary source: Ahmed, M. (2020). Maternal health risk [Data set].
                 UCI Machine Learning Repository. https://doi.org/10.24432/C5DP5D

Cleaning rules (documented, deterministic, applied before any split):
  1. Drop identifier columns (Patient ID, Name).
  2. Remove exact duplicate rows on (features + label).
  3. Remove rows with physiologically implausible values, using the wide
     validation bounds of the prototype application and routine clinical
     plausibility ranges:
        Age                10-60 years
        Body temperature   90-110 degF
        Heart rate         30-220 bpm
        Systolic BP        60-260 mmHg
        Diastolic BP       30-160 mmHg
        BMI                10-70 kg/m^2
        Glucose columns    > 0
  4. Stratified record-level split 70 : 15 : 15 (train : val : test),
     random seed 42, matching the protocol reported in the project report.

Outputs (written to ../data/processed):
  cleaned_primary.csv, cleaned_uci.csv, splits.npz, cleaning_report.json
"""
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = DATA / "processed"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Age",
    "Body Temperature(F)",
    "Heart rate(bpm)",
    "Systolic Blood Pressure(mm Hg)",
    "Diastolic Blood Pressure(mm Hg)",
    "BMI(kg/m 2)",
    "Blood Glucose(HbA1c)",
    "Blood Glucose(Fasting hour-mg/dl)",
]
LABEL = "Status"
CLASSES = ["low risk", "mid risk", "high risk"]  # fixed class order

BOUNDS = {
    "Age": (10, 60),
    "Body Temperature(F)": (90.0, 110.0),
    "Heart rate(bpm)": (30, 220),
    "Systolic Blood Pressure(mm Hg)": (60, 260),
    "Diastolic Blood Pressure(mm Hg)": (30, 160),
    "BMI(kg/m 2)": (10, 70),
    "Blood Glucose(HbA1c)": (0, np.inf),
    "Blood Glucose(Fasting hour-mg/dl)": (0, np.inf),
}


def clean_primary() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(DATA / "Mathernal_Risk.csv")
    raw.columns = [c.strip() for c in raw.columns]
    report = {"raw_rows": int(len(raw))}

    df = raw.drop(columns=["Patient ID", "Name"])
    df[LABEL] = df[LABEL].str.strip().str.lower()

    dup = df.duplicated(subset=FEATURES + [LABEL]).sum()
    df = df.drop_duplicates(subset=FEATURES + [LABEL])
    report["duplicates_removed"] = int(dup)

    implausible = pd.Series(False, index=df.index)
    per_rule = {}
    for col, (lo, hi) in BOUNDS.items():
        bad = (df[col] < lo) | (df[col] > hi)
        per_rule[col] = int(bad.sum())
        implausible |= bad
    df = df.loc[~implausible].reset_index(drop=True)
    report["implausible_removed_per_feature"] = per_rule
    report["implausible_rows_removed"] = int(implausible.sum())
    report["cleaned_rows"] = int(len(df))
    report["class_counts"] = df[LABEL].value_counts().to_dict()
    return df, report


def clean_uci() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(DATA / "uci" / "Maternal Health Risk Data Set.csv")
    raw.columns = [c.strip() for c in raw.columns]
    report = {"raw_rows": int(len(raw))}
    df = raw.rename(columns={"RiskLevel": "Status"})
    df[LABEL] = df[LABEL].str.strip().str.lower()
    dup = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = int(dup)
    report["cleaned_rows"] = int(len(df))
    report["class_counts"] = df[LABEL].value_counts().to_dict()
    return df, report


def main() -> None:
    primary, rep_primary = clean_primary()
    uci, rep_uci = clean_uci()
    primary.to_csv(OUT / "cleaned_primary.csv", index=False)
    uci.to_csv(OUT / "cleaned_uci.csv", index=False)

    y = primary[LABEL].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    X = primary[FEATURES].to_numpy(dtype=np.float64)

    # stratified 70:15:15, seed 42
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )
    np.savez(
        OUT / "splits.npz",
        X_train=X_tr, y_train=y_tr,
        X_val=X_va, y_val=y_va,
        X_test=X_te, y_test=y_te,
    )
    split_counts = {
        "train": int(len(y_tr)), "val": int(len(y_va)), "test": int(len(y_te)),
        "train_class_counts": np.bincount(y_tr, minlength=3).tolist(),
        "val_class_counts": np.bincount(y_va, minlength=3).tolist(),
        "test_class_counts": np.bincount(y_te, minlength=3).tolist(),
    }
    report = {
        "seed": SEED,
        "features": FEATURES,
        "classes": CLASSES,
        "primary": rep_primary,
        "uci": rep_uci,
        "splits": split_counts,
    }
    (OUT / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
