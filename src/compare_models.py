"""
Churn model comparison on identical data splits.

Compares four classifiers so the paper's "Random Forest" choice is backed by
evidence instead of assertion:
  - Majority-class baseline (always predicts the most common class)
  - Logistic Regression
  - Decision Tree (shallow, interpretable)
  - Random Forest (same hyperparameters as backend/model.py)

Two evaluations are reported:
  1. The same 80/20 stratified hold-out split used by backend/model.py
     (random_state=42), so numbers are directly comparable to /metrics.
  2. Repeated stratified 5-fold cross-validation over all 1,000 users
     (3 repeats = 15 folds). A 200-user test set is small, so differences of
     a point or two on a single split are within noise; the CV mean +/- std
     shows whether differences between models are real.

Scaling is done inside a Pipeline so each CV fold fits its scaler on training
data only (no leakage).

NOTE: the churn labels in this project are synthetic and were generated from a
logistic function of the features (see src/generate_data.py). Logistic
Regression therefore matches the data-generating process, and is expected to
be competitive with, or better than, tree ensembles. That is a property of the
synthetic data, not evidence about real customers.

Output: prints a table and writes outputs/model_comparison.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "zomato_data.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = ["orders", "avg_rating", "last_order_days"]


def build_models():
    return {
        "Majority baseline": make_pipeline(StandardScaler(), DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision Tree (depth 4)": make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=4, random_state=42)),
        "Random Forest": make_pipeline(
            StandardScaler(), RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH).dropna()
    X, y = df[FEATURES], df["churn"].astype(int)

    # --- 1. Same hold-out split as backend/model.py ----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 2. Repeated stratified CV over all users ------------------------
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    rows = []
    fold_auc = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        cvres = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "roc_auc"])
        fold_auc[name] = cvres["test_roc_auc"]

        rows.append({
            "Model": name,
            "Test accuracy": round(accuracy_score(y_test, pred), 3),
            "Precision": round(precision_score(y_test, pred, zero_division=0), 3),
            "Recall": round(recall_score(y_test, pred, zero_division=0), 3),
            "F1": round(f1_score(y_test, pred, zero_division=0), 3),
            "Test ROC-AUC": round(roc_auc_score(y_test, prob), 3),
            "CV accuracy (mean)": round(cvres["test_accuracy"].mean(), 3),
            "CV accuracy (std)": round(cvres["test_accuracy"].std(), 3),
            "CV ROC-AUC (mean)": round(cvres["test_roc_auc"].mean(), 3),
            "CV ROC-AUC (std)": round(cvres["test_roc_auc"].std(), 3),
        })

    results = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "model_comparison.csv")
    results.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"Hold-out test set: {len(y_test)} users | CV: 5-fold x 3 repeats over {len(y)} users\n")
    print(results.to_string(index=False))
    print(f"\nSaved to {out_path}")

    # Paired comparison: both models are scored on exactly the same 15 folds
    lr_f, rf_f = fold_auc["Logistic Regression"], fold_auc["Random Forest"]
    paired = lr_f - rf_f
    print("\nPaired comparison of ROC-AUC on the same 15 CV folds (Logistic Regression minus Random Forest):")
    print(f"  mean difference {paired.mean():+.3f}, std {paired.std():.3f}, "
          f"Logistic Regression higher in {(paired > 0).sum()} of {len(paired)} folds")
    print("  Caveat: repeated-CV folds overlap, so they are not independent; treat this as descriptive,")
    print("  not as a formal significance test.")

if __name__ == "__main__":
    main()
