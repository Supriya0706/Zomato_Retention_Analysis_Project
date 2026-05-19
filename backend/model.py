import pandas as pd
import numpy as np
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
# DATA_PATH can be overridden by env var (useful for Docker/Render)
DATA_PATH = os.getenv("DATA_PATH")

if not DATA_PATH:
    # Robust relative path detection
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "zomato_data.csv")
    
    # Fallback for Docker where 'data' might be at /app/data and model.py at /app/model.py
    if not os.path.exists(DATA_PATH):
        LOCAL_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "zomato_data.csv")
        if os.path.exists(LOCAL_DATA):
            DATA_PATH = LOCAL_DATA

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

# ─── Train & cache model ──────────────────────────────────────────────────────
_model = None
_scaler = None
_metrics = None
_feature_names = ["orders", "avg_rating", "last_order_days"]


def _load_and_train():
    global _model, _scaler, _metrics
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        roc_auc_score, precision_score, recall_score, f1_score
    )
    from sklearn.preprocessing import StandardScaler
    import joblib

    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    df["churn"] = df["churn"].astype(int)

    X = df[_feature_names]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Random Forest — better than plain Logistic Regression
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_train_s, y_train)

    y_pred = rf.predict(X_test_s)
    y_prob = rf.predict_proba(X_test_s)[:, 1]

    cv_scores = cross_val_score(rf, X_train_s, y_train, cv=5, scoring="accuracy")

    cm = confusion_matrix(y_test, y_pred).tolist()
    feature_importance = dict(
        zip(_feature_names, rf.feature_importances_.tolist())
    )

    report = classification_report(y_test, y_pred, output_dict=True)

    _model = rf
    _scaler = scaler
    _metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "confusion_matrix": cm,
        "feature_importance": feature_importance,
        "classification_report": report,
    }

    # Persist for cold-start speed
    joblib.dump({"model": rf, "scaler": scaler, "metrics": _metrics}, MODEL_PATH)
    print(f"[model.py] Trained & saved. Accuracy={_metrics['accuracy']}")


def get_model_and_scaler():
    global _model, _scaler, _metrics
    import joblib
    if _model is None:
        if os.path.exists(MODEL_PATH):
            saved = joblib.load(MODEL_PATH)
            _model, _scaler, _metrics = saved["model"], saved["scaler"], saved["metrics"]
        else:
            _load_and_train()
    return _model, _scaler


def get_metrics():
    if _metrics is None:
        get_model_and_scaler()
    return _metrics


def predict(orders: float, avg_rating: float, last_order_days: float):
    model, scaler = get_model_and_scaler()
    X = np.array([[orders, avg_rating, last_order_days]])
    X_s = scaler.transform(X)
    prob = float(model.predict_proba(X_s)[0][1])
    pred = int(model.predict(X_s)[0])

    if prob < 0.30:
        risk = "Low"
    elif prob < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "churn_probability": round(prob * 100, 2),
        "prediction": pred,
        "prediction_label": "Churned" if pred == 1 else "Retained",
        "risk_level": risk,
    }

def get_dataset_stats():
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    churn_dist = df["churn"].value_counts().to_dict()
    return {
        "total_users": len(df),
        "churned": int(churn_dist.get(1, 0)),
        "retained": int(churn_dist.get(0, 0)),
        "churn_rate": round(churn_dist.get(1, 0) / len(df) * 100, 2),
        "avg_orders": round(df["orders"].mean(), 2),
        "avg_rating": round(df["avg_rating"].mean(), 2),
        "avg_days_since_order": round(df["last_order_days"].mean(), 2),
    }

def get_full_dataset():
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    df["churn"] = df["churn"].astype(int)

    # Computed columns
    df["churn_label"] = df["churn"].map({0: "Retained", 1: "Churned"})
    df["order_segment"] = pd.cut(
        df["orders"], bins=[0, 3, 7, 100],
        labels=["Low", "Medium", "High"], right=True
    ).astype(str)
    df["rating_segment"] = pd.cut(
        df["avg_rating"], bins=[0, 2, 3, 4, 5],
        labels=["Poor", "Average", "Good", "Excellent"], right=True
    ).astype(str)

    return df.to_dict(orient="records")
