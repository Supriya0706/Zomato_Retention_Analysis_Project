import pandas as pd
import numpy as np
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "zomato_data.csv")

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
