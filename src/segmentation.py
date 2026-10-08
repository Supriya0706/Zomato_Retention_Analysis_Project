"""
K-Means customer segmentation on RFM features (Recency, Frequency, Monetary).

Unlike the rule-based RFM quartile segments in sql/rfm_scoring.sql (which use
fixed NTILE buckets), this module lets K-Means discover natural groupings in
the data itself — a genuinely different, complementary technique worth
comparing in the write-up (rule-based vs. unsupervised clustering).

The number of clusters (k) is chosen by silhouette score rather than
hardcoded, and each resulting cluster is auto-labeled by ranking its center
on recency/frequency/monetary — not by arbitrary cluster index — so labels
like "Loyal" or "At-Risk" are backed by the actual cluster centers.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "zomato_data.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "user_segments.csv")

RFM_FEATURES = ["last_order_days", "orders", "total_spend"]  # Recency, Frequency, Monetary


def _pick_best_k(X_scaled, k_range=range(3, 6)):
    """Choose k by silhouette score, restricted to a business-interpretable range."""
    best_k, best_score = None, -1
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = round(score, 4)
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score, scores


def _label_clusters(centers_df):
    """
    Rank clusters on each RFM dimension and assign a human-readable label
    based on where each cluster's center actually sits — not an arbitrary
    cluster index.
    """
    df = centers_df.copy()
    # Lower last_order_days = better (more recent); higher orders/spend = better
    df["recency_rank"] = df["last_order_days"].rank(ascending=True)   # 1 = most recent
    df["frequency_rank"] = df["orders"].rank(ascending=False)          # 1 = most orders
    df["monetary_rank"] = df["total_spend"].rank(ascending=False)      # 1 = highest spend
    df["avg_rank"] = df[["recency_rank", "frequency_rank", "monetary_rank"]].mean(axis=1)

    labels = {}
    sorted_clusters = df.sort_values("avg_rank")
    n = len(df)
    for position, (cluster_id, row) in enumerate(sorted_clusters.iterrows()):
        if position == 0:
            labels[cluster_id] = "Loyal"
        elif position == n - 1:
            labels[cluster_id] = "Dormant"
        else:
            labels[cluster_id] = "At-Risk"
    return labels


def run_segmentation():
    df = pd.read_csv(DATA_PATH)
    X = df[RFM_FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, best_score, all_scores = _pick_best_k(X_scaled)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    # Cluster centers in ORIGINAL units (inverse-transform) for interpretable labeling
    centers_original = scaler.inverse_transform(km.cluster_centers_)
    centers_df = pd.DataFrame(centers_original, columns=RFM_FEATURES)
    cluster_label_map = _label_clusters(centers_df)

    df["segment"] = df["cluster"].map(cluster_label_map)

    try:
        df[["user_id", "cluster", "segment"] + RFM_FEATURES].to_csv(OUT_PATH, index=False)
    except OSError as e:  # e.g. read-only filesystem in a container; results are still returned
        print(f"Could not write {OUT_PATH}: {e}")

    summary = df.groupby("segment")[RFM_FEATURES + ["churn"]].mean().round(2)
    summary["user_count"] = df["segment"].value_counts()

    print(f"Best k chosen: {best_k} (silhouette={best_score:.4f}); all k tried: {all_scores}")
    print(f"Segments written to {OUT_PATH}")
    print("\nSegment summary (mean values):")
    print(summary.to_string())

    return df, summary


if __name__ == "__main__":
    run_segmentation()
