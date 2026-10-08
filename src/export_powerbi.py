"""
Exports a proper Power BI star schema from the project's real, generated
data — fixing two issues found during the Phase-by-phase code review:

1. The original version RE-SIMULATED region/acquisition_channel/join_date
   independently inside this script (with its own np.random calls), instead
   of using the real ones already produced by src/generate_data.py. That
   meant Dim_Users here could silently drift out of sync with the rest of
   the project. Fixed: this script now reads join_date/region/channel
   directly from data/zomato_data.csv (the single source of truth).

2. Fact_Predictions.csv previously existed in the repo with NO script
   generating it — a stale, non-reproducible file disconnected from the
   actual trained model. Fixed: this script now imports backend/model.py
   and calls predict() on every real user to generate this file from the
   live model, so it's reproducible and always matches the current model.

Also adds a proper Dim_Date table (one row per month present in
Fact_Activity), completing the star schema for the SARIMA/cohort data.

Output: data/powerbi_model/{Dim_Users, Dim_Date, Fact_Activity, Fact_Predictions}.csv
"""

import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "backend"))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(DATA_DIR, "powerbi_model")
os.makedirs(OUT_DIR, exist_ok=True)


def export():
    users = pd.read_csv(os.path.join(DATA_DIR, "zomato_data.csv"), parse_dates=["join_date"])
    activity = pd.read_csv(os.path.join(DATA_DIR, "fact_activity.csv"), parse_dates=["month"])

    # --- Dim_Users: real join_date/region/acquisition_channel, not re-simulated ---
    dim_users = users[["user_id", "region", "acquisition_channel", "join_date"]].copy()
    dim_users["join_date"] = dim_users["join_date"].dt.strftime("%Y-%m-%d")
    dim_users.to_csv(os.path.join(OUT_DIR, "Dim_Users.csv"), index=False)

    # --- Dim_Date: one row per month actually present in Fact_Activity -----
    dim_date = pd.DataFrame({"month": sorted(activity["month"].dt.strftime("%Y-%m-%d").unique())})
    dim_date["month"] = pd.to_datetime(dim_date["month"])
    dim_date["year"] = dim_date["month"].dt.year
    dim_date["month_number"] = dim_date["month"].dt.month
    dim_date["month_name"] = dim_date["month"].dt.strftime("%B")
    dim_date["quarter"] = dim_date["month"].dt.quarter
    dim_date["month"] = dim_date["month"].dt.strftime("%Y-%m-%d")
    dim_date.to_csv(os.path.join(OUT_DIR, "Dim_Date.csv"), index=False)

    # --- Fact_Activity: the REAL monthly activity table from Phase 3 -------
    fact_activity = activity.copy()
    fact_activity["month"] = fact_activity["month"].dt.strftime("%Y-%m-%d")
    fact_activity.to_csv(os.path.join(OUT_DIR, "Fact_Activity.csv"), index=False)

    # --- Fact_Predictions: generated from the LIVE trained model -----------
    from model import predict as model_predict  # imported here so the file
    # is only loaded/trained when export() actually runs, not on module import

    predictions = []
    for _, row in users.iterrows():
        result = model_predict(
            orders=row["orders"], avg_rating=row["avg_rating"], last_order_days=row["last_order_days"]
        )
        predictions.append({
            "user_id": row["user_id"],
            "churn_probability_score": result["churn_probability"],
            "risk_segment": result["risk_level"] + " Risk",
            "model_run_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        })
    fact_predictions = pd.DataFrame(predictions)
    fact_predictions.to_csv(os.path.join(OUT_DIR, "Fact_Predictions.csv"), index=False)

    print(f"Power BI star schema exported to {OUT_DIR}")
    print("  - Dim_Users.csv   :", len(dim_users), "rows")
    print("  - Dim_Date.csv    :", len(dim_date), "rows")
    print("  - Fact_Activity.csv:", len(fact_activity), "rows")
    print("  - Fact_Predictions.csv:", len(fact_predictions), "rows (from the LIVE trained model)")

    # Sanity check: predictions should correlate with actual churn behavior
    merged = fact_predictions.merge(users[["user_id", "churn"]], on="user_id")
    corr = merged["churn_probability_score"].corr(merged["churn"])
    print(f"  Sanity check - correlation(predicted_prob, actual_churn): {corr:.3f} (should be clearly positive)")


if __name__ == "__main__":
    export()
