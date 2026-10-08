"""
Exploratory Data Analysis: correlation heatmap + cohort retention heatmap.

Phase 5 rebuild notes:
- The original version used os.getcwd()-based paths, which only worked if
  the script happened to be run from a specific working directory, and
  depended on cleaned_zomato_data.csv existing from a separate manual step
  (data_cleaning.py). Rewritten here to use __file__-based paths (works
  regardless of where it's invoked from) and to read zomato_data.csv
  directly, dropping NaNs inline instead of depending on an external script.
- plt.show() removed — this environment is headless; figures are saved
  to disk and verified by inspection instead.
- Added: a cohort retention heatmap built from data/fact_activity.csv
  (signup month x months-since-join), the classic cohort analysis
  visualization discussed in the rebuild plan.
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def correlation_heatmap():
    df = pd.read_csv(os.path.join(DATA_DIR, "zomato_data.csv"))
    df = df.dropna()

    numeric_cols = ["orders", "avg_rating", "last_order_days", "total_spend", "churn"]
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Matrix — Churn Drivers")
    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "correlation_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Correlation heatmap saved to {out_path}")

    # Sanity check: churn should correlate negatively with orders, positively with last_order_days
    print(f"  correlation(churn, orders): {corr.loc['churn', 'orders']:.3f} (expect negative)")
    print(f"  correlation(churn, last_order_days): {corr.loc['churn', 'last_order_days']:.3f} (expect positive)")


def cohort_retention_heatmap():
    """
    Builds the classic cohort retention heatmap: rows = signup cohort month,
    columns = months since signup, color = % of that cohort still active.
    Uses the same fact_activity.csv that sql/cohort_retention.sql queries,
    so the chart and the SQL analysis are backed by the same underlying data.
    """
    users = pd.read_csv(os.path.join(DATA_DIR, "zomato_data.csv"), parse_dates=["join_date"])
    activity = pd.read_csv(os.path.join(DATA_DIR, "fact_activity.csv"))

    users["cohort_month"] = users["join_date"].dt.strftime("%Y-%m")
    merged = activity.merge(users[["user_id", "cohort_month"]], on="user_id")

    cohort_sizes = merged[merged["months_since_join"] == 0].groupby("cohort_month")["user_id"].nunique()

    # unstack WITHOUT fill_value so not-yet-observed cells stay NaN, not 0 —
    # a cohort that signed up last month genuinely has no "month 6" data yet,
    # which is different from "0% retained" and must not be drawn the same way.
    observed_counts = merged.groupby(["cohort_month", "months_since_join"])["user_id"].nunique().unstack()
    active_counts = merged.groupby(["cohort_month", "months_since_join"])["is_active"].sum().unstack()
    retention_pct = (active_counts / cohort_sizes.values.reshape(-1, 1)) * 100
    retention_pct = retention_pct.where(observed_counts.notna())  # mask unobserved future months

    plt.figure(figsize=(12, 6))
    sns.heatmap(retention_pct, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=100,
                cbar_kws={"label": "% of cohort still active"}, mask=retention_pct.isna(),
                linewidths=0.5, linecolor="white")
    plt.title("Cohort Retention Heatmap (% active by months since signup)")
    plt.xlabel("Months since signup")
    plt.ylabel("Signup cohort month")
    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "cohort_retention_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Cohort retention heatmap saved to {out_path}")

    # Sanity check: retention should decline left to right within each row
    month0 = retention_pct.get(0)
    month3 = retention_pct.get(3)
    if month0 is not None and month3 is not None:
        print(f"  Avg retention at month 0: {month0.mean():.1f}%  |  at month 3: {month3.mean():.1f}% (should drop)")


if __name__ == "__main__":
    correlation_heatmap()
    cohort_retention_heatmap()
