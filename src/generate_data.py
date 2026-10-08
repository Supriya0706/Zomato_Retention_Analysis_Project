"""
Synthetic data generator for the Zomato Retention Analysis project.

Fix applied (Phase 1 of the rebuild):
- The original version assigned `churn` via np.random.choice(), completely
  independent of any user behavior feature. That meant any model trained on
  it (e.g. the Random Forest in backend/model.py) was learning from noise,
  and its "accuracy" numbers were meaningless.
- This version computes a churn PROBABILITY from real behavioral signal
  (orders, avg_rating, last_order_days) using a logistic function, then
  samples the actual churn label from that probability. This keeps it
  realistic (not deterministic / not a lookup table) while guaranteeing a
  genuine, learnable relationship between features and the label.

Also added:
- A monthly aggregate order-volume time series (data/monthly_orders.csv)
  with a trend + seasonal pattern, needed for the SARIMA forecasting phase
  later in the rebuild. Pure np.random noise has no seasonality for SARIMA
  to find, so trend + seasonality are deliberately baked in here and must
  not be "discovered" by SARIMA out of nothing.

Phase 3 addition:
- A per-user `join_date` and a monthly Fact_Activity table (user_id, month,
  is_active), needed for real cohort retention analysis in SQL. Each user's
  monthly activity decays over time since joining, with the decay rate tied
  to that user's own churn_probability_true — so users who are "supposed"
  to churn genuinely show declining activity in the cohort data, instead of
  activity being generated independently of churn risk.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

N_USERS = 1000
N_MONTHS = 24  # 2 years of monthly history, for SARIMA later

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_path, "data")
os.makedirs(data_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Per-user snapshot data (used for classification, segmentation, RFM)
# ---------------------------------------------------------------------------

user_id = np.arange(1, N_USERS + 1)
orders = np.random.poisson(5, N_USERS)                      # avg 5 orders
avg_rating = np.random.uniform(1, 5, N_USERS)
last_order_days = np.random.exponential(30, N_USERS)        # days since last order

# --- Real, learnable churn signal -----------------------------------------
# Standardize each driver so coefficients are comparable, then combine into
# a logistic function. Direction of each effect is intentional and matches
# real-world retention behavior:
#   - fewer orders           -> higher churn risk
#   - lower avg_rating       -> higher churn risk
#   - longer since last order -> higher churn risk
orders_z = (orders - orders.mean()) / orders.std()
rating_z = (avg_rating - avg_rating.mean()) / avg_rating.std()
recency_z = (last_order_days - last_order_days.mean()) / last_order_days.std()

logit = (
    -1.1 * orders_z          # more orders -> lower logit -> lower churn prob
    - 0.6 * rating_z         # higher rating -> lower churn prob
    + 1.3 * recency_z        # longer since last order -> higher churn prob
    - 0.75                   # intercept, tunes overall churn base rate
)
churn_prob = 1 / (1 + np.exp(-logit))
churn = np.random.binomial(1, churn_prob)

# Monetary value for RFM: realistically correlated with order count
# (avg order value ~ Rs. 250-450, with noise), not independent of behavior.
avg_order_value = np.random.uniform(250, 450, N_USERS)
total_spend = np.round(orders * avg_order_value * np.random.uniform(0.85, 1.15, N_USERS), 2)

users_df = pd.DataFrame({
    "user_id": user_id,
    "orders": orders,
    "avg_rating": avg_rating,
    "last_order_days": last_order_days,
    "total_spend": total_spend,
    "churn_probability_true": churn_prob,  # kept for validation/debugging, not a feature
    "churn": churn,
})

users_csv_path = os.path.join(data_dir, "zomato_data.csv")
users_df.to_csv(users_csv_path, index=False)
print(f"Synthetic user data generated at {users_csv_path}")
print(f"  Churn rate: {churn.mean():.1%}  (should sit roughly 20-40%, not fixed)")
corr = np.corrcoef(orders, churn)[0, 1]
print(f"  Sanity check - correlation(orders, churn): {corr:.3f} (should be clearly negative)")


# ---------------------------------------------------------------------------
# 2. Monthly aggregate order-volume series (for SARIMA forecasting later)
# ---------------------------------------------------------------------------

months = pd.date_range(end=pd.Timestamp.today().replace(day=1), periods=N_MONTHS, freq="MS")

t = np.arange(N_MONTHS)
trend = 4000 + 60 * t                                   # gradual platform growth
seasonality = 500 * np.sin(2 * np.pi * (t % 12) / 12)    # yearly seasonal cycle
noise = np.random.normal(0, 150, N_MONTHS)

monthly_orders = np.maximum(trend + seasonality + noise, 0).round().astype(int)

monthly_df = pd.DataFrame({
    "month": months,
    "total_orders": monthly_orders,
})

monthly_csv_path = os.path.join(data_dir, "monthly_orders.csv")
monthly_df.to_csv(monthly_csv_path, index=False)
print(f"Monthly order-volume series generated at {monthly_csv_path}")


# ---------------------------------------------------------------------------
# 3. Per-user join_date + monthly Fact_Activity (for SQL cohort retention)
# ---------------------------------------------------------------------------

N_COHORT_MONTHS = 12  # users join at some point in the last 12 months
today = pd.Timestamp.today().replace(day=1)

# Assign each user a join month, weighted slightly toward earlier months so
# later cohorts have fewer "months since join" to observe (realistic).
join_month_offset = np.random.choice(
    np.arange(N_COHORT_MONTHS), size=N_USERS,
    p=np.linspace(1.3, 0.7, N_COHORT_MONTHS) / np.linspace(1.3, 0.7, N_COHORT_MONTHS).sum()
)
join_dates = [today - pd.DateOffset(months=int(off)) for off in join_month_offset]
users_df["join_date"] = join_dates

region_choices = ["North", "South", "East", "West", "Central"]
channel_choices = ["Organic", "Paid Social", "Referral", "App Store", "Email"]
users_df["region"] = np.random.choice(region_choices, N_USERS)
users_df["acquisition_channel"] = np.random.choice(channel_choices, N_USERS)

# Re-save users_df now that join_date/region/channel are added
users_df.to_csv(users_csv_path, index=False)

# --- Build Fact_Activity: one row per user per month since they joined ----
# Activity probability starts high right after joining and decays over the
# months since join. The decay speed is driven by the user's own
# churn_probability_true, so "at risk" users genuinely show faster-fading
# activity in the cohort data (not an independent random pattern).
rows = []
for uid, join_dt, cp in zip(users_df["user_id"], users_df["join_date"], users_df["churn_probability_true"]):
    months_observed = (today.year - join_dt.year) * 12 + (today.month - join_dt.month) + 1
    # decay rate: higher churn_probability_true -> faster drop-off per month
    decay_rate = 0.05 + 0.35 * cp
    for m in range(months_observed):
        month_date = join_dt + pd.DateOffset(months=m)
        active_prob = max(0.05, (1 - decay_rate) ** m)
        is_active = np.random.binomial(1, active_prob)
        rows.append((uid, month_date, m, is_active))

activity_df = pd.DataFrame(rows, columns=["user_id", "month", "months_since_join", "is_active"])
activity_csv_path = os.path.join(data_dir, "fact_activity.csv")
activity_df.to_csv(activity_csv_path, index=False)
print(f"Monthly Fact_Activity table generated at {activity_csv_path} ({len(activity_df)} rows)")

# Sanity check: retention % should clearly decline as months_since_join increases
sanity = activity_df.groupby("months_since_join")["is_active"].mean()
print(f"  Sanity check - retention % at month 0 vs month 6: "
      f"{sanity.get(0, float('nan')):.2f} vs {sanity.get(6, float('nan')):.2f} (should drop)")
