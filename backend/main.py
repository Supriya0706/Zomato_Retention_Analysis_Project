from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from model import get_dataset_stats, get_full_dataset, get_metrics, predict as model_predict
import sys
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_here, "src"), os.path.join(os.path.dirname(_here), "src")):
    if os.path.isdir(_p):
        sys.path.append(_p)
from segmentation import run_segmentation
from forecasting import _load_series, forecast_future, evaluate_against_baselines

_segment_cache = None
_forecast_cache = None

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Zomato Retention Analysis API",
    description="Analytics API for Zomato users.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,https://*.vercel.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production with ALLOWED_ORIGINS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request schema for prediction ────────────────────────────────────────────
class ChurnPredictionRequest(BaseModel):
    orders: float = Field(..., ge=0, description="Total number of orders placed")
    avg_rating: float = Field(..., ge=1, le=5, description="Average rating given by the user (1-5)")
    last_order_days: float = Field(..., ge=0, description="Days since the user's last order")

    class Config:
        json_schema_extra = {
            "example": {"orders": 2, "avg_rating": 2.1, "last_order_days": 45}
        }


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    """Health check — used by Render for uptime monitoring."""
    return {"status": "ok", "service": "Zomato Retention Analysis API"}


@app.get("/stats", tags=["Analytics"])
def dataset_stats():
    """Returns aggregate statistics: total users, churn rate, averages."""
    return get_dataset_stats()


@app.get("/data", tags=["Analytics"])
def full_data():
    """
    Returns the full dataset as JSON.
    Includes computed fields: churn_label, order_segment, rating_segment.
    Used by the frontend charts and Power BI.
    """
    return get_full_dataset()


@app.post("/predict", tags=["Machine Learning"])
def predict_churn(request: ChurnPredictionRequest):
    """
    Predicts churn probability for a single user from live behavioral inputs,
    using the trained Random Forest classifier (backend/model.py).
    """
    try:
        return model_predict(
            orders=request.orders,
            avg_rating=request.avg_rating,
            last_order_days=request.last_order_days,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/metrics", tags=["Machine Learning"])
def model_metrics():
    """
    Returns the trained Random Forest's evaluation metrics: accuracy,
    precision, recall, F1, ROC-AUC, cross-validation scores, confusion
    matrix, and feature importance. Computed once at training time.
    """
    try:
        return get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load metrics: {e}")


@app.get("/segments", tags=["Machine Learning"])
def customer_segments():
    """
    Returns K-Means customer segments (Loyal / At-Risk / Dormant) computed
    from RFM features, plus a per-segment summary (avg orders, spend,
    recency, and churn rate for cross-validation against the classifier).
    Cached after first call since K-Means is re-fit each time it's run.
    """
    global _segment_cache
    try:
        if _segment_cache is None:
            df, summary = run_segmentation()
            _segment_cache = {
                "users": df[["user_id", "segment", "last_order_days", "orders", "total_spend"]].to_dict(orient="records"),
                "summary": summary.reset_index().to_dict(orient="records"),
            }
        return _segment_cache
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")


@app.get("/forecast", tags=["Machine Learning"])
def order_forecast():
    """
    Returns a 6-month SARIMA forecast of monthly order volume, plus the
    backtest comparison against persistence and seasonal-naive baselines
    (RMSE/MAE for all three), so the forecast's accuracy claim is
    verifiable rather than asserted.
    """
    global _forecast_cache
    try:
        if _forecast_cache is None:
            series = _load_series()
            backtest_results, test, sarima_pred, persistence_pred, seasonal_naive_pred = evaluate_against_baselines(series)
            mean_fc, ci = forecast_future(series)
            _forecast_cache = {
                "historical": [{"month": d.strftime("%Y-%m"), "total_orders": int(v)} for d, v in series.items()],
                "forecast": [
                    {
                        "month": d.strftime("%Y-%m"),
                        "forecast_orders": round(float(v), 0),
                        "lower_80": round(float(ci.iloc[i, 0]), 0),
                        "upper_80": round(float(ci.iloc[i, 1]), 0),
                    }
                    for i, (d, v) in enumerate(mean_fc.items())
                ],
                "backtest": backtest_results,
            }
        return _forecast_cache
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting failed: {e}")


@app.get("/", tags=["System"])
def root():
    return {
        "message": "Zomato Retention Analysis API",
        "docs": "/docs",
        "endpoints": ["/health", "/stats", "/data", "/predict", "/metrics", "/segments", "/forecast"],
    }
