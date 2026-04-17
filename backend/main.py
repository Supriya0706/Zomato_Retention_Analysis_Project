from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from model import predict, get_metrics, get_dataset_stats, get_full_dataset, get_model_and_scaler
import os

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Zomato Retention Analysis API",
    description="Churn prediction API for Zomato users. Built with FastAPI + Random Forest.",
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

# ─── Warm up model on startup ─────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("[startup] Warming up model...")
    get_model_and_scaler()
    print("[startup] Model ready ✓")


# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    orders: float = Field(..., ge=0, le=100, example=5, description="Number of orders placed")
    avg_rating: float = Field(..., ge=1.0, le=5.0, example=3.8, description="Average rating given (1–5)")
    last_order_days: float = Field(..., ge=0, le=365, example=45, description="Days since last order")


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    """Health check — used by Render for uptime monitoring."""
    return {"status": "ok", "service": "Zomato Retention Analysis API"}


@app.post("/predict", tags=["Prediction"])
def predict_churn(req: PredictRequest):
    """
    Predict churn probability for a single user.
    Returns probability (%), prediction label, and risk level (Low / Medium / High).
    """
    try:
        result = predict(req.orders, req.avg_rating, req.last_order_days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["Analytics"])
def dataset_stats():
    """Returns aggregate statistics: total users, churn rate, averages."""
    return get_dataset_stats()


@app.get("/model-metrics", tags=["Model"])
def model_metrics():
    """Returns ML model performance: accuracy, F1, ROC-AUC, confusion matrix, feature importance."""
    return get_metrics()


@app.get("/data", tags=["Analytics"])
def full_data():
    """
    Returns the full enriched dataset as JSON.
    Includes computed fields: churn_label, order_segment, rating_segment, risk_score.
    Used by the frontend charts and Power BI.
    """
    return get_full_dataset()


@app.get("/", tags=["System"])
def root():
    return {
        "message": "Zomato Retention Analysis API",
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/stats", "/model-metrics", "/data"],
    }
