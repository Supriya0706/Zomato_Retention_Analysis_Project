from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from model import get_dataset_stats, get_full_dataset

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


@app.get("/", tags=["System"])
def root():
    return {
        "message": "Zomato Retention Analysis API",
        "docs": "/docs",
        "endpoints": ["/health", "/stats", "/data"],
    }
