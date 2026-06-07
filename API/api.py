"""
api.py — FastAPI backend for Karachi AQI Predictor
Endpoints:
  GET  /              — health check
  GET  /current       — latest AQI reading + forecasts
  GET  /history?n=24  — last n hourly readings
  POST /predict       — run inference on provided features
Deploy on Render:
  1. Add api.py to repo root
  2. Create new Web Service on render.com
  3. Build command : pip install -r requirements.txt
  4. Start command : uvicorn api:app --host 0.0.0.0 --port $PORT
  5. Add environment variables: MONGODB_URI, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

import hopsworks
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipelines.mongo_store import read_df, read_latest

# ── App setup ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Karachi AQI Predictor API",
    description="Real-time AQI forecasting for Karachi at 1h, 24h, 48h, 72h horizons",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Features list (must match training pipeline) ─────────────────────────
FEATURES = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "dust", "european_aqi", "us_aqi",
    "temp", "humidity", "wind", "wind_gusts",
    "precipitation", "pressure", "cloud_cover",
    "dew_point", "apparent_temp",
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "aqi_lag1", "aqi_lag2", "aqi_lag3",
    "aqi_lag6", "aqi_lag12", "aqi_lag24", "aqi_lag48",
    "aqi_lag72", "aqi_lag96", "aqi_lag120", "aqi_lag168",
    "aqi_roll3_mean", "aqi_roll6_mean",
    "aqi_roll12_mean", "aqi_roll24_mean",
    "aqi_roll48_mean", "aqi_roll72_mean",
    "aqi_roll6_std", "aqi_roll24_std",
    "aqi_change_rate", "aqi_diff1", "aqi_diff6", "aqi_diff24",
    "pm25_lag1", "pm25_lag24", "pm25_roll6_mean",
    "temp_humidity", "pressure_diff",
    "wind_dir_sin", "wind_dir_cos",
    "pm25_wind", "dew_depression",
    "aqi_trend",
]

HORIZONS = ["1h", "24h", "48h", "72h"]

AQI_CATEGORIES = [
    (0,   50,  "Good",                "#00E400"),
    (51,  100, "Moderate",            "#FFFF00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#FF7E00"),
    (151, 200, "Unhealthy",           "#FF0000"),
    (201, 300, "Very Unhealthy",      "#8F3F97"),
    (301, 500, "Hazardous",           "#7E0023"),
]

def aqi_category(val):
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= val <= hi:
            return label, color
    return "Hazardous", "#7E0023"


# ── Model cache ──────────────────────────────────────────────────────────
_models   = {}
_scalers  = {}
_ensemble = {}
_weights  = {}
_project  = None


def load_models():
    global _models, _scalers, _ensemble, _weights, _project
    if _models:
        return

    _project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    mr = _project.get_model_registry()

    for horizon in HORIZONS:
        try:
            versions = mr.get_models(name=f"aqi_model_{horizon}")
            if not versions:
                continue
            m    = sorted(versions, key=lambda x: x.version)[-1]
            mdir = m.download()

            pkl = os.path.join(mdir, f"model_{horizon}.pkl")
            stub = joblib.load(pkl)
            if isinstance(stub, dict) and stub.get("type") == "lstm":
                import tensorflow as tf
                _models[horizon] = tf.keras.models.load_model(
                    os.path.join(mdir, stub["path"]))
            else:
                _models[horizon] = stub

            sc = os.path.join(mdir, f"scaler_{horizon}.pkl")
            if os.path.exists(sc):
                _scalers[horizon] = joblib.load(sc)

            all_p = os.path.join(mdir, f"all_{horizon}.pkl")
            if os.path.exists(all_p):
                _ensemble[horizon] = joblib.load(all_p)

            wts_p = os.path.join(mdir, f"ensemble_weights_{horizon}.pkl")
            if os.path.exists(wts_p):
                _weights[horizon] = joblib.load(wts_p)

        except Exception as e:
            print(f"[WARN] Could not load {horizon} model: {e}")


def _predict_horizon(row_df, horizon):
    """Run ensemble inference for one horizon. Returns predicted AQI (absolute)."""
    available = [f for f in FEATURES if f in row_df.columns]
    X = row_df[available].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    aqi_now = float(row_df["aqi"].iloc[-1])

    weights = _weights.get(horizon, {})
    ensemble = _ensemble.get(horizon, {})
    scaler   = _scalers.get(horizon)

    if not ensemble or not weights:
        # fallback to best single model
        model = _models.get(horizon)
        if model is None:
            return None
        X_s = scaler.transform(X) if scaler else X.values
        diff = float(model.predict(X_s)[-1])
        return round(aqi_now + diff)

    # Weighted ensemble
    X_arr = X.values
    X_s   = scaler.transform(X_arr) if scaler else X_arr
    total_w = sum(weights.values())
    pred_diff = 0.0
    for name, w in weights.items():
        if name not in ensemble:
            continue
        entry = ensemble[name]
        mdl   = entry["model"]
        scaled = entry.get("scaled", True)
        Xin = X_s if scaled else X_arr
        try:
            p = float(mdl.predict(Xin)[-1])
        except Exception:
            continue
        pred_diff += (w / total_w) * p

    return round(aqi_now + pred_diff)


# ── Response schemas ─────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    timestamp:    str
    current_aqi:  int
    category:     str
    color:        str
    forecast: dict


class HistoryResponse(BaseModel):
    rows:  int
    data:  list


class HealthResponse(BaseModel):
    status:    str
    timestamp: str
    mongodb:   str
    models:    list


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
def health():
    """Health check — confirms MongoDB connection and loaded models."""
    try:
        df = read_latest(1)
        mongo_status = f"connected ({len(df)} row read)"
    except Exception as e:
        mongo_status = f"error: {e}"

    load_models()
    return {
        "status":    "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mongodb":   mongo_status,
        "models":    list(_models.keys()),
    }


@app.get("/current", response_model=PredictionResponse)
def current():
    """
    Return the latest AQI reading and 4-horizon forecasts.
    Uses the most recent row from MongoDB as the feature vector.
    """
    load_models()

    df = read_df()
    if df is None or len(df) == 0:
        raise HTTPException(status_code=503, detail="No data in MongoDB — run backfill first")

    df["aqi_trend"] = (
        df["aqi"].rolling(168, min_periods=24).mean()
        .fillna(df["aqi"].expanding().mean())
    )

    latest = df.iloc[[-1]]
    aqi_now = int(latest["aqi"].iloc[0])
    ts      = str(latest["timestamp"].iloc[0])
    cat, color = aqi_category(aqi_now)

    forecast = {}
    for h in HORIZONS:
        pred = _predict_horizon(latest, h)
        if pred is not None:
            pcat, pcol = aqi_category(pred)
            forecast[h] = {"aqi": pred, "category": pcat, "color": pcol}

    return {
        "timestamp":   ts,
        "current_aqi": aqi_now,
        "category":    cat,
        "color":       color,
        "forecast":    forecast,
    }


@app.get("/history")
def history(n: int = 24):
    """
    Return the last n hourly AQI readings from MongoDB.
    Default: last 24 hours.
    """
    if n < 1 or n > 720:
        raise HTTPException(status_code=400, detail="n must be between 1 and 720")

    df = read_latest(n)
    if df is None or len(df) == 0:
        raise HTTPException(status_code=503, detail="No data available")

    records = []
    for _, row in df[["timestamp", "aqi", "pm25", "temp", "humidity"]].iterrows():
        cat, color = aqi_category(float(row["aqi"]))
        records.append({
            "timestamp": str(row["timestamp"]),
            "aqi":       int(row["aqi"]),
            "pm25":      round(float(row["pm25"]), 1),
            "temp":      round(float(row["temp"]), 1),
            "humidity":  round(float(row["humidity"]), 1),
            "category":  cat,
            "color":     color,
        })

    return {"rows": len(records), "data": records}


@app.get("/stats")
def stats():
    """Summary statistics for the full dataset."""
    df = read_df()
    if df is None or len(df) == 0:
        raise HTTPException(status_code=503, detail="No data available")

    aqi = df["aqi"].dropna()
    return {
        "total_rows":   len(df),
        "date_from":    str(df["timestamp"].min().date()),
        "date_to":      str(df["timestamp"].max().date()),
        "aqi_mean":     round(float(aqi.mean()), 1),
        "aqi_median":   round(float(aqi.median()), 1),
        "aqi_min":      int(aqi.min()),
        "aqi_max":      int(aqi.max()),
        "aqi_std":      round(float(aqi.std()), 1),
        "pct_unhealthy": round(float((aqi > 100).mean() * 100), 1),
    }