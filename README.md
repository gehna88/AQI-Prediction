# Karachi AQI Predictor

End-to-end serverless AQI forecasting pipeline for Karachi — predicts US Air Quality Index up to 72 hours ahead using automated ML training, live Open-Meteo and OpenWeather data, and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green?logo=mongodb)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?logo=streamlit)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=github-actions)

---

## About

Karachi is one of South Asia's most polluted megacities, yet real-time AQI forecasting tools built specifically for the city remain scarce. This project addresses that by building a fully automated, serverless ML system that:

- Fetches live weather and pollution data **every hour** from Open-Meteo and OpenWeatherMap
- Engineers **60 time-series features** per row — lags, rolling statistics, cyclic encodings, derived weather interactions
- Trains an **ensemble of Ridge, RandomForest, XGBoost, and LSTM** models daily using differenced targets
- Serves **1h, 24h, 48h, and 72h AQI forecasts** through a public Streamlit dashboard with health alerts
- Stores all features in **MongoDB Atlas** and all model artefacts in **Hopsworks** — nothing committed to the repo
- Explains every prediction using **SHAP** (for tree/linear models) and **LIME** (for the LSTM)

The best model (XGBoost) achieves R²=0.994 and RMSE=1.30 at 1h. The 24h Ridge model achieves R²=0.549 and RMSE=11.76. Full methodology and evaluation are in the notebooks.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Actions                           │
│                                                                 │
│   ⏱ Hourly (cron-job.org)          📅 Daily (02:00 UTC)        │
│   feature_pipeline.py              training_pipeline.py        │
│         │                                   │                  │
└─────────┼───────────────────────────────────┼──────────────────┘
          │                                   │
          ▼                                   ▼
   MongoDB Atlas                       Hopsworks
   (aqi_predictor.features)            (Model Registry)
   ← feature store →                   ← Ridge, RF, XGBoost, LSTM →
          │                                   │
          └───────────────┬───────────────────┘
                          ▼
                 Streamlit Dashboard
                 (inference + alerts + SHAP)
```

**Data flow:**
1. Open-Meteo AQ API + OpenWeatherMap → raw hourly reading → `feature_pipeline.py`
2. Feature engineering (60 features: lags, rolling stats, cyclic time, weather interactions) → MongoDB Atlas
3. Daily: fetch full feature history → train 4 models × 4 horizons → ensemble weights → Hopsworks
4. Dashboard: load latest feature row from MongoDB + models from Hopsworks → predict Δ_AQI → reconstruct AQI
5. SHAP + LIME explanations generated in `notebooks/SHAP_LIME_EXPLANATION.ipynb`

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data sources | Open-Meteo Archive + AQ APIs (free, no key); OpenWeatherMap (free key) |
| Feature store | MongoDB Atlas (M0 free tier, `aqi_predictor.features`) |
| Model registry | Hopsworks (free tier) |
| Models | Ridge Regression, RandomForest, XGBoost, LSTM (TensorFlow/Keras) |
| Explainability | SHAP (LinearExplainer, TreeExplainer) + LIME (LimeTabularExplainer) |
| Dashboard | Streamlit + Plotly |
| Orchestration | GitHub Actions + cron-job.org |
| Deployment | Streamlit Cloud (Python 3.11.9) |

---

## Getting Started

### Prerequisites

- Python 3.11
- Free accounts: MongoDB Atlas, Hopsworks, OpenWeatherMap

### Installation

```bash
git clone https://github.com/gehna88/AQI-Prediction.git
cd AQI-Prediction

# Windows — create venv OUTSIDE OneDrive to avoid file corruption
python -m venv C:\venvs\aqi-predictor
C:\venvs\aqi-predictor\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
pip install tensorflow==2.16.2   # Python 3.12 locally
```

### Configuration

```bash
cp .env.example .env
```

Fill in `.env`:

```env
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT=your_project_name
OPENWEATHER_API_KEY=your_openweather_key
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority
```

### Run Locally

```bash
# Step 1 — one-time historical backfill (180 days)
python pipelines/backfill_pipeline.py

# Step 2 — train models
python pipelines/training_pipeline.py

# Step 3 — launch dashboard
streamlit run app/app.py
```

---

## Key Files

| File | Purpose |
|------|---------|
| `pipelines/backfill_pipeline.py` | One-time: fetch 180 days of historical AQI + weather from Open-Meteo |
| `pipelines/feature_pipeline.py` | Hourly: fetch current reading → engineer 60 features → upsert to MongoDB |
| `pipelines/training_pipeline.py` | Daily: load features → train Ridge/RF/XGBoost/LSTM → save to Hopsworks |
| `pipelines/mongo_store.py` | MongoDB Atlas helper — `store_df()`, `read_df()`, `read_latest()` |
| `app/app.py` | Streamlit dashboard — live AQI, 3-day forecast, health alerts |
| `notebooks/EDA_KARACHI.ipynb` | 173-day EDA — distribution, seasonality, ACF/PACF, correlations |
| `notebooks/SHAP_LIME_EXPLANATION.ipynb` | SHAP global importance + cross-horizon heatmap + LIME for LSTM |

---

## MongoDB Collections

| Collection | Contents |
|-----------|---------|
| `aqi_predictor.features` | One document per hour; 60 engineered features + AQI targets |

## Hopsworks Model Registry

| Model | Contents |
|-------|---------|
| `aqi_model_1h` | Ridge + RF + XGBoost + LSTM artefacts, scaler, ensemble weights |
| `aqi_model_24h` | Same |
| `aqi_model_48h` | Same |
| `aqi_model_72h` | Same |

---

## Models & Results

All models predict **Δ_AQI** (change from current reading). Reconstruction: `predicted_aqi = aqi_now + model.predict(features)`. Ensemble weights are R²-proportional across positive-R² models per horizon.

| Horizon | Best model | RMSE | R² |
|---------|-----------|------|----|
| 1h | XGBoost | 1.30 | **0.994** |
| 24h | Ridge (α=500) | 11.76 | **0.549** |
| 48h | Ridge (α=500) | 18.01 | 0.441 |
| 72h | Ridge (α=500) | 19.09 | 0.201 |


---

## CI/CD

Two GitHub Actions workflows triggered via `workflow_dispatch` from **cron-job.org**:

| Workflow | Schedule | Steps |
|---------|---------|-------|
| `feature_pipeline.yml` | Every hour | Fetch AQ + weather → engineer features → upsert to MongoDB |
| `training_pipeline.yml` | Daily 02:00 UTC | Train models → compute ensemble weights → save to Hopsworks |

**Required GitHub Secrets:**

| Secret | Description |
|--------|-------------|
| `HOPSWORKS_API_KEY` | Hopsworks API key |
| `HOPSWORKS_PROJECT` | Hopsworks project name |
| `OPENWEATHER_API_KEY` | OpenWeatherMap API key |
| `MONGODB_URI` | MongoDB Atlas connection string |

> **MongoDB Atlas network access:** Add `0.0.0.0/0` to allow GitHub Actions runners which use dynamic IPs.

**cron-job.org setup (one-time):**

```
URL:     https://api.github.com/repos/YOUR_USERNAME/YOUR_REPO/actions/workflows/feature_pipeline.yml/dispatches
Method:  POST
Headers: Authorization: Bearer YOUR_GITHUB_PAT
Body:    {"ref":"main"}
Schedule: 0 * * * *
```

---

## AQI Alert Levels

| AQI | Category | Guidance |
|-----|---------|---------|
| 0–50 | 🟢 Good | No action |
| 51–100 | 🟡 Moderate | Sensitive individuals take care |
| 101–150 | 🟠 Unhealthy for Sensitive Groups | Reduce outdoor activity |
| 151–200 | 🔴 Unhealthy | Avoid prolonged outdoor exposure |
| 201–300 | 🟣 Very Unhealthy | Stay indoors |
| 301+ | 🟤 Hazardous | Emergency conditions |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
