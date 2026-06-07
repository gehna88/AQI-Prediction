# Karachi AQI Predictor

End-to-end serverless AQI forecasting pipeline for Karachi — predicts US Air Quality Index up to 72 hours ahead using automated ML training, live Open-Meteo and OpenWeather data, and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green?logo=mongodb)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?logo=streamlit)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=github-actions)

---

## 🚀 Live Deployments

| Resource | URL |
|----------|-----|
| 🌐 Dashboard | https://karachi-aqi-prediction.streamlit.app |
| 🔌 REST API | https://gehna88-karachi-aqi-api.hf.space |
| 📖 API Docs (Swagger) | https://gehna88-karachi-aqi-api.hf.space/docs |
| 💻 GitHub | https://github.com/gehna88/AQI-Prediction |

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

## 🌐 Web Dashboard

**Live:** https://karachi-aqi-prediction.streamlit.app

**Features:**
- Real-time AQI display with EPA health category and color coding
- Health advisory with actionable recommendations per AQI level

---

## 🔌 REST API

FastAPI backend with auto-generated Swagger documentation — deployed on **Hugging Face Spaces**.

**Live API:** https://gehna88-karachi-aqi-api.hf.space  
**Interactive Docs:** https://gehna88-karachi-aqi-api.hf.space/docs

**Run locally:**
```bash
uvicorn api.api:app --reload --port 8000
```

| Endpoint | Description |
|----------|-------------|
| `GET /` | API info and health status |
| `GET /current` | Current AQI + weather snapshot + category |
| `GET /predict` | Full 72h forecast + ensemble inference |
| `GET /history?n=24` | Last N hourly readings from MongoDB |
| `GET /stats` | Dataset summary statistics |

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
             ┌────────────────────────┐
             │   Streamlit Dashboard  │  ← karachi-aqi-prediction.streamlit.app
             │   FastAPI Backend      │  ← gehna88-karachi-aqi-api.hf.space
             └────────────────────────┘
```

**Data flow:**
1. Open-Meteo AQ API + OpenWeatherMap → raw hourly reading → `feature_pipeline.py`
2. Feature engineering (60 features: lags, rolling stats, cyclic time, weather interactions) → MongoDB Atlas
3. Daily: fetch full feature history → train 4 models × 4 horizons → ensemble weights → Hopsworks
4. Dashboard + API: load latest feature row from MongoDB + models from Hopsworks → predict Δ_AQI → reconstruct AQI
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
| REST API | FastAPI + Uvicorn (Hugging Face Spaces) |
| Orchestration | GitHub Actions + cron-job.org |
| Deployment | Streamlit Cloud (Python 3.11.9) + Hugging Face Spaces (Docker) |

---

## Getting Started

### Prerequisites

- Python 3.11
- Free accounts: MongoDB Atlas, Hopsworks, OpenWeatherMap, Hugging Face

### Installation

```bash
git clone https://github.com/gehna88/AQI-Prediction.git
cd AQI-Prediction

python -m venv C:\venvs\aqi-predictor
C:\venvs\aqi-predictor\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

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

# Step 4 — launch API
uvicorn api.api:app --reload --port 8000
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
| `api/api.py` | FastAPI backend — REST endpoints for AQI data and predictions |
| `api/Dockerfile` | Docker config for Hugging Face Spaces deployment |
| `notebooks/EDA_KARACHI.ipynb` | 180-day EDA — distribution, seasonality, ACF/PACF, correlations |
| `notebooks/SHAP_LIME_EXPLANATION.ipynb` | SHAP global importance + cross-horizon heatmap + LIME for LSTM |

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

| Workflow | Schedule | Steps |
|---------|---------|-------|
| `feature_pipeline.yml` | Every hour (cron-job.org) | Fetch AQ + weather → engineer features → upsert to MongoDB |
| `training_pipeline.yml` | Daily 02:00 UTC | Train models → compute ensemble weights → save to Hopsworks |

**Required GitHub Secrets:**

| Secret | Description |
|--------|-------------|
| `HOPSWORKS_API_KEY` | Hopsworks API key |
| `HOPSWORKS_PROJECT` | Hopsworks project name |
| `OPENWEATHER_API_KEY` | OpenWeatherMap API key |
| `MONGODB_URI` | MongoDB Atlas connection string |

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