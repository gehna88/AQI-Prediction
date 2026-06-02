"""
app.py  –  v3 (deseasonalized inference)
=========================================
What changed vs v2:
  1. INFERENCE RESEASONALIZATION: models now predict anomaly from monthly mean.
     At inference, monthly_stats.pkl is loaded and the current month's mean
     is added back to produce the final AQI forecast.
  2. All other fixes from v2 retained (direct models for cards, scaler applied,
     SHAP on 24h model, pd.isna() for N/A display).
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import hopsworks
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌫️",
    layout="wide"
)

FEATURES = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "dust", "ammonia", "european_aqi", "us_aqi",
    "temp", "humidity", "wind", "wind_dir",
    "wind_gusts", "precipitation", "pressure",
    "cloud_cover", "visibility", "dew_point",
    "apparent_temp", "soil_temp", "solar_rad",
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "aqi_lag1", "aqi_lag2", "aqi_lag3",
    "aqi_lag6", "aqi_lag12", "aqi_lag24", "aqi_lag48",
    "aqi_roll3_mean", "aqi_roll6_mean",
    "aqi_roll12_mean", "aqi_roll24_mean",
    "aqi_roll6_std", "aqi_roll24_std",
    "aqi_roll24_max", "aqi_roll24_min",
    "aqi_change_rate", "aqi_diff1",
    "aqi_diff6", "aqi_diff24",
    "pm25_lag1", "pm25_lag24", "pm25_roll6_mean",
    "temp_humidity", "wind_chill", "pressure_diff",
    "wind_dir_sin", "wind_dir_cos",
    "pm25_wind", "dew_depression", "is_daytime",
    # seasonal features added in v3 training
    "monthly_mean_aqi", "monthly_std_aqi",
]


def aqi_color(val):
    if val <= 50:   return "#00e400", "Good"
    if val <= 100:  return "#ffff00", "Moderate"
    if val <= 150:  return "#ff7e00", "Unhealthy for Sensitive Groups"
    if val <= 200:  return "#ff0000", "Unhealthy"
    if val <= 300:  return "#8f3f97", "Very Unhealthy"
    return "#7e0023", "Hazardous"


def text_color(val):
    return "#000000" if val <= 100 else "#ffffff"


def iterative_forecast(model, scaler, latest_row,
                       available, monthly_stats, hours=72):
    """
    Iterative 1h forecast. Predictions are in deseasonalized space
    if the model was trained with deseasonalization, then
    re-seasonalized per hour using the current month mean.
    """
    monthly_mean_map = dict(
        zip(monthly_stats["month_num"],
            monthly_stats["monthly_mean_aqi"])
    ) if monthly_stats is not None else {}

    row  = latest_row[available].copy().to_dict()
    preds_raw = []

    for h in range(hours):
        X_raw    = pd.DataFrame([row])[available]
        X_scaled = scaler.transform(X_raw.values) \
                   if scaler is not None else X_raw.values
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            pred_deseason = float(model.predict(
                X_scaled.reshape(1, 1, X_scaled.shape[1]), verbose=0)[0][0])
        else:
            pred_deseason = float(model.predict(X_scaled)[0])

        # re-seasonalize
        current_month = int(row.get("month", 1))
        monthly_mean  = monthly_mean_map.get(current_month, 0)
        pred_aqi      = pred_deseason + monthly_mean
        pred_aqi      = max(15, min(500, pred_aqi))
        preds_raw.append(pred_aqi)

        # update rolling features with raw AQI
        row["aqi_lag3"]  = row["aqi_lag2"]
        row["aqi_lag2"]  = row["aqi_lag1"]
        row["aqi_lag1"]  = row["aqi"]
        row["aqi"]       = pred_aqi
        row["pm25_lag1"] = row.get("pm25", pred_aqi / 2)

        row["aqi_roll3_mean"]  = (row["aqi_roll3_mean"]  * 2  + pred_aqi) / 3
        row["aqi_roll6_mean"]  = (row["aqi_roll6_mean"]  * 5  + pred_aqi) / 6
        row["aqi_roll12_mean"] = (row["aqi_roll12_mean"] * 11 + pred_aqi) / 12
        row["aqi_roll24_mean"] = (row["aqi_roll24_mean"] * 23 + pred_aqi) / 24

        prev = row["aqi_lag1"]
        row["aqi_change_rate"] = (
            (pred_aqi - prev) / prev if prev != 0 else 0)
        row["aqi_diff1"] = pred_aqi - prev

        new_hour = (int(row["hour"]) + 1) % 24
        row["hour"]       = new_hour
        row["hour_sin"]   = np.sin(2 * np.pi * new_hour / 24)
        row["hour_cos"]   = np.cos(2 * np.pi * new_hour / 24)
        row["is_weekend"] = int(int(row["day_of_week"]) in [5, 6])

    return preds_raw


@st.cache_resource(show_spinner="Loading data and models...")
def load_everything():
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)

    mr      = project.get_model_registry()
    models  = {}
    scalers = {}
    monthly_stats = None

    for horizon in ["1h", "24h", "48h", "72h"]:
        try:
            m    = mr.get_model(f"aqi_model_{horizon}")
            mdir = m.download()

            pkl_path = os.path.join(mdir, f"model_{horizon}.pkl")
            stub = joblib.load(pkl_path)
            if isinstance(stub, dict) and stub.get("type") == "lstm":
                # LSTM saved as .keras — load with Keras
                import tensorflow as tf
                keras_path = os.path.join(mdir, stub["path"])
                models[horizon] = tf.keras.models.load_model(keras_path)
            else:
                models[horizon] = stub

            scaler_path = os.path.join(mdir, f"scaler_{horizon}.pkl")
            scalers[horizon] = (
                joblib.load(scaler_path)
                if os.path.exists(scaler_path) else None)

            # load monthly stats if present (saved with the 24h model)
            if monthly_stats is None:
                stats_path = os.path.join(mdir, "monthly_stats.pkl")
                if os.path.exists(stats_path):
                    monthly_stats = joblib.load(stats_path)
                    print("Loaded monthly_stats for deseasonalization")

            print(f"Loaded {horizon} model")
        except Exception as e:
            print(f"{horizon} model not found: {e}")

    return df, models, scalers, monthly_stats


# ── Load ──────────────────────────────────────────────
df, models, scalers, monthly_stats = load_everything()
latest      = df.iloc[-1].copy()
current_aqi = float(latest["aqi"])
bg, label   = aqi_color(current_aqi)
tc          = text_color(current_aqi)

# add seasonal features to latest row for inference
if monthly_stats is not None:
    monthly_mean_map = dict(
        zip(monthly_stats["month_num"],
            monthly_stats["monthly_mean_aqi"]))
    monthly_std_map  = dict(
        zip(monthly_stats["month_num"],
            monthly_stats["monthly_std_aqi"]))
    cur_month = int(pd.to_datetime(latest["timestamp"]).month)
    latest["monthly_mean_aqi"] = monthly_mean_map.get(cur_month, 80)
    latest["monthly_std_aqi"]  = monthly_std_map.get(cur_month, 20)

available = [f for f in FEATURES if f in df.columns
             or f in latest.index]
# rebuild available to include seasonal features even if not in df cols
available = [f for f in FEATURES
             if f in latest.index and not pd.isna(latest.get(f, np.nan))]

# ── TITLE ─────────────────────────────────────────────
st.title("🌫️ Karachi Air Quality Forecast")
st.caption(f"Last updated: {latest['timestamp']}")

# ── CURRENT AQI BANNER ────────────────────────────────
st.markdown(f"""
<div style='background:{bg};padding:24px;
border-radius:12px;text-align:center;margin-bottom:20px'>
<h1 style='color:{tc};margin:0;font-size:52px'>AQI {current_aqi:.0f}</h1>
<h3 style='color:{tc};margin:6px 0'>{label}</h3>
<p style='color:{tc};margin:0;opacity:0.85'>Karachi · Current reading</p>
</div>
""", unsafe_allow_html=True)

# ── 3-DAY FORECAST ────────────────────────────────────
st.subheader("3-Day AQI Forecast")

has_direct = ("24h" in models and "48h" in models and "72h" in models)
has_1h     = "1h" in models

def predict_horizon(horizon):
    """Predict and re-seasonalize. Handles both sklearn and LSTM models."""
    import tensorflow as tf
    feat_cols = [f for f in FEATURES if f in latest.index]
    row_df    = pd.DataFrame([latest[feat_cols]])
    sc        = scalers.get(horizon)
    X         = sc.transform(row_df.values) if sc else row_df.values
    if isinstance(models[horizon], tf.keras.Model):
        pred_ds = float(models[horizon].predict(
            X.reshape(1, 1, X.shape[1]), verbose=0)[0][0])
    else:
        pred_ds = float(models[horizon].predict(X)[0])

    # re-seasonalize
    if monthly_stats is not None:
        monthly_mean_map_local = dict(
            zip(monthly_stats["month_num"],
                monthly_stats["monthly_mean_aqi"]))
        pred_raw = pred_ds + monthly_mean_map_local.get(cur_month, 0)
    else:
        pred_raw = pred_ds   # model not deseasonalized

    return max(15, min(500, pred_raw))

if has_direct:
    pred_24h = predict_horizon("24h")
    pred_48h = predict_horizon("48h")
    pred_72h = predict_horizon("72h")
    method   = "Direct multi-horizon models (24h/48h/72h) + deseasonalization"
elif has_1h:
    sc_1h        = scalers.get("1h")
    hourly_preds = iterative_forecast(
        models["1h"], sc_1h, latest, available,
        monthly_stats, hours=72)
    pred_24h = float(np.mean(hourly_preds[0:24]))
    pred_48h = float(np.mean(hourly_preds[24:48]))
    pred_72h = float(np.mean(hourly_preds[48:72]))
    method   = "Iterative hourly forecasting (fallback)"
else:
    st.error("No models found. Run training pipeline first.")
    st.stop()

forecasts = [
    ("Tomorrow (Day 1)", pred_24h),
    ("Day 2",            pred_48h),
    ("Day 3",            pred_72h),
]

cols = st.columns(3)
for col, (day, pred) in zip(cols, forecasts):
    bg2, lbl2 = aqi_color(pred)
    tc2       = text_color(pred)
    with col:
        st.markdown(f"""
        <div style='background:{bg2};padding:20px;
        border-radius:10px;text-align:center;margin-bottom:8px'>
        <h2 style='color:{tc2};margin:0'>{pred:.0f}</h2>
        <p style='color:{tc2};margin:4px 0'><b>{day}</b></p>
        <p style='color:{tc2};margin:0;font-size:13px'>{lbl2}</p>
        </div>""", unsafe_allow_html=True)

st.caption(f"Forecast method: {method}")

# ── ALERTS ────────────────────────────────────────────
max_pred = max(pred_24h, pred_48h, pred_72h)
if max_pred >= 200:
    st.error("🚨 Very Unhealthy air predicted! Avoid all outdoor activity.")
elif max_pred >= 150:
    st.warning("⚠️ Unhealthy air predicted for sensitive groups. Limit outdoor exposure.")
elif max_pred >= 100:
    st.warning("⚠️ Moderate to Unhealthy air predicted. Sensitive individuals take precautions.")
else:
    st.success("✅ Air quality looks acceptable for the next 3 days.")

st.divider()

# ── HOURLY FORECAST CHART ─────────────────────────────
if has_1h:
    st.subheader("72-Hour Hourly Forecast")
    try:
        sc_1h        = scalers.get("1h")
        hourly_preds = iterative_forecast(
            models["1h"], sc_1h, latest,
            available, monthly_stats, hours=72)

        last_ts  = pd.to_datetime(latest["timestamp"])
        from datetime import timedelta
        fc_times = [
            str(last_ts + timedelta(hours=i+1))
            for i in range(72)
        ]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=fc_times, y=hourly_preds,
            mode="lines", fill="tozeroy",
            name="Forecast",
            line=dict(color="#0066cc", width=2),
            fillcolor="rgba(0,102,204,0.1)"
        ))
        for y_val, color, lbl_text in [
            (50,  "#00e400", "Good"),
            (100, "#ffa500", "Moderate"),
            (150, "#ff7e00", "Sensitive"),
            (200, "#ff0000", "Unhealthy"),
        ]:
            fig_fc.add_hline(
                y=y_val, line_dash="dot", line_color=color,
                annotation_text=lbl_text,
                annotation_position="right")
        fig_fc.update_layout(
            xaxis_title="Time", yaxis_title="Predicted AQI",
            height=350, margin=dict(l=0, r=60, t=10, b=0),
            hovermode="x unified")
        st.plotly_chart(fig_fc, use_container_width=True)
        st.caption(
            "Hourly chart uses iterative 1h model. "
            "Day 1/2/3 cards use dedicated 24h/48h/72h models.")
    except Exception as e:
        st.info(f"Forecast chart error: {e}")
    st.divider()

# ── HISTORICAL CHART ──────────────────────────────────
st.subheader("AQI History — Last 7 Days")
last7 = df.tail(7 * 24).copy()
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=last7["timestamp"], y=last7["aqi"],
    mode="lines", fill="tozeroy", name="Observed AQI",
    line=dict(color="#ff7e00", width=2),
    fillcolor="rgba(255,126,0,0.1)"
))
for y_val, color, lbl_text in [
    (50,  "#00e400", "Good"), (100, "#ffa500", "Moderate"),
    (150, "#ff7e00", "Sensitive"), (200, "#ff0000", "Unhealthy"),
]:
    fig.add_hline(y=y_val, line_dash="dot", line_color=color,
                  annotation_text=lbl_text,
                  annotation_position="right")
fig.update_layout(
    xaxis_title="Time", yaxis_title="AQI", height=400,
    margin=dict(l=0, r=60, t=10, b=0), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
st.divider()

# ── POLLUTANTS ────────────────────────────────────────
st.subheader("Current Pollutant Levels")

def fmt(val):
    return "N/A" if pd.isna(val) else f"{float(val):.1f}"

pollutants = {
    "PM2.5 (μg/m³)": latest.get("pm25"),
    "PM10 (μg/m³)":  latest.get("pm10"),
    "O3 (μg/m³)":    latest.get("o3"),
    "NO2 (μg/m³)":   latest.get("no2"),
    "SO2 (μg/m³)":   latest.get("so2"),
    "CO (μg/m³)":    latest.get("co"),
}
p_cols = st.columns(6)
for col, (name, val) in zip(p_cols, pollutants.items()):
    with col:
        st.metric(name, fmt(val))

extra_cols = st.columns(4)
extras = {
    "Dust (μg/m³)":    latest.get("dust"),
    "Ammonia (μg/m³)": latest.get("ammonia"),
    "EU AQI":          latest.get("european_aqi"),
    "US AQI":          latest.get("us_aqi"),
}
for col, (name, val) in zip(extra_cols, extras.items()):
    with col:
        st.metric(name, fmt(val))

st.divider()

# ── WEATHER ───────────────────────────────────────────
st.subheader("Current Weather Conditions")
w_cols = st.columns(5)
weather = {
    "Temperature":  f"{float(latest.get('temp', 0)):.1f}°C",
    "Humidity":     f"{float(latest.get('humidity', 0)):.0f}%",
    "Wind Speed":   f"{float(latest.get('wind', 0)):.1f} m/s",
    "Pressure":     f"{float(latest.get('pressure', 0)):.0f} hPa",
    "Cloud Cover":  f"{float(latest.get('cloud_cover', 0)):.0f}%",
}
for col, (name, val) in zip(w_cols, weather.items()):
    with col:
        st.metric(name, val)

st.divider()

# ── SEASONAL CONTEXT ──────────────────────────────────
if monthly_stats is not None:
    st.subheader("Seasonal AQI Context")
    fig_season = go.Figure()
    fig_season.add_trace(go.Bar(
        x=monthly_stats["month_num"],
        y=monthly_stats["monthly_mean_aqi"],
        error_y=dict(type="data",
                     array=monthly_stats["monthly_std_aqi"],
                     visible=True),
        name="Monthly Mean AQI",
        marker_color="#ff7e00"
    ))
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig_season.update_layout(
        xaxis=dict(tickmode="array",
                   tickvals=list(range(1, 13)),
                   ticktext=month_names),
        yaxis_title="AQI",
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        title_text="Monthly Average AQI — Karachi"
    )
    st.plotly_chart(fig_season, use_container_width=True)
    cur_mean = monthly_mean_map.get(cur_month, 80)
    st.caption(
        f"Current month ({month_names[cur_month-1]}) "
        f"typical AQI: {cur_mean:.0f}. "
        f"Forecasts are calibrated against this baseline.")
    st.divider()

# ── SHAP ──────────────────────────────────────────────
st.subheader("What drives the 24h AQI forecast?")
try:
    import shap
    shap_model  = models.get("24h")
    shap_scaler = scalers.get("24h")

    if shap_model is not None:
        feat_cols = [f for f in FEATURES if f in latest.index]
        row_df    = pd.DataFrame([latest[feat_cols]])
        X_shap    = (shap_scaler.transform(row_df.values)
                     if shap_scaler else row_df.values)
        X_shap_df = pd.DataFrame(X_shap, columns=feat_cols)

        explainer  = shap.Explainer(shap_model, X_shap_df)
        shap_vals  = explainer(X_shap_df)
        importance = pd.DataFrame({
            "Feature":    feat_cols,
            "Importance": np.abs(shap_vals.values[0])
        }).sort_values("Importance", ascending=True).tail(15)

        fig2 = px.bar(
            importance, x="Importance", y="Feature",
            orientation="h", color="Importance",
            color_continuous_scale="Oranges")
        fig2.update_layout(
            height=450, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "SHAP values for the 24h model. "
            "Larger bar = more influence on tomorrow's AQI prediction.")
    else:
        st.info("24h model not loaded.")
except Exception as e:
    st.info(f"Feature importance unavailable: {e}")

st.divider()

# ── AQI SCALE ─────────────────────────────────────────
st.subheader("AQI Scale Reference")
st.dataframe(pd.DataFrame({
    "Category": [
        "Good", "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy", "Very Unhealthy", "Hazardous"],
    "AQI Range": [
        "0–50", "51–100", "101–150",
        "151–200", "201–300", "301–500"],
    "Health Implications": [
        "Air quality is satisfactory",
        "Acceptable; some pollutants may affect sensitive people",
        "Sensitive groups may experience health effects",
        "Everyone may begin to experience health effects",
        "Health alert: everyone may experience serious effects",
        "Health warnings of emergency conditions"],
    "Indicator": ["🟢", "🟡", "🟠", "🔴", "🟣", "🟤"]
}), hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "Data: Open-Meteo API · Models: Hopsworks · "
    "Pipeline: GitHub Actions · Built with Streamlit"
)