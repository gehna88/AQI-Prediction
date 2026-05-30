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

def iterative_forecast(model, latest_row,
                       available, hours=72):
    """
    Predict hour by hour iteratively.
    Each prediction feeds back as input for next hour.
    Returns list of hourly AQI predictions.
    """
    row   = latest_row[available].copy().to_dict()
    preds = []

    for h in range(hours):
        X        = pd.DataFrame([row])[available]
        pred_aqi = float(model.predict(X.values)[0])
        pred_aqi = max(15, min(500, pred_aqi))
        preds.append(pred_aqi)

        # update lag features
        row["aqi_lag3"]  = row["aqi_lag2"]
        row["aqi_lag2"]  = row["aqi_lag1"]
        row["aqi_lag1"]  = row["aqi"]
        row["aqi"]       = pred_aqi
        row["pm25_lag1"] = row.get("pm25", pred_aqi/2)

        # update rolling means
        row["aqi_roll3_mean"]  = (
            row["aqi_roll3_mean"] * 2 + pred_aqi) / 3
        row["aqi_roll6_mean"]  = (
            row["aqi_roll6_mean"] * 5 + pred_aqi) / 6
        row["aqi_roll12_mean"] = (
            row["aqi_roll12_mean"] * 11 + pred_aqi) / 12
        row["aqi_roll24_mean"] = (
            row["aqi_roll24_mean"] * 23 + pred_aqi) / 24

        # update change features
        prev = row["aqi_lag1"]
        row["aqi_change_rate"] = (
            (pred_aqi - prev) / prev
            if prev != 0 else 0)
        row["aqi_diff1"] = pred_aqi - prev

        # update time features
        new_hour = (int(row["hour"]) + 1) % 24
        row["hour"]      = new_hour
        row["hour_sin"]  = np.sin(
            2 * np.pi * new_hour / 24)
        row["hour_cos"]  = np.cos(
            2 * np.pi * new_hour / 24)
        row["is_weekend"] = int(
            int(row["day_of_week"]) in [5, 6])

    return preds

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

    mr     = project.get_model_registry()
    models = {}

    # load 1h model for iterative forecasting
    try:
        m    = mr.get_model("aqi_model_1h")
        mdir = m.download()
        models["1h"] = joblib.load(
            os.path.join(mdir, "model_1h.pkl"))
        print("Loaded 1h model")
    except Exception as e:
        print(f"1h model not found: {e}")

    # load direct models as fallback
    for horizon in ["24h", "48h", "72h"]:
        try:
            m    = mr.get_model(f"aqi_model_{horizon}")
            mdir = m.download()
            models[horizon] = joblib.load(
                os.path.join(mdir,
                             f"model_{horizon}.pkl"))
        except Exception as e:
            print(f"{horizon} model not found: {e}")

    return df, models

# ── LOAD ─────────────────────────────────────────────
df, models = load_everything()
latest      = df.iloc[-1].copy()
current_aqi = float(latest["aqi"])
bg, label   = aqi_color(current_aqi)
tc          = text_color(current_aqi)

# get available features that exist in data
available = [f for f in FEATURES if f in df.columns]

# ── TITLE ────────────────────────────────────────────
st.title("🌫️ Karachi Air Quality Forecast")
st.caption(f"Last updated: {latest['timestamp']}")

# ── CURRENT AQI BANNER ───────────────────────────────
st.markdown(f"""
<div style='background:{bg};padding:24px;
border-radius:12px;text-align:center;
margin-bottom:20px'>
<h1 style='color:{tc};margin:0;font-size:52px'>
AQI {current_aqi:.0f}</h1>
<h3 style='color:{tc};margin:6px 0'>{label}</h3>
<p style='color:{tc};margin:0;opacity:0.85'>
Karachi · Current reading</p>
</div>
""", unsafe_allow_html=True)

# ── 3-DAY FORECAST ───────────────────────────────────
st.subheader("3-Day AQI Forecast")

# use 1h model with iterative forecasting if available
# otherwise fall back to direct 24h/48h/72h models
if "1h" in models:
    hourly_preds = iterative_forecast(
        models["1h"], latest, available, hours=72)
    day1_hours = hourly_preds[0:24]
    day2_hours = hourly_preds[24:48]
    day3_hours = hourly_preds[48:72]

    pred_24h = float(np.mean(day1_hours))
    pred_48h = float(np.mean(day2_hours))
    pred_72h = float(np.mean(day3_hours))
    method   = "Iterative hourly forecasting (24 steps averaged)"

elif "24h" in models and "48h" in models \
        and "72h" in models:
    row      = pd.DataFrame([latest[available]])
    pred_24h = float(models["24h"].predict(row.values)[0])
    pred_48h = float(models["48h"].predict(row.values)[0])
    pred_72h = float(models["72h"].predict(row.values)[0])
    pred_24h = max(15, min(500, pred_24h))
    pred_48h = max(15, min(500, pred_48h))
    pred_72h = max(15, min(500, pred_72h))
    method   = "Direct multi-horizon models"
    hourly_preds = None

else:
    st.error("No models found in registry. "
             "Run training pipeline first.")
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
        border-radius:10px;text-align:center;
        margin-bottom:8px'>
        <h2 style='color:{tc2};margin:0'>
        {pred:.0f}</h2>
        <p style='color:{tc2};margin:4px 0'>
        <b>{day}</b></p>
        <p style='color:{tc2};margin:0;
        font-size:13px'>{lbl2}</p>
        </div>""", unsafe_allow_html=True)

st.caption(f"Forecast method: {method}")

# ── ALERT ────────────────────────────────────────────
max_pred = max(pred_24h, pred_48h, pred_72h)
if max_pred >= 200:
    st.error(
        "🚨 Very Unhealthy air predicted! "
        "Avoid all outdoor activity.")
elif max_pred >= 150:
    st.warning(
        "⚠️ Unhealthy air predicted for sensitive "
        "groups. Limit outdoor exposure.")
elif max_pred >= 100:
    st.warning(
        "⚠️ Moderate to Unhealthy air predicted. "
        "Sensitive individuals take precautions.")
else:
    st.success(
        "✅ Air quality looks acceptable "
        "for the next 3 days.")

st.divider()

# ── HOURLY FORECAST CHART ────────────────────────────
if hourly_preds is not None:
    st.subheader("72-Hour Hourly Forecast")
    from datetime import datetime, timedelta
    try:
        last_ts   = pd.to_datetime(latest["timestamp"])
        fc_times  = [
            str(last_ts + timedelta(hours=i+1))
            for i in range(72)
        ]
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=fc_times,
            y=hourly_preds,
            mode="lines",
            fill="tozeroy",
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
                y=y_val,
                line_dash="dot",
                line_color=color,
                annotation_text=lbl_text,
                annotation_position="right"
            )
        fig_fc.update_layout(
            xaxis_title="Time",
            yaxis_title="Predicted AQI",
            height=350,
            margin=dict(l=0, r=60, t=10, b=0),
            hovermode="x unified"
        )
        st.plotly_chart(fig_fc,
                        use_container_width=True)
    except Exception as e:
        st.info(f"Forecast chart error: {e}")
    st.divider()

# ── HISTORICAL CHART ─────────────────────────────────
st.subheader("AQI History — Last 7 Days")
last7 = df.tail(7 * 24).copy()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=last7["timestamp"],
    y=last7["aqi"],
    mode="lines",
    fill="tozeroy",
    name="Observed AQI",
    line=dict(color="#ff7e00", width=2),
    fillcolor="rgba(255,126,0,0.1)"
))

for y_val, color, lbl_text in [
    (50,  "#00e400", "Good"),
    (100, "#ffa500", "Moderate"),
    (150, "#ff7e00", "Sensitive"),
    (200, "#ff0000", "Unhealthy"),
]:
    fig.add_hline(
        y=y_val,
        line_dash="dot",
        line_color=color,
        annotation_text=lbl_text,
        annotation_position="right"
    )

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="AQI",
    height=400,
    margin=dict(l=0, r=60, t=10, b=0),
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)
st.divider()

# ── POLLUTANTS ───────────────────────────────────────
st.subheader("Current Pollutant Levels")

pollutants = {
    "PM2.5 (μg/m³)": float(latest.get("pm25", 0)),
    "PM10 (μg/m³)":  float(latest.get("pm10", 0)),
    "O3 (μg/m³)":    float(latest.get("o3",   0)),
    "NO2 (μg/m³)":   float(latest.get("no2",  0)),
    "SO2 (μg/m³)":   float(latest.get("so2",  0)),
    "CO (μg/m³)":    float(latest.get("co",   0)),
}

p_cols = st.columns(6)
for col, (name, val) in zip(p_cols, pollutants.items()):
    with col:
        display = f"{val:.1f}" if val > 0 else "N/A"
        st.metric(name, display)

extra_cols = st.columns(4)
extras = {
    "Dust (μg/m³)":    float(latest.get("dust",    0)),
    "Ammonia (μg/m³)": float(latest.get("ammonia", 0)),
    "EU AQI":          float(latest.get("european_aqi", 0)),
    "US AQI":          float(latest.get("us_aqi",  0)),
}
for col, (name, val) in zip(extra_cols, extras.items()):
    with col:
        display = f"{val:.1f}" if val > 0 else "N/A"
        st.metric(name, display)

st.divider()

# ── WEATHER CONDITIONS ───────────────────────────────
st.subheader("Current Weather Conditions")

w_cols = st.columns(5)
weather = {
    "Temperature":  f"{float(latest.get('temp',0)):.1f}°C",
    "Humidity":     f"{float(latest.get('humidity',0)):.0f}%",
    "Wind Speed":   f"{float(latest.get('wind',0)):.1f} m/s",
    "Pressure":     f"{float(latest.get('pressure',0)):.0f} hPa",
    "Cloud Cover":  f"{float(latest.get('cloud_cover',0)):.0f}%",
}
for col, (name, val) in zip(w_cols, weather.items()):
    with col:
        st.metric(name, val)

st.divider()

# ── SHAP FEATURE IMPORTANCE ──────────────────────────
st.subheader("What drives the AQI forecast?")

try:
    import shap
    # use 24h model for SHAP if available
    shap_model = models.get("24h") or \
                 models.get("1h")
    if shap_model:
        row_df    = pd.DataFrame(
            [latest[available]])
        explainer = shap.Explainer(
            shap_model, row_df)
        shap_vals = explainer(row_df)
        importance = pd.DataFrame({
            "Feature":    available,
            "Importance": np.abs(
                shap_vals.values[0])
        }).sort_values(
            "Importance",
            ascending=True
        ).tail(15)

        fig2 = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Oranges"
        )
        fig2.update_layout(
            height=450,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False
        )
        st.plotly_chart(fig2,
                        use_container_width=True)
        st.caption(
            "SHAP values show how much each "
            "feature contributes to the prediction. "
            "Larger bar = more influence.")
except Exception as e:
    st.info(f"Feature importance unavailable: {e}")

st.divider()

# ── AQI SCALE REFERENCE ──────────────────────────────
st.subheader("AQI Scale Reference")
st.dataframe(pd.DataFrame({
    "Category": [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous"],
    "AQI Range": [
        "0–50", "51–100", "101–150",
        "151–200", "201–300", "301–500"],
    "Health Implications": [
        "Air quality is satisfactory",
        "Acceptable; some pollutants may affect sensitive people",
        "Members of sensitive groups may experience effects",
        "Everyone may begin to experience health effects",
        "Health alert: everyone may experience serious effects",
        "Health warnings of emergency conditions"],
    "Indicator": ["🟢", "🟡", "🟠", "🔴", "🟣", "🟤"]
}), hide_index=True, use_container_width=True)

# ── FOOTER ───────────────────────────────────────────
st.divider()
st.caption(
    "Data source: Open-Meteo API · "
    "Models stored in Hopsworks · "
    "Forecasts updated hourly via GitHub Actions · "
    "Built with Streamlit"
)