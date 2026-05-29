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
    "temp", "humidity", "wind",
    "hour", "day_of_week", "month", "is_weekend",
    "aqi_lag1", "aqi_lag2", "aqi_lag24",
    "aqi_roll6_mean", "aqi_roll24_mean",
    "aqi_roll6_std", "aqi_change_rate"
]

def aqi_color(val):
    if val <= 50:   return "#00e400", "Good"
    if val <= 100:  return "#ffff00", "Moderate"
    if val <= 150:  return "#ff7e00", "Unhealthy for Sensitive Groups"
    if val <= 200:  return "#ff0000", "Unhealthy"
    if val <= 300:  return "#8f3f97", "Very Unhealthy"
    return "#7e0023", "Hazardous"

def aqi_text_color(val):
    if val <= 100:  return "#000000"
    return "#ffffff"

@st.cache_resource(show_spinner="Loading model and data...")
def load_everything():
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)

    mr = project.get_model_registry()
    model_meta = mr.get_model("aqi_model")
    model_dir  = model_meta.download()
    model = joblib.load(
        os.path.join(model_dir, "model.pkl"))

    return df, model, project

# ── HEADER ───────────────────────────────────────────
st.title("🌫️ Karachi Air Quality Forecast")
st.caption("Real-time AQI monitoring and 3-day predictions")

df, model, project = load_everything()

latest      = df.iloc[-1]
current_aqi = float(latest["aqi"])
bg_color, label = aqi_color(current_aqi)
txt_color       = aqi_text_color(current_aqi)

# ── CURRENT AQI BANNER ───────────────────────────────
st.markdown(f"""
<div style='background:{bg_color};padding:24px;
border-radius:12px;text-align:center;
margin-bottom:24px'>
<h1 style='color:{txt_color};margin:0;font-size:48px'>
AQI {current_aqi:.0f}</h1>
<h3 style='color:{txt_color};margin:4px 0'>{label}</h3>
<p style='color:{txt_color};margin:0;opacity:0.8'>
Karachi · Latest reading</p>
</div>
""", unsafe_allow_html=True)

# ── 3-DAY FORECAST ───────────────────────────────────
st.subheader("3-Day AQI Forecast")

row      = pd.DataFrame([latest[FEATURES]])
pred_24h = float(model.predict(row)[0])

# approximate 48h and 72h by shifting features
row_48          = row.copy()
row_48["aqi"]   = pred_24h
row_48["aqi_lag1"]  = pred_24h
row_48["hour"]  = (int(latest["hour"]) + 24) % 24
pred_48h = float(model.predict(row_48)[0])

row_72          = row_48.copy()
row_72["aqi"]   = pred_48h
row_72["aqi_lag1"]  = pred_48h
pred_72h = float(model.predict(row_72)[0])

forecasts = [
    ("Tomorrow",   pred_24h),
    ("Day 2",      pred_48h),
    ("Day 3",      pred_72h),
]

cols = st.columns(3)
for col, (day, pred) in zip(cols, forecasts):
    bg, lbl = aqi_color(pred)
    tc       = aqi_text_color(pred)
    with col:
        st.markdown(f"""
        <div style='background:{bg};padding:20px;
        border-radius:10px;text-align:center'>
        <h2 style='color:{tc};margin:0'>{pred:.0f}</h2>
        <p style='color:{tc};margin:4px 0'>
        <b>{day}</b></p>
        <p style='color:{tc};margin:0;
        font-size:13px'>{lbl}</p>
        </div>""", unsafe_allow_html=True)

# ── HAZARD ALERT ─────────────────────────────────────
max_pred = max(pred_24h, pred_48h, pred_72h)
if max_pred >= 200:
    st.error(
        "🚨 Very Unhealthy air quality predicted! "
        "Avoid all outdoor activity.")
elif max_pred >= 150:
    st.warning(
        "⚠️ Unhealthy air quality predicted. "
        "Sensitive groups should stay indoors.")
else:
    st.success("✅ Air quality forecast looks acceptable.")

st.divider()

# ── HISTORICAL CHART ─────────────────────────────────
st.subheader("AQI Trend — Last 7 Days")

last7 = df.tail(7 * 24).copy()
last7["color"] = last7["aqi"].apply(
    lambda x: aqi_color(x)[0])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=last7["timestamp"],
    y=last7["aqi"],
    mode="lines",
    fill="tozeroy",
    name="AQI",
    line=dict(color="#ff7e00", width=2),
    fillcolor="rgba(255,126,0,0.1)"
))
fig.add_hline(y=50,  line_dash="dot",
              line_color="#00e400",
              annotation_text="Good")
fig.add_hline(y=100, line_dash="dot",
              line_color="#ffff00",
              annotation_text="Moderate")
fig.add_hline(y=150, line_dash="dot",
              line_color="#ff7e00",
              annotation_text="Unhealthy (Sensitive)")
fig.add_hline(y=200, line_dash="dot",
              line_color="#ff0000",
              annotation_text="Unhealthy")
fig.update_layout(
    xaxis_title="Time",
    yaxis_title="AQI",
    height=400,
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── POLLUTANTS ───────────────────────────────────────
st.subheader("Current Pollutant Levels (μg/m³)")

pollutants = {
    "PM2.5": float(latest["pm25"]),
    "PM10":  float(latest["pm10"]),
    "O3":    float(latest["o3"]),
    "NO2":   float(latest["no2"]),
    "SO2":   float(latest["so2"]),
    "CO":    float(latest["co"]),
}
p_cols = st.columns(6)
for col, (name, val) in zip(p_cols, pollutants.items()):
    with col:
        st.metric(name, f"{val:.1f}")

st.divider()

# ── SHAP FEATURE IMPORTANCE ──────────────────────────
st.subheader("What drives the forecast?")
st.caption("Feature importance from the trained model")

try:
    import shap
    explainer  = shap.Explainer(model, row)
    shap_vals  = explainer(row)
    importance = pd.DataFrame({
        "Feature":   FEATURES,
        "Importance": np.abs(shap_vals.values[0])
    }).sort_values("Importance", ascending=True).tail(10)

    fig2 = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Features (SHAP values)",
        color="Importance",
        color_continuous_scale="Oranges"
    )
    fig2.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.info("Feature importance chart not available: "
            f"{e}")

st.divider()

# ── AQI SCALE REFERENCE ──────────────────────────────
st.subheader("AQI Scale Reference")
scale_data = {
    "Category":   ["Good", "Moderate",
                   "Unhealthy (Sensitive)",
                   "Unhealthy", "Very Unhealthy",
                   "Hazardous"],
    "AQI Range":  ["0–50", "51–100", "101–150",
                   "151–200", "201–300", "301–500"],
    "Color":      ["🟢", "🟡", "🟠",
                   "🔴", "🟣", "🟤"]
}
st.dataframe(
    pd.DataFrame(scale_data),
    hide_index=True,
    use_container_width=True
)