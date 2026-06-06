"""
app.py  –  v5

Changes vs v4:
  1. Ensemble inference — loads all model artifacts + weights, blends predictions
  2. Side-by-side UI — AQI banner left, pollutants + weather right
  3. Fixed: UserWarning "X does not have valid feature names"
     Pass DataFrame (not .values) to scaler.transform() so column names are kept
  4. Fixed: use_container_width → width='stretch' (Streamlit deprecation)
  5. Fixed: InconsistentVersionWarning — suppress with warnings.filterwarnings
     The versions differ between CI (1.7.2) and local (1.9.0). Predictions are
     still correct for RF/Ridge across minor sklearn versions. Suppressed so
     the terminal output is readable.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import hopsworks
from dotenv import load_dotenv

# Suppress sklearn version mismatch warnings — predictions are still valid
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*valid feature names.*")

load_dotenv()

st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌫️",
    layout="wide"
)

# ── Feature list: must match training_pipeline.py FEATURES exactly ──
FEATURES = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "dust", "european_aqi", "us_aqi",
    "temp", "humidity", "wind", "wind_gusts",
    "precipitation", "pressure", "cloud_cover",
    "dew_point", "apparent_temp",
    # solar_rad removed — always NaN from OpenWeather, causes spurious SHAP
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

SEQ_LEN = 24  # must match training_pipeline.py


def aqi_category(val):
    if val <= 50:   return "#00e400", "#000000", "Good"
    if val <= 100:  return "#ffff00", "#000000", "Moderate"
    if val <= 150:  return "#ff7e00", "#000000", "Unhealthy for Sensitive"
    if val <= 200:  return "#ff0000", "#ffffff", "Unhealthy"
    if val <= 300:  return "#8f3f97", "#ffffff", "Very Unhealthy"
    return "#7e0023", "#ffffff", "Hazardous"


def is_lstm(model):
    try:
        import tensorflow as tf
        return isinstance(model, tf.keras.Model)
    except ImportError:
        return False


@st.cache_resource(show_spinner="Loading models from Hopsworks...")
def load_models():
    """
    Load models once and cache permanently — models only change after retraining.
    Separated from data loading so the hourly data refresh doesn't force
    a full model reload (which takes 2-3 minutes due to TensorFlow).
    """
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )

    mr = project.get_model_registry()
    best_models  = {}
    scalers      = {}
    ensemble_all = {}
    ensemble_wts = {}

    for horizon in ["1h", "24h", "48h", "72h"]:
        try:
            all_versions = mr.get_models(name=f"aqi_model_{horizon}")
            if not all_versions:
                continue
            m    = sorted(all_versions, key=lambda x: x.version)[-1]
            mdir = m.download()
            print(f"Loaded {horizon} model v{m.version}")

            pkl  = os.path.join(mdir, f"model_{horizon}.pkl")
            stub = joblib.load(pkl)
            if isinstance(stub, dict) and stub.get("type") == "lstm":
                import tensorflow as tf
                best_models[horizon] = tf.keras.models.load_model(
                    os.path.join(mdir, stub["path"]))
            else:
                best_models[horizon] = stub

            sc = os.path.join(mdir, f"scaler_{horizon}.pkl")
            scalers[horizon] = joblib.load(sc) if os.path.exists(sc) else None

            all_p = os.path.join(mdir, f"all_{horizon}.pkl")
            if os.path.exists(all_p):
                bundle = joblib.load(all_p)
                loaded = {}
                for name, entry in bundle.items():
                    mo = entry["model"]
                    if isinstance(mo, dict) and mo.get("type") == "lstm":
                        import tensorflow as tf
                        kp = os.path.join(mdir, mo["path"])
                        if os.path.exists(kp):
                            mo = tf.keras.models.load_model(kp)
                        else:
                            continue
                    loaded[name] = {**entry, "model": mo}
                ensemble_all[horizon] = loaded

            wts_p = os.path.join(mdir, f"ensemble_weights_{horizon}.pkl")
            if os.path.exists(wts_p):
                ensemble_wts[horizon] = joblib.load(wts_p)

        except Exception as e:
            print(f"  [WARN] {horizon}: {e}")

    return project, best_models, scalers, ensemble_all, ensemble_wts


@st.cache_data(ttl=3600, show_spinner="Fetching latest AQI data...")
def load_data(_project=None):
    """
    Load feature data from MongoDB Atlas and refresh every hour.
    Returns None if MongoDB is empty or unavailable.
    """
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipelines'))
        from mongo_store import read_df
        df = read_df()
        if df is None or len(df) == 0:
            return None
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["aqi_trend"] = (
            df["aqi"].rolling(168, min_periods=24).mean()
            .fillna(df["aqi"].expanding().mean())
        )
        return df
    except Exception as e:
        print(f"[WARN] Could not load data from MongoDB: {e}")
        return None


def _impute(df_row):
    """
    Fill NaN values before passing to any model.
    NaNs arise from:
      - solar_rad: OpenWeather free tier doesn't provide it (always NaN)
      - aqi_lag168: needs 168h of history — NaN until pipeline has run 7 days
      - aqi_lag96/120: same — NaN for first few days of data
    Strategy: fill each column with its median across the full df history,
    falling back to 0 if median is also NaN.
    Called on a single-row or multi-row DataFrame before transform/predict.
    """
    return df_row.fillna(df_row.median()).fillna(0)


def _scale(scaler, df_row):
    """
    Pass a DataFrame to scaler.transform() keeping column names.
    Handles schema mismatch between old models (61 features, with solar_rad)
    and new models (60 features, solar_rad removed):
      - If scaler expects solar_rad but df_row doesn't have it: add as 0.0
      - If scaler doesn't expect solar_rad but df_row has it: drop it
    """
    if scaler is None:
        return df_row.values

    # Get the columns the scaler was fitted on
    if hasattr(scaler, "feature_names_in_"):
        expected = list(scaler.feature_names_in_)
        # Add any missing columns as 0 (e.g. solar_rad in old scaler)
        for col in expected:
            if col not in df_row.columns:
                df_row = df_row.copy()
                df_row[col] = 0.0
        # Drop any extra columns not in the scaler (e.g. solar_rad removed)
        df_row = df_row[expected]

    return scaler.transform(df_row)


def _predict_one_model(model, scaler, df, feat_cols):
    """
    Run inference for one model. Returns predicted DIFF (not absolute AQI).
    NaN values are imputed with column medians before any model call.
    LSTM: uses last SEQ_LEN rows as a sequence.
    sklearn/XGBoost: uses single latest row.
    """
    if is_lstm(model):
        recent = df[feat_cols].tail(SEQ_LEN).copy()
        if len(recent) < SEQ_LEN:
            pad    = pd.DataFrame(
                [recent.iloc[0].values] * (SEQ_LEN - len(recent)),
                columns=feat_cols)
            recent = pd.concat([pad, recent], ignore_index=True)
        recent = _impute(recent)
        X = _scale(scaler, recent)
        return float(model.predict(
            X.reshape(1, SEQ_LEN, X.shape[1]), verbose=0)[0][0])
    else:
        row = df[feat_cols].iloc[[-1]].copy()
        row = _impute(row)
        X   = _scale(scaler, row)
        return float(model.predict(X)[0])


def predict_ensemble(horizon, df, best_models, scalers,
                     ensemble_all, ensemble_wts):
    """
    Weighted ensemble: sum(weight_i * diff_i) across positive-R² models.
    Falls back to single best model if ensemble artifacts absent.
    Reconstruction: aqi_now + weighted_diff.
    """
    feat_cols = [f for f in FEATURES if f in df.columns]
    aqi_now   = float(df["aqi"].iloc[-1])

    if horizon in ensemble_all and horizon in ensemble_wts:
        wts    = ensemble_wts[horizon]
        bundle = ensemble_all[horizon]
        scaler = scalers.get(horizon)
        w_diff, w_total = 0.0, 0.0
        for name, weight in wts.items():
            if name not in bundle:
                continue
            try:
                diff    = _predict_one_model(
                    bundle[name]["model"], scaler, df, feat_cols)
                w_diff  += weight * diff
                w_total += weight
            except Exception as e:
                print(f"    [WARN] ensemble {name}: {e}")
        if w_total > 0:
            return float(np.clip(aqi_now + w_diff / w_total, 15, 500)), "ensemble"

    if horizon in best_models:
        scaler = scalers.get(horizon)
        diff   = _predict_one_model(
            best_models[horizon], scaler, df, feat_cols)
        return float(np.clip(aqi_now + diff, 15, 500)), "single"

    return aqi_now, "fallback"


def iterative_1h_forecast(df, best_models, scalers, hours=72):
    model  = best_models.get("1h")
    scaler = scalers.get("1h")
    if model is None:
        return []

    feat_cols = [f for f in FEATURES if f in df.columns]
    row       = df[feat_cols].iloc[-1].copy().to_dict()
    preds     = []

    for _ in range(hours):
        row_df  = _impute(pd.DataFrame([row])[feat_cols])
        X       = _scale(scaler, row_df)
        if is_lstm(model):
            diff = float(model.predict(
                X.reshape(1, 1, X.shape[1]), verbose=0)[0][0])
        else:
            diff = float(model.predict(X)[0])

        cur      = float(row["aqi"])
        nxt      = float(np.clip(cur + diff, 15, 500))
        preds.append(nxt)

        row["aqi_lag3"]  = row["aqi_lag2"]
        row["aqi_lag2"]  = row["aqi_lag1"]
        row["aqi_lag1"]  = cur
        row["aqi"]       = nxt
        for w, k in [(3,"aqi_roll3_mean"),(6,"aqi_roll6_mean"),
                     (12,"aqi_roll12_mean"),(24,"aqi_roll24_mean"),
                     (48,"aqi_roll48_mean"),(72,"aqi_roll72_mean")]:
            row[k] = (row[k] * (w-1) + nxt) / w
        row["aqi_diff1"]       = nxt - cur
        row["aqi_change_rate"] = row["aqi_diff1"] / cur if cur else 0.0
        row["aqi_trend"]       = (row["aqi_trend"] * 167 + nxt) / 168
        new_hour               = (int(row["hour"]) + 1) % 24
        row["hour"]            = new_hour
        row["hour_sin"]        = np.sin(2 * np.pi * new_hour / 24)
        row["hour_cos"]        = np.cos(2 * np.pi * new_hour / 24)
        row["is_weekend"]      = int(int(row.get("day_of_week", 0)) in [5, 6])
    return preds


def fmt(val, decimals=1):
    try:
        v = float(val)
        return "N/A" if pd.isna(v) else f"{v:.{decimals}f}"
    except Exception:
        return "N/A"


# ── LOAD ────────────────────────────────────────────────────────────────
project, best_models, scalers, ensemble_all, ensemble_wts = load_models()
df = load_data()

if df is None or not best_models:
    st.warning("⏳ Data is not yet available — the Hopsworks feature store is initializing. This happens after a fresh setup and resolves automatically within a few minutes once the data pipeline completes. Please refresh the page shortly.")
    st.stop()

latest      = df.iloc[-1]
current_aqi = float(latest["aqi"])
bg, tc, lbl = aqi_category(current_aqi)

# ── TITLE ───────────────────────────────────────────────────────────────
st.title("🌫️ Karachi Air Quality Forecast")
st.caption(f"Last updated: {latest['timestamp']}")

# ═══════════════════════════════════════════════════════════════════════
# ROW 1: AQI banner (left) + Pollutants & Weather (right)
# ═══════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 1.7], gap="large")

with left:
    st.markdown(f"""
    <div style='background:{bg};padding:36px 24px;border-radius:14px;
    text-align:center;box-shadow:0 4px 14px rgba(0,0,0,0.15);height:100%;
    box-sizing:border-box;display:flex;flex-direction:column;
    justify-content:center'>
    <div style='color:{tc};font-size:68px;font-weight:800;
    letter-spacing:-2px;line-height:1'>AQI {current_aqi:.0f}</div>
    <div style='color:{tc};font-size:18px;font-weight:600;
    margin:10px 0 4px'>{lbl}</div>
    <div style='color:{tc};font-size:14px;opacity:0.8'>
    Karachi · Real-time reading</div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("**Current Pollutants**")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    with c1: st.metric("PM2.5 (μg/m³)", fmt(latest.get("pm25")))
    with c2: st.metric("PM10 (μg/m³)",  fmt(latest.get("pm10")))
    with c3: st.metric("O₃ (μg/m³)",    fmt(latest.get("o3")))
    with c4: st.metric("NO₂ (μg/m³)",   fmt(latest.get("no2")))
    with c5: st.metric("SO₂ (μg/m³)",   fmt(latest.get("so2")))
    with c6: st.metric("CO (μg/m³)",    fmt(latest.get("co")))

    st.markdown("**Current Weather**")
    w1, w2, w3, w4, w5 = st.columns(5)
    with w1: st.metric("Temp",     f"{fmt(latest.get('temp'))}°C")
    with w2: st.metric("Humidity", f"{fmt(latest.get('humidity'),0)}%")
    with w3: st.metric("Wind",     f"{fmt(latest.get('wind'))} m/s")
    with w4: st.metric("Pressure", f"{fmt(latest.get('pressure'),0)} hPa")
    with w5: st.metric("Cloud",    f"{fmt(latest.get('cloud_cover'),0)}%")

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# ROW 2: 3-Day Forecast Cards
# ═══════════════════════════════════════════════════════════════════════
st.subheader("3-Day AQI Forecast")

has_direct = all(h in best_models for h in ["24h", "48h", "72h"])
has_1h     = "1h" in best_models

if has_direct:
    p24, s24 = predict_ensemble("24h", df, best_models, scalers,
                                 ensemble_all, ensemble_wts)
    p48, s48 = predict_ensemble("48h", df, best_models, scalers,
                                 ensemble_all, ensemble_wts)
    p72, s72 = predict_ensemble("72h", df, best_models, scalers,
                                 ensemble_all, ensemble_wts)
    method = f"Direct 24h/48h/72h models ({s24}/{s48}/{s72})"
elif has_1h:
    hourly = iterative_1h_forecast(df, best_models, scalers, 72)
    p24    = float(np.mean(hourly[0:24]))  if len(hourly) >= 24 else current_aqi
    p48    = float(np.mean(hourly[24:48])) if len(hourly) >= 48 else current_aqi
    p72    = float(np.mean(hourly[48:72])) if len(hourly) >= 72 else current_aqi
    method = "Iterative 1h model (fallback)"
else:
    st.error("No models found. Run the training pipeline first.")
    st.stop()

fc1, fc2, fc3 = st.columns(3)
for col, (day, pred) in zip([fc1, fc2, fc3], [
    ("Tomorrow (Day 1)", p24),
    ("Day 2",            p48),
    ("Day 3",            p72),
]):
    bg2, tc2, lbl2 = aqi_category(pred)
    with col:
        st.markdown(f"""
        <div style='background:{bg2};padding:24px;border-radius:12px;
        text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.12)'>
        <div style='color:{tc2};font-size:46px;font-weight:700;
        line-height:1'>{pred:.0f}</div>
        <div style='color:{tc2};font-size:15px;font-weight:600;
        margin:8px 0 3px'>{day}</div>
        <div style='color:{tc2};font-size:12px;opacity:0.9'>{lbl2}</div>
        </div>""", unsafe_allow_html=True)

st.caption(f"Forecast method: {method}")

# ══════════════════════════════════════════════════════════════════════
# ALERT SYSTEM — current + forecast, with health guidance
# ══════════════════════════════════════════════════════════════════════

ALERT_CONFIG = {
    # (min_aqi, bg_color, icon, title, who, actions)
    "hazardous":    (300, "#7e0023", "☠️",  "HAZARDOUS",
                     "Everyone",
                     ["Stay indoors with windows sealed",
                      "Use N95/P100 respirator if going outside",
                      "Run air purifier on highest setting",
                      "Seek medical attention if experiencing symptoms",
                      "Avoid ALL physical exertion outdoors"]),
    "very_unhealthy":(200, "#8f3f97", "🚨", "VERY UNHEALTHY",
                      "Everyone",
                      ["Avoid all outdoor activity",
                       "Keep windows closed",
                       "Use air purifier indoors",
                       "Wear N95 mask if outside is unavoidable"]),
    "unhealthy":    (150, "#ff0000", "⚠️",  "UNHEALTHY",
                     "Everyone may be affected; sensitive groups most at risk",
                     ["Limit prolonged outdoor exertion",
                      "Children and elderly should stay indoors",
                      "Wear a mask outdoors if possible"]),
    "sensitive":    (100, "#ff7e00", "⚠️",  "UNHEALTHY FOR SENSITIVE GROUPS",
                     "People with asthma, heart/lung disease, elderly, children",
                     ["Sensitive groups: reduce outdoor activity",
                      "Keep rescue medication accessible",
                      "Monitor symptoms closely"]),
    "moderate":     ( 50, "#b8a000", "ℹ️",  "MODERATE",
                     "Unusually sensitive individuals",
                     ["Consider reducing prolonged outdoor exertion",
                      "People with respiratory conditions: take precautions"]),
    "good":         (  0, "#006400", "✅",  "GOOD",
                     "Air quality is satisfactory for all",
                     ["Enjoy outdoor activities"]),
}


def get_alert_config(aqi_val):
    if aqi_val >= 300: return ALERT_CONFIG["hazardous"]
    if aqi_val >= 200: return ALERT_CONFIG["very_unhealthy"]
    if aqi_val >= 150: return ALERT_CONFIG["unhealthy"]
    if aqi_val >= 100: return ALERT_CONFIG["sensitive"]
    if aqi_val >= 50:  return ALERT_CONFIG["moderate"]
    return ALERT_CONFIG["good"]


def render_alert_banner(aqi_val, context="Current AQI"):
    """Render a full-width alert banner with health guidance."""
    min_aqi, bg, icon, title, who, actions = get_alert_config(aqi_val)
    is_dark_bg = aqi_val >= 150
    tc_alert   = "#ffffff" if is_dark_bg else "#1a1a1a"
    action_html = "".join(f"<li>{a}</li>" for a in actions)
    st.markdown(f"""
    <div style='background:{bg};padding:20px 24px;border-radius:12px;
    margin:8px 0;box-shadow:0 3px 10px rgba(0,0,0,0.2)'>
      <div style='color:{tc_alert};font-size:18px;font-weight:700;
      margin-bottom:6px'>{icon} {context} — {title}</div>
      <div style='color:{tc_alert};font-size:13px;opacity:0.9;
      margin-bottom:8px'><b>Who is affected:</b> {who}</div>
      <ul style='color:{tc_alert};font-size:13px;margin:0;
      padding-left:18px;opacity:0.95'>{action_html}</ul>
    </div>
    """, unsafe_allow_html=True)


# ── Current AQI alert ──────────────────────────────────────────────────
render_alert_banner(current_aqi, f"AQI {current_aqi:.0f} right now")

# ── Forecast alerts per day ────────────────────────────────────────────
forecast_vals = [("Tomorrow", p24), ("Day 2", p48), ("Day 3", p72)]
worst_forecast = max(p24, p48, p72)

# Only show per-day breakdown if conditions are concerning
if worst_forecast >= 100:
    st.markdown("**Forecast health guidance:**")
    fa1, fa2, fa3 = st.columns(3)
    for col, (day, val) in zip([fa1, fa2, fa3], forecast_vals):
        _, bg2, icon2, title2, _, _ = get_alert_config(val)
        is_dark = val >= 150
        tc2 = "#ffffff" if is_dark else "#1a1a1a"
        with col:
            st.markdown(f"""
            <div style='background:{bg2};padding:12px 16px;border-radius:8px;
            font-size:13px;box-shadow:0 1px 4px rgba(0,0,0,0.1)'>
            <b style='color:{tc2}'>{icon2} {day}</b>
            <div style='color:{tc2};opacity:0.9'>{title2}</div>
            </div>""", unsafe_allow_html=True)

# Special hazardous full-width banner
if worst_forecast >= 300 or current_aqi >= 300:
    st.markdown("""
    <div style='background:#7e0023;padding:16px 24px;border-radius:10px;
    border:2px solid #ff0000;margin:12px 0;text-align:center'>
    <div style='color:#ffffff;font-size:22px;font-weight:800'>
    ☠️ HAZARDOUS AIR QUALITY ALERT ☠️</div>
    <div style='color:#ffcccc;font-size:14px;margin-top:6px'>
    This is a public health emergency. Everyone should remain indoors.
    Seal windows and doors. Use an air purifier. Contact a doctor if
    you experience difficulty breathing, chest pain, or dizziness.</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()



# ── 7-Day Historical Chart ─────────────────────────────────────────────
st.subheader("AQI History — Last 7 Days")
last7 = df.tail(7 * 24).copy()
fig2  = go.Figure()
fig2.add_trace(go.Scatter(
    x=last7["timestamp"], y=last7["aqi"],
    mode="lines", fill="tozeroy", name="Observed AQI",
    line=dict(color="#ff7e00", width=2.5),
    fillcolor="rgba(255,126,0,0.1)"))
for yv, col, txt in [(50,"#00e400","Good"),(100,"#ffa500","Moderate"),
                     (150,"#ff7e00","Sensitive"),(200,"#ff0000","Unhealthy")]:
    fig2.add_hline(y=yv, line_dash="dot", line_color=col,
                   annotation_text=txt, annotation_position="right",
                   annotation_font_size=11)
fig2.update_layout(
    xaxis_title="Time", yaxis_title="AQI", height=360,
    margin=dict(l=0,r=70,t=10,b=0), hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig2, width="stretch")

st.divider()

# ── AQI Reference Table ────────────────────────────────────────────────
st.subheader("AQI Scale Reference")
st.dataframe(pd.DataFrame({
    "": ["🟢","🟡","🟠","🔴","🟣","🟤"],
    "Category": ["Good","Moderate","Unhealthy for Sensitive Groups",
                  "Unhealthy","Very Unhealthy","Hazardous"],
    "AQI Range": ["0–50","51–100","101–150","151–200","201–300","301–500"],
    "Health Implications": [
        "Air quality is satisfactory",
        "Acceptable; minor risk for very sensitive people",
        "Sensitive groups may experience health effects",
        "Everyone may begin to experience health effects",
        "Health alert: serious effects for everyone",
        "Health warnings — emergency conditions"],
}), hide_index=True, width="stretch")

st.divider()
st.caption(
    "Data: Open-Meteo (AQ) + OpenWeather · "
    "Feature store & models: Hopsworks · "
    "Pipeline: GitHub Actions · Built with Streamlit"
)