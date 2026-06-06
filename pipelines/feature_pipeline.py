"""
feature_pipeline.py  –  v6

WHAT CHANGED vs v4 and WHY:
=============================================================================

1. CRASH FIX: KeyError 'aqi' on empty history DataFrame
2. AQI SOURCE: Open-Meteo air-quality-api (consistent with backfill training data)
3. WEATHER SOURCE CHANGED: api.open-meteo.com/v1/forecast → OpenWeatherMap
   The Open-Meteo Forecast API has intermittent 502 outages (83% uptime on
   Jun 04). The air-quality-api is on a separate stable server (100% uptime).
   OpenWeather current weather API is extremely reliable.
   This eliminates the 502 errors from the hourly pipeline.
4. RETRY LOGIC: _get_json() retries 5× with 15s backoff for transient errors.
5. GRACEFUL EXIT: API outage exits with code 0 so GitHub Actions does not
   send failure notifications for a skipped hourly row.
6. FORECAST FEATURES DECOUPLED FROM TRAINING SCHEMA (no train/serve skew)
7. confluent_kafka preflight check
=============================================================================
"""

import sys
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import hopsworks

# ── Preflight: confluent_kafka must be present ────────────────────────────
try:
    import confluent_kafka  # noqa: F401
except ImportError:
    print(
        "\n[ERROR] confluent_kafka is not installed.\n"
        "Hopsworks 4.7 routes all fg.insert() calls through Kafka.\n"
        "Fix:  pip install confluent-kafka\n"
        "  or: pip install \"hopsworks[python]\"\n"
    )
    sys.exit(1)

load_dotenv()

LAT = 24.8607
LON = 67.0011

# OpenWeather API key — add OPENWEATHER_API_KEY to your .env and GitHub secrets
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")


def pm25_to_aqi(pm25):
    """Fallback: compute US AQI from PM2.5 only."""
    if np.isnan(pm25):
        return np.nan
    breakpoints = [
        (0.0,    12.0,   0,   50),
        (12.1,   35.4,  51,  100),
        (35.5,   55.4, 101,  150),
        (55.5,  150.4, 151,  200),
        (150.5, 250.4, 201,  300),
        (250.5, 350.4, 301,  400),
        (350.5, 500.4, 401,  500),
    ]
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            return round(
                ((aqi_hi - aqi_lo) / (bp_hi - bp_lo))
                * (pm25 - bp_lo) + aqi_lo
            )
    return 500


def _get_json(url, timeout=20, retries=5, backoff=15):
    """
    Fetch a URL and return parsed JSON with retry logic.
    Open-Meteo occasionally returns 502/empty under load or during brief outages.
    5 retries × 15s backoff = up to 75 seconds of waiting before giving up.
    """
    import time as _time
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code != 200:
                raise ValueError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}")
            if not resp.text.strip():
                raise ValueError("Empty response body")
            return resp.json()
        except Exception as e:
            last_err = e
            print(f"  [WARN] Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                _time.sleep(backoff)
    raise RuntimeError(
        f"All {retries} attempts failed for {url}. Last error: {last_err}")


def fetch_current():
    """
    Fetch current conditions using two separate stable sources:

    1. air-quality-api.open-meteo.com → AQI + pollutants
       Same server as backfill (100% uptime on status page).
       Same source as training data → no train/serve skew on AQI values.

    2. api.openweathermap.org → weather (temp, wind, humidity, pressure)
       Replaces api.open-meteo.com/v1/forecast which had 502 outages.
       Free tier, requires OPENWEATHER_API_KEY env var.
    """
    if not OPENWEATHER_API_KEY:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is not set.\n"
            "Get a free key at openweathermap.org and add it to your "
            ".env file and GitHub Actions secrets as OPENWEATHER_API_KEY."
        )

    # ── Air quality: Open-Meteo (same source as training data) ────────
    aq_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        f"&current=pm2_5,pm10,carbon_monoxide,"
        f"nitrogen_dioxide,sulphur_dioxide,ozone,"
        f"dust,european_aqi,us_aqi"
    )
    aq_r = _get_json(aq_url)
    aq   = aq_r.get("current", {})

    # ── Weather: OpenWeatherMap (replaces unstable Forecast API) ──────
    wx_url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={LAT}&lon={LON}"
        f"&appid={OPENWEATHER_API_KEY}"
        f"&units=metric"
    )
    wx_r  = _get_json(wx_url)
    main  = wx_r.get("main",   {})
    wind  = wx_r.get("wind",   {})
    cloud = wx_r.get("clouds", {})
    rain  = wx_r.get("rain",   {})

    def s(d, k):
        v = d.get(k)
        return float(v) if v is not None else np.nan

    pm25   = s(aq, "pm2_5")
    us_aqi = s(aq, "us_aqi")
    temp   = float(main.get("temp",       np.nan))
    hum    = float(main.get("humidity",   np.nan))
    pres   = float(main.get("pressure",   np.nan))
    wsp    = float(wind.get("speed",      np.nan))   # already m/s with units=metric
    wdir   = float(wind.get("deg",        np.nan))
    wgust  = float(wind.get("gust",       np.nan) if wind.get("gust") else np.nan)
    prec   = float(rain.get("1h",         0.0))
    cc     = float(cloud.get("all",       np.nan))
    app    = float(main.get("feels_like", np.nan))

    # OpenWeather free tier has no dew point — compute from Magnus formula
    if not (np.isnan(temp) or np.isnan(hum)):
        a, b  = 17.27, 237.7
        alpha = (a * temp) / (b + temp) + np.log(max(hum, 1) / 100.0)
        dew   = (b * alpha) / (a - alpha)
    else:
        dew = np.nan

    # solar_rad not available from OpenWeather free tier — store NaN
    sol = np.nan

    aqi_val = us_aqi if not np.isnan(us_aqi) else (
        float(pm25_to_aqi(pm25)) if not np.isnan(pm25) else np.nan
    )

    return {
        "aqi":           aqi_val,
        "pm25":          pm25,
        "pm10":          s(aq, "pm10"),
        "o3":            s(aq, "ozone"),
        "no2":           s(aq, "nitrogen_dioxide"),
        "so2":           s(aq, "sulphur_dioxide"),
        "co":            s(aq, "carbon_monoxide"),
        "dust":          s(aq, "dust"),
        "european_aqi":  s(aq, "european_aqi"),
        "us_aqi":        us_aqi,
        "temp":          temp,
        "humidity":      hum,
        "wind":          wsp,
        "wind_gusts":    wgust,
        "precipitation": prec,
        "pressure":      pres,
        "cloud_cover":   cc,
        "dew_point":     dew,
        "apparent_temp": app,
        "solar_rad":     sol,
        "temp_humidity": temp * hum / 100
                         if not (np.isnan(temp) or np.isnan(hum))
                         else np.nan,
        "wind_dir_sin":  float(np.sin(np.radians(wdir)))
                         if not np.isnan(wdir) else np.nan,
        "wind_dir_cos":  float(np.cos(np.radians(wdir)))
                         if not np.isnan(wdir) else np.nan,
        "pm25_wind":     pm25 / (wsp + 0.1)
                         if not (np.isnan(pm25) or np.isnan(wsp))
                         else np.nan,
        "dew_depression": temp - dew
                          if not (np.isnan(temp) or np.isnan(dew))
                          else np.nan,
    }


def fetch_weather_forecasts(now_ts):
    """
    Fetch +24h/+48h/+72h weather forecasts from Open-Meteo.
    These are NOT stored in the feature group (to avoid train/serve skew).
    They are returned for use at inference time only.
    """
    fc_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,wind_speed_10m,"
        f"precipitation,surface_pressure"
        f"&forecast_days=4"
        f"&wind_speed_unit=ms&timezone=UTC"
    )
    try:
        r = _get_json(fc_url)
    except Exception as e:
        print(f"  [WARN] Forecast fetch failed: {e}")
        return {}

    if "hourly" not in r:
        print(f"  [WARN] No hourly data in forecast response")
        return {}

    times   = pd.to_datetime(r["hourly"]["time"], utc=True)
    temps   = r["hourly"]["temperature_2m"]
    winds   = r["hourly"]["wind_speed_10m"]
    precips = r["hourly"]["precipitation"]
    presss  = r["hourly"]["surface_pressure"]

    def get_at_offset(hours_ahead):
        target = now_ts + pd.Timedelta(hours=hours_ahead)
        diffs  = abs(times - target)
        idx    = diffs.argmin()
        if diffs[idx] > pd.Timedelta(hours=2):
            return np.nan, np.nan, np.nan, np.nan
        def sv(lst):
            v = lst[idx]
            return float(v) if v is not None else np.nan
        return sv(temps), sv(winds), sv(precips), sv(presss)

    result = {}
    for h in [24, 48, 72]:
        t, w, p, pr = get_at_offset(h)
        result[f"temp_forecast_{h}h"]     = t
        result[f"wind_forecast_{h}h"]     = w
        result[f"precip_forecast_{h}h"]   = p
        result[f"pressure_forecast_{h}h"] = pr

    for h in [24, 48, 72]:
        t  = result.get(f"temp_forecast_{h}h", np.nan)
        w  = result.get(f"wind_forecast_{h}h", np.nan)
        p  = result.get(f"precip_forecast_{h}h", np.nan)
        print(f"  Forecast +{h}h: temp={t:.1f}°C  wind={w:.1f}m/s  precip={p:.1f}mm")
    return result


def fetch_recent_history(n_hours=169):
    """
    Read the last n_hours rows from MongoDB for lag feature computation.
    n_hours=169 because lag168 requires 168 historical rows plus current = 169.
    Returns DataFrame sorted oldest-first, or empty DataFrame on failure.
    """
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from mongo_store import read_latest
        df = read_latest(n_hours)
        if df is None or len(df) == 0:
            print("  [WARN] MongoDB is empty — first run after backfill?")
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"  [WARN] Could not fetch history from MongoDB: {e}")
        return pd.DataFrame()


def compute_lag_features(history_df, current_aqi, current_pm25):
    """
    Compute all lag/rolling features for the new row from history.

    FIX: old code crashed with KeyError('aqi') when history_df was empty
    (empty DataFrame has no columns). Now we check for column presence
    explicitly and return NaN defaults safely on first run.
    """
    # Guard: history must have the expected columns
    has_aqi  = len(history_df) > 0 and "aqi"  in history_df.columns
    has_pm25 = len(history_df) > 0 and "pm25" in history_df.columns

    aqi_series  = (list(history_df["aqi"].values)  if has_aqi  else []) + [current_aqi]
    pm25_series = (list(history_df["pm25"].values) if has_pm25 else []) + [current_pm25]

    def lag(series, n):
        idx = len(series) - 1 - n
        return float(series[idx]) if idx >= 0 else np.nan

    def roll_mean(series, n):
        # Include current value in rolling window
        end   = len(series)          # exclusive, includes current
        start = max(0, end - n)
        window = series[start:end]
        return float(np.nanmean(window)) if len(window) > 0 else np.nan

    def roll_std(series, n):
        end    = len(series)
        start  = max(0, end - n)
        window = series[start:end]
        if len(window) < 2:
            return np.nan
        arr = np.array(window, dtype=float)
        return float(np.nanstd(arr, ddof=1))

    prev_aqi = lag(aqi_series, 1)
    lag6     = lag(aqi_series, 6)
    lag24    = lag(aqi_series, 24)

    return {
        "aqi_lag1":        prev_aqi,
        "aqi_lag2":        lag(aqi_series, 2),
        "aqi_lag3":        lag(aqi_series, 3),
        "aqi_lag6":        lag6,
        "aqi_lag12":       lag(aqi_series, 12),
        "aqi_lag24":       lag24,
        "aqi_lag48":       lag(aqi_series, 48),
        # Extended lags matching backfill v6 schema
        "aqi_lag72":       lag(aqi_series, 72),
        "aqi_lag96":       lag(aqi_series, 96),
        "aqi_lag120":      lag(aqi_series, 120),
        "aqi_lag168":      lag(aqi_series, 168),
        "aqi_roll3_mean":  roll_mean(aqi_series, 3),
        "aqi_roll6_mean":  roll_mean(aqi_series, 6),
        "aqi_roll12_mean": roll_mean(aqi_series, 12),
        "aqi_roll24_mean": roll_mean(aqi_series, 24),
        "aqi_roll48_mean": roll_mean(aqi_series, 48),
        "aqi_roll72_mean": roll_mean(aqi_series, 72),
        "aqi_roll6_std":   roll_std(aqi_series, 6),
        "aqi_roll24_std":  roll_std(aqi_series, 24),
        "aqi_change_rate": (
            (current_aqi - prev_aqi) / prev_aqi
            if (not np.isnan(prev_aqi)) and prev_aqi != 0
            else np.nan
        ),
        "aqi_diff1":  (current_aqi - prev_aqi)
                      if not np.isnan(prev_aqi)  else np.nan,
        "aqi_diff6":  (current_aqi - lag6)
                      if not np.isnan(lag6)       else np.nan,
        "aqi_diff24": (current_aqi - lag24)
                      if not np.isnan(lag24)      else np.nan,
        "pm25_lag1":        lag(pm25_series, 1),
        "pm25_lag24":       lag(pm25_series, 24),
        "pm25_roll6_mean":  roll_mean(pm25_series, 6),
    }


def compute_pressure_diff(history_df, current_pressure):
    """Pressure change since last stored row."""
    if (len(history_df) == 0
            or "pressure" not in history_df.columns
            or np.isnan(current_pressure)):
        return np.nan
    prev = float(history_df["pressure"].iloc[-1])
    return (current_pressure - prev) if not np.isnan(prev) else np.nan


def store_features(df, project=None):
    """
    Upsert one new row into MongoDB Atlas.
    project argument kept for backward compatibility but not used.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mongo_store import store_df

    df = df.copy()
    df["aqi"]         = df["aqi"].round().astype("int64")
    df["hour"]        = df["hour"].astype("int64")
    df["day_of_week"] = df["day_of_week"].astype("int64")
    df["month"]       = df["month"].astype("int64")
    df["is_weekend"]  = df["is_weekend"].astype("int64")
    df["timestamp"]   = pd.to_datetime(df["timestamp"], utc=True)

    store_df(df)
    print(f"Stored row — timestamp={df['timestamp'].iloc[0]}, "
          f"aqi={df['aqi'].iloc[0]}, columns={len(df.columns)}")


if __name__ == "__main__":
    print("=== AQI Feature Pipeline v6 ===")

    # ── Step 1: fetch current reading ─────────────────────────────
    try:
        data = fetch_current()
    except RuntimeError as e:
        print(f"\n[ERROR] API fetch failed: {e}")
        print("[INFO] Skipping this hourly run — no row stored.")
        print("[INFO] If this is a 401 error, check OPENWEATHER_API_KEY in GitHub secrets.")
        print("[INFO] If this is a 502 error, it is a transient Open-Meteo outage.")
        print("[INFO] The next hourly run will try again.")
        sys.exit(0)  # exit 0 = success, so GitHub Actions doesn't flag it
    now  = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    now_str = now.isoformat()
    ts   = pd.Timestamp(now_str)

    print(f"AQI={data['aqi']}  PM2.5={data['pm25']}  "
          f"Temp={data['temp']}°C  Humidity={data['humidity']}%  "
          f"Wind={data['wind']}m/s")

    # ── Step 2: fetch weather forecasts (NOT stored, inference only) ─
    print("Fetching weather forecasts (inference-only, not stored)...")
    forecast_feats = fetch_weather_forecasts(ts)

    # ── Step 3: connect to Hopsworks (for model registry only) ───
    print("Connecting to Hopsworks (model registry)...")
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )

    # ── Step 4: fetch history for real lag features ────────────────
    print("Fetching recent history for lag features...")
    history = fetch_recent_history()
    if len(history) == 0:
        print("  [WARN] No history available — lag features will be NaN")
        print("  [INFO] Run backfill_pipeline.py first to populate history")
    else:
        print(f"  Got {len(history)} historical rows "
              f"(timestamps: {history['timestamp'].iloc[0]} → "
              f"{history['timestamp'].iloc[-1]})")

    current_aqi  = float(data["aqi"])  if not np.isnan(data["aqi"])  else 0.0
    current_pm25 = float(data["pm25"]) if not np.isnan(data["pm25"]) else 0.0
    current_pres = float(data["pressure"]) if not np.isnan(data["pressure"]) else np.nan

    lag_feats     = compute_lag_features(history, current_aqi, current_pm25)
    pressure_diff = compute_pressure_diff(history, current_pres)

    print(f"  aqi_lag1={lag_feats['aqi_lag1']}  "
          f"aqi_lag24={lag_feats['aqi_lag24']}  "
          f"aqi_lag48={lag_feats['aqi_lag48']}")

    # ── Step 5: build feature row (NO forecast columns in schema) ──
    row = {
        "timestamp":   now,   # keep as datetime — store_features casts to datetime64
        **data,
        "pressure_diff": pressure_diff,
        "hour":        ts.hour,
        "day_of_week": ts.dayofweek,
        "month":       ts.month,
        "is_weekend":  int(ts.dayofweek in [5, 6]),
        "hour_sin":    np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos":    np.cos(2 * np.pi * ts.hour / 24),
        "month_sin":   np.sin(2 * np.pi * ts.month / 12),
        "month_cos":   np.cos(2 * np.pi * ts.month / 12),
        **lag_feats,
        # targets are unknown at inference time
        "target_1h":  np.nan,
        "target_24h": np.nan,
        "target_48h": np.nan,
        "target_72h": np.nan,
    }

    df = pd.DataFrame([row])
    print(f"\nRow to insert: timestamp={now}  aqi={row['aqi']}")

    # ── Step 6: store feature row ──────────────────────────────────
    store_features(df, project)
    print("Row submitted to Hopsworks. Materialization job running asynchronously (~2-3 min).")

    # ── Step 7: print forecast summary (available for app use) ────
    if forecast_feats:
        print("\nWeather forecasts (for inference, not stored):")
        for h in [24, 48, 72]:
            t = forecast_feats.get(f"temp_forecast_{h}h", np.nan)
            w = forecast_feats.get(f"wind_forecast_{h}h", np.nan)
            print(f"  +{h}h: temp={t:.1f}°C  wind={w:.1f}m/s")

    print("=== Pipeline Complete ===")