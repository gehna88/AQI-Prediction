"""
feature_pipeline.py  –  v4
Changes vs v3:
  1. DROPPED zero-importance features:
       soil_temp, ammonia, visibility, is_daytime,
       wind_chill, wind_dir (raw degrees),
       aqi_roll24_max, aqi_roll24_min
     wind_dir_sin and wind_dir_cos are still computed and kept.

  2. ADDED live weather forecast features:
       temp_forecast_24h/48h/72h
       wind_forecast_24h/48h/72h
       precip_forecast_24h/48h/72h
       pressure_forecast_24h/48h/72h
     Fetched from Open-Meteo forecast API (free, no key needed).
     These give 48h/72h models genuine future weather signals at
     inference time — the biggest structural improvement possible
     without changing the model architecture.

  3. All other fixes from v3 retained:
       - Real lag features computed from Hopsworks history
       - No NaN placeholders for lag/rolling columns
       - pressure_diff computed from previous row
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import hopsworks

load_dotenv()

LAT = 24.8607
LON = 67.0011


def pm25_to_aqi(pm25):
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            return round(
                ((aqi_hi - aqi_lo) / (bp_hi - bp_lo))
                * (pm25 - bp_lo) + aqi_lo
            )
    return 500


def fetch_current():
    """Fetch current AQI and weather readings."""
    aq_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        f"&current=pm2_5,pm10,carbon_monoxide,"
        f"nitrogen_dioxide,sulphur_dioxide,ozone,"
        f"dust,european_aqi,us_aqi"
        # ammonia removed — zero importance
    )
    wx_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&current=temperature_2m,relative_humidity_2m,"
        f"wind_speed_10m,wind_direction_10m,"
        f"wind_gusts_10m,precipitation,"
        f"surface_pressure,cloud_cover,"
        f"apparent_temperature,dew_point_2m,"
        f"shortwave_radiation"
        # visibility removed — zero importance
        f"&wind_speed_unit=ms"
    )

    aq_r = requests.get(aq_url, timeout=10).json()
    wx_r = requests.get(wx_url, timeout=10).json()

    aq = aq_r.get("current", {})
    wx = wx_r.get("current", {})

    def s(d, k):
        v = d.get(k)
        return float(v) if v is not None else np.nan

    pm25 = s(aq, "pm2_5")
    temp = s(wx, "temperature_2m")
    wind = s(wx, "wind_speed_10m")
    hum  = s(wx, "relative_humidity_2m")
    pres = s(wx, "surface_pressure")
    dew  = s(wx, "dew_point_2m")
    wdir = s(wx, "wind_direction_10m")
    sol  = s(wx, "shortwave_radiation")

    aqi_val = float(pm25_to_aqi(pm25)) \
              if not np.isnan(pm25) else np.nan

    return {
        "aqi":           aqi_val,
        "pm25":          pm25,
        "pm10":          s(aq, "pm10"),
        "o3":            s(aq, "ozone"),
        "no2":           s(aq, "nitrogen_dioxide"),
        "so2":           s(aq, "sulphur_dioxide"),
        "co":            s(aq, "carbon_monoxide"),
        "dust":          s(aq, "dust"),
        # ammonia dropped
        "european_aqi":  s(aq, "european_aqi"),
        "us_aqi":        s(aq, "us_aqi"),
        "temp":          temp,
        "humidity":      hum,
        "wind":          wind,
        # wind_dir raw degrees dropped
        "wind_gusts":    s(wx, "wind_gusts_10m"),
        "precipitation": s(wx, "precipitation"),
        "pressure":      pres,
        "cloud_cover":   s(wx, "cloud_cover"),
        # visibility dropped
        "dew_point":     dew,
        "apparent_temp": s(wx, "apparent_temperature"),
        # soil_temp dropped
        "solar_rad":     sol,
        # derived
        "temp_humidity": temp * hum / 100
                         if not (np.isnan(temp) or np.isnan(hum))
                         else np.nan,
        # wind_chill dropped
        "wind_dir_sin":  float(np.sin(np.radians(wdir)))
                         if not np.isnan(wdir) else np.nan,
        "wind_dir_cos":  float(np.cos(np.radians(wdir)))
                         if not np.isnan(wdir) else np.nan,
        "pm25_wind":     pm25 / (wind + 0.1)
                         if not (np.isnan(pm25) or np.isnan(wind))
                         else np.nan,
        "dew_depression":temp - dew
                         if not (np.isnan(temp) or np.isnan(dew))
                         else np.nan,
        # is_daytime dropped
    }


def fetch_weather_forecasts(now_ts):
    """
    Fetch hourly weather forecasts from Open-Meteo for the next 4 days.
    Returns forecast values at +24h, +48h, +72h offsets from now_ts.
    Free API, no key needed.
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
        r = requests.get(fc_url, timeout=10).json()
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
        # find closest hour in forecast
        diffs = abs(times - target)
        idx   = diffs.argmin()
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

    print(f"  Forecast +24h: temp={result['temp_forecast_24h']:.1f}°C  "
          f"wind={result['wind_forecast_24h']:.1f}m/s  "
          f"precip={result['precip_forecast_24h']:.1f}mm")
    print(f"  Forecast +48h: temp={result['temp_forecast_48h']:.1f}°C  "
          f"wind={result['wind_forecast_48h']:.1f}m/s  "
          f"precip={result['precip_forecast_48h']:.1f}mm")
    print(f"  Forecast +72h: temp={result['temp_forecast_72h']:.1f}°C  "
          f"wind={result['wind_forecast_72h']:.1f}m/s  "
          f"precip={result['precip_forecast_72h']:.1f}mm")
    return result


def fetch_recent_history(project, n_hours=49):
    """
    Read the last n_hours rows from Hopsworks to compute real lag features.
    Returns a DataFrame sorted oldest-first, or empty DataFrame on failure.
    """
    try:
        fs = project.get_feature_store()
        fg = fs.get_feature_group("aqi_features", version=1)
        df = fg.read()
        if df is None or len(df) == 0:
            return pd.DataFrame()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df.tail(n_hours).copy()
    except Exception as e:
        print(f"  [WARN] Could not fetch history: {e}")
        return pd.DataFrame()


def compute_lag_features(history_df, current_aqi, current_pm25):
    """
    Compute all lag/rolling features for the new row from history.
    Eliminates train-serving skew: no NaN placeholders.
    """
    aqi_series  = list(history_df["aqi"].values) + [current_aqi]
    pm25_series = list(history_df["pm25"].values) + [current_pm25]

    def lag(series, n):
        idx = len(series) - 1 - n
        return float(series[idx]) if idx >= 0 else np.nan

    def roll_mean(series, n):
        window = series[max(0, len(series)-1-n): len(series)-1]
        return float(np.mean(window)) if window else np.nan

    def roll_std(series, n):
        window = series[max(0, len(series)-1-n): len(series)-1]
        return float(np.std(window, ddof=1)) if len(window) >= 2 else np.nan

    prev_aqi = lag(aqi_series, 1)
    return {
        "aqi_lag1":        prev_aqi,
        "aqi_lag2":        lag(aqi_series, 2),
        "aqi_lag3":        lag(aqi_series, 3),
        "aqi_lag6":        lag(aqi_series, 6),
        "aqi_lag12":       lag(aqi_series, 12),
        "aqi_lag24":       lag(aqi_series, 24),
        "aqi_lag48":       lag(aqi_series, 48),
        "aqi_roll3_mean":  roll_mean(aqi_series, 3),
        "aqi_roll6_mean":  roll_mean(aqi_series, 6),
        "aqi_roll12_mean": roll_mean(aqi_series, 12),
        "aqi_roll24_mean": roll_mean(aqi_series, 24),
        "aqi_roll6_std":   roll_std(aqi_series, 6),
        "aqi_roll24_std":  roll_std(aqi_series, 24),
        # aqi_roll24_max and aqi_roll24_min dropped
        "aqi_change_rate": (
            (current_aqi - prev_aqi) / prev_aqi
            if prev_aqi and prev_aqi != 0 and not np.isnan(prev_aqi)
            else np.nan),
        "aqi_diff1":  current_aqi - prev_aqi
                      if not np.isnan(prev_aqi) else np.nan,
        "aqi_diff6":  current_aqi - lag(aqi_series, 6)
                      if not np.isnan(lag(aqi_series, 6)) else np.nan,
        "aqi_diff24": current_aqi - lag(aqi_series, 24)
                      if not np.isnan(lag(aqi_series, 24)) else np.nan,
        "pm25_lag1":       lag(pm25_series, 1),
        "pm25_lag24":      lag(pm25_series, 24),
        "pm25_roll6_mean": roll_mean(pm25_series, 6),
    }


def compute_pressure_diff(history_df, current_pressure):
    if len(history_df) == 0 or "pressure" not in history_df.columns:
        return np.nan
    prev = float(history_df["pressure"].iloc[-1])
    return (current_pressure - prev
            if not (np.isnan(prev) or np.isnan(current_pressure))
            else np.nan)


def store_features(df, project):
    df = df.copy()
    df["aqi"]         = df["aqi"].round().astype("int64")
    df["hour"]        = df["hour"].astype("int64")
    df["day_of_week"] = df["day_of_week"].astype("int64")
    df["month"]       = df["month"].astype("int64")
    df["is_weekend"]  = df["is_weekend"].astype("int64")

    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="Hourly AQI features Karachi — v4 (forecast weather)",
        online_enabled=False
    )
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"Stored {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    print("=== AQI Feature Pipeline v4 ===")

    # ── Step 1: fetch current reading ─────────────────────────────
    data = fetch_current()
    now  = datetime.now(timezone.utc).isoformat()
    ts   = pd.Timestamp(now)

    print(f"AQI={data['aqi']}  PM2.5={data['pm25']}  "
          f"Temp={data['temp']}°C  Humidity={data['humidity']}%  "
          f"Wind={data['wind']}m/s")

    # ── Step 2: fetch weather forecasts (+24h, +48h, +72h) ────────
    print("Fetching weather forecasts...")
    forecast_feats = fetch_weather_forecasts(ts)

    # ── Step 3: connect to Hopsworks ──────────────────────────────
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )

    # ── Step 4: fetch history for real lag features ────────────────
    print("Fetching recent history for lag features...")
    history = fetch_recent_history(project, n_hours=49)
    if len(history) == 0:
        print("  [WARN] No history — lag features will be NaN (first run?)")

    current_aqi  = float(data["aqi"]) if not np.isnan(data["aqi"]) else 0.0
    current_pm25 = float(data["pm25"]) if not np.isnan(data["pm25"]) else 0.0

    lag_feats     = compute_lag_features(history, current_aqi, current_pm25)
    pressure_diff = compute_pressure_diff(
        history,
        float(data["pressure"]) if not np.isnan(data["pressure"]) else np.nan)

    print(f"  Lag features from {len(history)} historical rows  "
          f"aqi_lag1={lag_feats['aqi_lag1']}  "
          f"aqi_lag24={lag_feats['aqi_lag24']}")

    # ── Step 5: build row ──────────────────────────────────────────
    df = pd.DataFrame([{
        "timestamp":     now,
        **data,
        "pressure_diff": pressure_diff,
        "hour":          ts.hour,
        "day_of_week":   ts.dayofweek,
        "month":         ts.month,
        "is_weekend":    int(ts.dayofweek in [5, 6]),
        "hour_sin":      np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos":      np.cos(2 * np.pi * ts.hour / 24),
        "month_sin":     np.sin(2 * np.pi * ts.month / 12),
        "month_cos":     np.cos(2 * np.pi * ts.month / 12),
        # real lag features from history
        **lag_feats,
        # live weather forecasts — the key new addition
        **forecast_feats,
        # targets unknown at inference time
        "target_1h":  np.nan,
        "target_24h": np.nan,
        "target_48h": np.nan,
        "target_72h": np.nan,
    }])

    df["timestamp"] = df["timestamp"].astype(str)
    store_features(df, project)
    print("=== Pipeline Complete ===")