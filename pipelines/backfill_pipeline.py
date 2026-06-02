"""
backfill_pipeline.py  –  v4
Changes vs v3:
  1. DROPPED zero-importance features:
       soil_temp, ammonia, visibility, is_daytime,
       wind_chill, wind_dir (raw degrees),
       aqi_roll24_max, aqi_roll24_min
     These scored 0.000 importance across all four horizons.
     wind_dir_sin and wind_dir_cos are kept (they encode direction
     without the circular ambiguity of raw degrees).

  2. ADDED weather forecast features:
       temp_forecast_24h/48h/72h
       wind_forecast_24h/48h/72h
       precip_forecast_24h/48h/72h
       pressure_forecast_24h/48h/72h
     Historical data has no forecasts (they were never stored), so these
     are filled with NaN for all backfill rows.  The training pipeline
     fills NaN with column median at fit time, so models train fine.
     At inference (feature_pipeline.py), live Open-Meteo forecast values
     are used — giving 48h/72h models genuine future weather signals.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import time
from dotenv import load_dotenv
import hopsworks

load_dotenv()

LAT = 24.8607
LON = 67.0011


def pm25_to_aqi(pm25):
    if np.isnan(pm25):
        return np.nan
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


def fetch_chunk(start_date, end_date):
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    aq_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=pm2_5,pm10,carbon_monoxide,"
        f"nitrogen_dioxide,sulphur_dioxide,ozone,"
        f"dust,european_aqi,us_aqi"
        # ammonia removed — zero importance
        f"&start_date={start_str}&end_date={end_str}"
        f"&timezone=UTC"
    )
    wx_url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relative_humidity_2m,"
        f"wind_speed_10m,wind_direction_10m,"
        f"wind_gusts_10m,precipitation,"
        f"surface_pressure,cloud_cover,"
        # visibility, soil_temperature_0cm removed — zero importance
        f"dew_point_2m,"
        f"apparent_temperature,"
        f"shortwave_radiation"
        f"&start_date={start_str}&end_date={end_str}"
        f"&wind_speed_unit=ms&timezone=UTC"
    )

    aq_r = requests.get(aq_url, timeout=30).json()
    wx_r = requests.get(wx_url, timeout=30).json()

    if "hourly" not in aq_r or "hourly" not in wx_r:
        print(f"\n  Failed: "
              f"AQ={aq_r.get('reason','?')} "
              f"WX={wx_r.get('reason','?')}")
        return []

    aq = aq_r["hourly"]
    wx = wx_r["hourly"]

    def s(lst, i):
        try:
            v = lst[i]
            return float(v) if v is not None else np.nan
        except Exception:
            return np.nan

    rows = []
    for i, ts_str in enumerate(aq["time"]):
        pm25 = s(aq["pm2_5"], i)
        wdir = s(wx["wind_direction_10m"], i)
        rows.append({
            "timestamp":     pd.to_datetime(ts_str, utc=True),
            "aqi":           pm25_to_aqi(pm25),
            "pm25":          pm25,
            "pm10":          s(aq["pm10"], i),
            "o3":            s(aq["ozone"], i),
            "no2":           s(aq["nitrogen_dioxide"], i),
            "so2":           s(aq["sulphur_dioxide"], i),
            "co":            s(aq["carbon_monoxide"], i),
            "dust":          s(aq["dust"], i),
            # ammonia dropped
            "european_aqi":  s(aq["european_aqi"], i),
            "us_aqi":        s(aq["us_aqi"], i),
            "temp":          s(wx["temperature_2m"], i),
            "humidity":      s(wx["relative_humidity_2m"], i),
            "wind":          s(wx["wind_speed_10m"], i),
            # wind_dir raw degrees dropped; sin/cos computed below
            "wind_gusts":    s(wx["wind_gusts_10m"], i),
            "precipitation": s(wx["precipitation"], i),
            "pressure":      s(wx["surface_pressure"], i),
            "cloud_cover":   s(wx["cloud_cover"], i),
            # visibility dropped
            "dew_point":     s(wx["dew_point_2m"], i),
            "apparent_temp": s(wx["apparent_temperature"], i),
            # soil_temp dropped
            "solar_rad":     s(wx["shortwave_radiation"], i),
            # wind direction encoded as sin/cos only
            "wind_dir_sin":  float(np.sin(np.radians(wdir)))
                             if not np.isnan(wdir) else np.nan,
            "wind_dir_cos":  float(np.cos(np.radians(wdir)))
                             if not np.isnan(wdir) else np.nan,
            # forecast features — NaN for historical rows
            # (filled at training time with column median)
            "temp_forecast_24h":     np.nan,
            "wind_forecast_24h":     np.nan,
            "precip_forecast_24h":   np.nan,
            "pressure_forecast_24h": np.nan,
            "temp_forecast_48h":     np.nan,
            "wind_forecast_48h":     np.nan,
            "precip_forecast_48h":   np.nan,
            "pressure_forecast_48h": np.nan,
            "temp_forecast_72h":     np.nan,
            "wind_forecast_72h":     np.nan,
            "precip_forecast_72h":   np.nan,
            "pressure_forecast_72h": np.nan,
        })
    return rows


def compute_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    pollutant_cols = [
        "pm25", "pm10", "o3", "no2", "so2", "co",
        "dust", "european_aqi", "us_aqi",
        "temp", "humidity", "wind",
        "wind_gusts", "precipitation", "pressure",
        "cloud_cover", "dew_point",
        "apparent_temp", "solar_rad", "aqi",
        "wind_dir_sin", "wind_dir_cos",
    ]
    df[pollutant_cols] = df[pollutant_cols].ffill().bfill()

    df = df.dropna(subset=["aqi"])
    df = df[df["aqi"] > 5].copy()

    # ── Time features ─────────────────────────────────────────────
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

    # ── AQI lag features ──────────────────────────────────────────
    df["aqi_lag1"]  = df["aqi"].shift(1)
    df["aqi_lag2"]  = df["aqi"].shift(2)
    df["aqi_lag3"]  = df["aqi"].shift(3)
    df["aqi_lag6"]  = df["aqi"].shift(6)
    df["aqi_lag12"] = df["aqi"].shift(12)
    df["aqi_lag24"] = df["aqi"].shift(24)
    df["aqi_lag48"] = df["aqi"].shift(48)

    # ── Rolling features (min/max dropped — redundant with mean+std) ─
    df["aqi_roll3_mean"]  = df["aqi"].rolling(3).mean()
    df["aqi_roll6_mean"]  = df["aqi"].rolling(6).mean()
    df["aqi_roll12_mean"] = df["aqi"].rolling(12).mean()
    df["aqi_roll24_mean"] = df["aqi"].rolling(24).mean()
    df["aqi_roll6_std"]   = df["aqi"].rolling(6).std()
    df["aqi_roll24_std"]  = df["aqi"].rolling(24).std()
    # aqi_roll24_max and aqi_roll24_min dropped

    # ── Diff / rate features ──────────────────────────────────────
    df["aqi_change_rate"] = df["aqi"].diff() / df["aqi"].shift(1)
    df["aqi_diff1"]       = df["aqi"].diff(1)
    df["aqi_diff6"]       = df["aqi"].diff(6)
    df["aqi_diff24"]      = df["aqi"].diff(24)

    # ── PM2.5 lags ────────────────────────────────────────────────
    df["pm25_lag1"]       = df["pm25"].shift(1)
    df["pm25_lag24"]      = df["pm25"].shift(24)
    df["pm25_roll6_mean"] = df["pm25"].rolling(6).mean()

    # ── Derived weather ───────────────────────────────────────────
    # wind_chill dropped (linear combo of temp+wind already present)
    df["temp_humidity"]  = df["temp"] * df["humidity"] / 100
    df["pressure_diff"]  = df["pressure"].diff(1)
    df["pm25_wind"]      = df["pm25"] / (df["wind"] + 0.1)
    df["dew_depression"] = df["temp"] - df["dew_point"]
    # is_daytime dropped (zero importance)

    # ── Targets ───────────────────────────────────────────────────
    df["target_1h"]  = df["aqi"].shift(-1)
    df["target_24h"] = df["aqi"].shift(-24)
    df["target_48h"] = df["aqi"].shift(-48)
    df["target_72h"] = df["aqi"].shift(-72)

    # require all three targets + key lag features
    df = df.dropna(subset=[
        "target_24h", "target_48h", "target_72h",
        "aqi_lag48", "aqi_roll24_mean", "pm25_lag24"
    ])

    # fill remaining NaN (forecast cols + early-window lags) with median
    skip_fill = {"timestamp",
                 "target_1h", "target_24h",
                 "target_48h", "target_72h"}
    for col in df.columns:
        if col in skip_fill:
            continue
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(0.0 if np.isnan(med) else med)

    print(f"NaN remaining after fill: {df.isna().sum().sum()}")

    # ── Types ─────────────────────────────────────────────────────
    df["aqi"]         = df["aqi"].round().astype("int64")
    df["hour"]        = df["hour"].astype("int64")
    df["day_of_week"] = df["day_of_week"].astype("int64")
    df["month"]       = df["month"].astype("int64")
    df["is_weekend"]  = df["is_weekend"].astype("int64")
    df["timestamp"]   = df["timestamp"].astype(str)
    return df


def store_features(df):
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    fs = project.get_feature_store()

    try:
        fs.get_feature_group("aqi_features", version=1).delete()
        print("Deleted old feature group")
    except Exception:
        pass

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
    print("=== Backfill Pipeline v4 ===\n")

    all_rows   = []
    today      = datetime.now(timezone.utc)
    chunk_days = 30
    start      = today - timedelta(days=365)

    while start < today:
        end = min(start + timedelta(days=chunk_days), today)
        print(f"Fetching {start.date()} → "
              f"{end.date()}...", end=" ", flush=True)
        rows = fetch_chunk(start, end)
        all_rows.extend(rows)
        print(f"{len(rows)} readings")
        start = end + timedelta(days=1)
        time.sleep(1)

    print(f"\nTotal raw rows: {len(all_rows)}")
    df = pd.DataFrame(all_rows)

    print(f"\nData quality check:")
    print(f"AQI missing:   {df['aqi'].isna().sum()}/{len(df)}")
    print(f"PM2.5 missing: {df['pm25'].isna().sum()}/{len(df)}")
    print(f"Temp missing:  {df['temp'].isna().sum()}/{len(df)}")
    print(f"AQI range: {df['aqi'].min():.0f} to {df['aqi'].max():.0f}")

    df = compute_features(df)

    print(f"\nAfter engineering:")
    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"AQI range: {df['aqi'].min():.0f} to {df['aqi'].max():.0f}")
    print(f"Any NaN remaining: {df.isna().any().any()}")

    store_features(df)
    print("\n=== Backfill Complete ===")