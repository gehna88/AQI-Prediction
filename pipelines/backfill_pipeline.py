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
        f"dust,ammonia,european_aqi,us_aqi"
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
        f"visibility,dew_point_2m,"
        f"apparent_temperature,"
        f"soil_temperature_0cm,"
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
        rows.append({
            "timestamp":    ts_str + ":00+00:00",
            "aqi":          pm25_to_aqi(pm25),
            "pm25":         pm25,
            "pm10":         s(aq["pm10"], i),
            "o3":           s(aq["ozone"], i),
            "no2":          s(aq["nitrogen_dioxide"], i),
            "so2":          s(aq["sulphur_dioxide"], i),
            "co":           s(aq["carbon_monoxide"], i),
            "dust":         s(aq["dust"], i),
            "ammonia":      s(aq["ammonia"], i),
            "european_aqi": s(aq["european_aqi"], i),
            "us_aqi":       s(aq["us_aqi"], i),
            "temp":         s(wx["temperature_2m"], i),
            "humidity":     s(wx["relative_humidity_2m"], i),
            "wind":         s(wx["wind_speed_10m"], i),
            "wind_dir":     s(wx["wind_direction_10m"], i),
            "wind_gusts":   s(wx["wind_gusts_10m"], i),
            "precipitation":s(wx["precipitation"], i),
            "pressure":     s(wx["surface_pressure"], i),
            "cloud_cover":  s(wx["cloud_cover"], i),
            "visibility":   s(wx["visibility"], i),
            "dew_point":    s(wx["dew_point_2m"], i),
            "apparent_temp":s(wx["apparent_temperature"], i),
            "soil_temp":    s(wx["soil_temperature_0cm"], i),
            "solar_rad":    s(wx["shortwave_radiation"], i),
        })
    return rows

def compute_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── forward fill then backfill missing values ────
    # this is correct — use actual nearby values
    # instead of zeroing out missing data
    pollutant_cols = [
        "pm25","pm10","o3","no2","so2","co",
        "dust","ammonia","european_aqi","us_aqi",
        "temp","humidity","wind","wind_dir",
        "wind_gusts","precipitation","pressure",
        "cloud_cover","visibility","dew_point",
        "apparent_temp","soil_temp","solar_rad","aqi"
    ]
    df[pollutant_cols] = (
        df[pollutant_cols]
        .ffill()   # forward fill first
        .bfill()   # then backward fill any remaining
    )

    # remove any rows still missing AQI after fill
    df = df.dropna(subset=["aqi"])
    df = df[df["aqi"] > 5].copy()

    # ── Time features ────────────────────────────────
    df["hour"]          = df["timestamp"].dt.hour
    df["day_of_week"]   = df["timestamp"].dt.dayofweek
    df["month"]         = df["timestamp"].dt.month
    df["is_weekend"]    = (
        df["day_of_week"].isin([5, 6])).astype(int)
    df["hour_sin"]      = np.sin(
        2 * np.pi * df["hour"] / 24)
    df["hour_cos"]      = np.cos(
        2 * np.pi * df["hour"] / 24)
    df["month_sin"]     = np.sin(
        2 * np.pi * df["month"] / 12)
    df["month_cos"]     = np.cos(
        2 * np.pi * df["month"] / 12)

    # ── AQI lag features ─────────────────────────────
    df["aqi_lag1"]      = df["aqi"].shift(1)
    df["aqi_lag2"]      = df["aqi"].shift(2)
    df["aqi_lag3"]      = df["aqi"].shift(3)
    df["aqi_lag6"]      = df["aqi"].shift(6)
    df["aqi_lag12"]     = df["aqi"].shift(12)
    df["aqi_lag24"]     = df["aqi"].shift(24)
    df["aqi_lag48"]     = df["aqi"].shift(48)

    # ── Rolling stats ────────────────────────────────
    df["aqi_roll3_mean"]  = df["aqi"].rolling(3).mean()
    df["aqi_roll6_mean"]  = df["aqi"].rolling(6).mean()
    df["aqi_roll12_mean"] = df["aqi"].rolling(12).mean()
    df["aqi_roll24_mean"] = df["aqi"].rolling(24).mean()
    df["aqi_roll6_std"]   = df["aqi"].rolling(6).std()
    df["aqi_roll24_std"]  = df["aqi"].rolling(24).std()
    df["aqi_roll24_max"]  = df["aqi"].rolling(24).max()
    df["aqi_roll24_min"]  = df["aqi"].rolling(24).min()

    # ── AQI change features ──────────────────────────
    df["aqi_change_rate"] = (
        df["aqi"].diff() / df["aqi"].shift(1))
    df["aqi_diff1"]       = df["aqi"].diff(1)
    df["aqi_diff6"]       = df["aqi"].diff(6)
    df["aqi_diff24"]      = df["aqi"].diff(24)

    # ── PM2.5 lags ───────────────────────────────────
    df["pm25_lag1"]       = df["pm25"].shift(1)
    df["pm25_lag24"]      = df["pm25"].shift(24)
    df["pm25_roll6_mean"] = df["pm25"].rolling(6).mean()

    # ── Derived weather ──────────────────────────────
    df["temp_humidity"]   = df["temp"] * df["humidity"] / 100
    df["wind_chill"]      = df["temp"] - df["wind"] * 0.7
    df["pressure_diff"]   = df["pressure"].diff(1)
    df["wind_dir_sin"]    = np.sin(np.radians(df["wind_dir"]))
    df["wind_dir_cos"]    = np.cos(np.radians(df["wind_dir"]))
    df["pm25_wind"]       = df["pm25"] / (df["wind"] + 0.1)
    df["dew_depression"]  = df["temp"] - df["dew_point"]
    df["is_daytime"]      = (df["solar_rad"] > 10).astype(int)

    # ── Targets — 3 separate targets ─────────────────
    df["target_24h"] = df["aqi"].shift(-24)
    df["target_48h"] = df["aqi"].shift(-48)
    df["target_72h"] = df["aqi"].shift(-72)

# drop rows missing targets or critical lag features
    df = df.dropna(subset=[
        "target_24h", "target_48h", "target_72h",
        "aqi_lag48", "aqi_roll24_mean", "pm25_lag24"
    ])

    # fill ALL remaining NaN with median
    # (handles rolling window NaN at start of series)
    for col in df.columns:
        if col in ["timestamp",
                   "target_24h", "target_48h",
                   "target_72h"]:
            continue
        if df[col].isna().any():
            median_val = df[col].median()
            if np.isnan(median_val):
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna(median_val)

    print(f"NaN remaining after fill: "
          f"{df.isna().sum().sum()}")

    df["timestamp"] = df["timestamp"].astype(str)
    return df

def store_features(df):
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    fs = project.get_feature_store()

    try:
        fs.get_feature_group(
            "aqi_features", version=1).delete()
        print("Deleted old feature group")
    except Exception:
        pass

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="Hourly AQI 58 features Karachi "
                    "Open-Meteo no API key",
        online_enabled=False
    )
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"Stored {len(df)} rows, "
          f"{len(df.columns)} columns")

if __name__ == "__main__":
    print("=== Backfill Pipeline (58 features) ===\n")

    all_rows   = []
    today      = datetime.now(timezone.utc)
    chunk_days = 30
    start      = today - timedelta(days=365)

    while start < today:
        end = min(
            start + timedelta(days=chunk_days), today)
        print(f"Fetching {start.date()} → "
              f"{end.date()}...", end=" ", flush=True)
        rows = fetch_chunk(start, end)
        all_rows.extend(rows)
        print(f"{len(rows)} readings")
        start = end + timedelta(days=1)
        time.sleep(1)

    print(f"\nTotal raw rows: {len(all_rows)}")
    df = pd.DataFrame(all_rows)

    # check data quality before processing
    print(f"\nData quality check:")
    print(f"AQI missing:  "
          f"{df['aqi'].isna().sum()}/{len(df)}")
    print(f"PM2.5 missing: "
          f"{df['pm25'].isna().sum()}/{len(df)}")
    print(f"Temp missing:  "
          f"{df['temp'].isna().sum()}/{len(df)}")
    print(f"AQI range (before fill): "
          f"{df['aqi'].min():.0f} to "
          f"{df['aqi'].max():.0f}")

    df = compute_features(df)

    print(f"\nAfter engineering:")
    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"AQI range: {df['aqi'].min():.0f} "
          f"to {df['aqi'].max():.0f}")
    print(f"Temp range: {df['temp'].min():.1f}°C "
          f"to {df['temp'].max():.1f}°C")
    print(f"Any NaN remaining: "
          f"{df.isna().any().any()}")

    store_features(df)
    print("\n=== Backfill Complete ===")