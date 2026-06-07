import sys
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import time
from dotenv import load_dotenv
import hopsworks

# ── Preflight: confluent_kafka must be present ────────────────────────────
# Hopsworks 4.7 server sets stream=True on all feature groups; insert()
# always routes to Kafka regardless of write_options. confluent_kafka is
# required. Check early so the error is readable, not buried in a traceback.
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


def pm25_to_aqi(pm25):
    """Fallback AQI from PM2.5 only — used when us_aqi is unavailable."""
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


def _get_json(url, timeout=30, retries=3, backoff=5):
    """
    Fetch a URL and return parsed JSON with retry logic.
    Open-Meteo occasionally returns empty responses under load.
    """
    import time as _time
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if not resp.text.strip():
                raise ValueError("Empty response body")
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                print(f"    [retry {attempt}/{retries}] {e}")
                _time.sleep(backoff)
    raise RuntimeError(f"All {retries} attempts failed: {last_err}")


def fetch_chunk(start_date, end_date):
    """
    Fetch one chunk of air quality + weather data from Open-Meteo.
    Uses us_aqi as the primary AQI source (official EPA value).
    """
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    aq_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=pm2_5,pm10,carbon_monoxide,"
        f"nitrogen_dioxide,sulphur_dioxide,ozone,"
        f"dust,european_aqi,us_aqi"
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
        f"dew_point_2m,apparent_temperature,"
        f"shortwave_radiation"
        f"&start_date={start_str}&end_date={end_str}"
        f"&wind_speed_unit=ms&timezone=UTC"
    )

    try:
        aq_r = _get_json(aq_url)
        wx_r = _get_json(wx_url)
    except Exception as e:
        print(f"\n  Request failed: {e}")
        return []

    if "hourly" not in aq_r or "hourly" not in wx_r:
        print(f"\n  API error: "
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
    n_aq = len(aq.get("time", []))
    n_wx = len(wx.get("time", []))
    n    = min(n_aq, n_wx)  # align if lengths differ

    for i in range(n):
        pm25   = s(aq["pm2_5"], i)
        us_aqi = s(aq["us_aqi"], i)
        wdir   = s(wx["wind_direction_10m"], i)

        # Use official us_aqi; fall back to formula only if missing
        aqi_val = us_aqi if not np.isnan(us_aqi) else pm25_to_aqi(pm25)

        rows.append({
            "timestamp":     pd.to_datetime(aq["time"][i], utc=True),
            "aqi":           aqi_val,
            "pm25":          pm25,
            "pm10":          s(aq["pm10"], i),
            "o3":            s(aq["ozone"], i),
            "no2":           s(aq["nitrogen_dioxide"], i),
            "so2":           s(aq["sulphur_dioxide"], i),
            "co":            s(aq["carbon_monoxide"], i),
            "dust":          s(aq["dust"], i),
            "european_aqi":  s(aq["european_aqi"], i),
            "us_aqi":        us_aqi,
            "temp":          s(wx["temperature_2m"], i),
            "humidity":      s(wx["relative_humidity_2m"], i),
            "wind":          s(wx["wind_speed_10m"], i),
            "wind_gusts":    s(wx["wind_gusts_10m"], i),
            "precipitation": s(wx["precipitation"], i),
            "pressure":      s(wx["surface_pressure"], i),
            "cloud_cover":   s(wx["cloud_cover"], i),
            "dew_point":     s(wx["dew_point_2m"], i),
            "apparent_temp": s(wx["apparent_temperature"], i),
            "solar_rad":     s(wx["shortwave_radiation"], i),
            "wind_dir_sin":  float(np.sin(np.radians(wdir)))
                             if not np.isnan(wdir) else np.nan,
            "wind_dir_cos":  float(np.cos(np.radians(wdir)))
                             if not np.isnan(wdir) else np.nan,
        })
    return rows


def compute_features(df):
    """
    Engineer all training features from raw hourly data.
    No forecast columns here — they are inference-only and would
    cause train/serve skew if included in historical rows as NaN-filled
    medians (see module docstring for explanation).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Fill gaps in raw signals ──────────────────────────────────
    raw_cols = [
        "pm25", "pm10", "o3", "no2", "so2", "co",
        "dust", "european_aqi", "us_aqi",
        "temp", "humidity", "wind",
        "wind_gusts", "precipitation", "pressure",
        "cloud_cover", "dew_point",
        "apparent_temp", "solar_rad", "aqi",
        "wind_dir_sin", "wind_dir_cos",
    ]
    df[raw_cols] = df[raw_cols].ffill().bfill()
    df = df.dropna(subset=["aqi"])
    # Removed the aqi > 5 filter — it distorted the low end of distribution

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
    df["aqi_lag1"]   = df["aqi"].shift(1)
    df["aqi_lag2"]   = df["aqi"].shift(2)
    df["aqi_lag3"]   = df["aqi"].shift(3)
    df["aqi_lag6"]   = df["aqi"].shift(6)
    df["aqi_lag12"]  = df["aqi"].shift(12)
    df["aqi_lag24"]  = df["aqi"].shift(24)
    df["aqi_lag48"]  = df["aqi"].shift(48)
    # Extended lags for 48h/72h models:
    # lag72  = 3-day lag; lag96/120 = 4/5-day; lag168 = same hour last week
    # Weekly traffic patterns in Karachi make lag168 physically meaningful.
    df["aqi_lag72"]  = df["aqi"].shift(72)
    df["aqi_lag96"]  = df["aqi"].shift(96)
    df["aqi_lag120"] = df["aqi"].shift(120)
    df["aqi_lag168"] = df["aqi"].shift(168)

    # ── Rolling features (min_periods=1 suppresses empty-slice warnings) ─
    df["aqi_roll3_mean"]  = df["aqi"].rolling(3,   min_periods=1).mean()
    df["aqi_roll6_mean"]  = df["aqi"].rolling(6,   min_periods=1).mean()
    df["aqi_roll12_mean"] = df["aqi"].rolling(12,  min_periods=1).mean()
    df["aqi_roll24_mean"] = df["aqi"].rolling(24,  min_periods=1).mean()
    # Longer rolling windows — 2-day and 3-day means give the 48h/72h models
    # a smoother baseline to detect mean-reversion (high roll mean → likely to fall)
    df["aqi_roll48_mean"] = df["aqi"].rolling(48,  min_periods=24).mean()
    df["aqi_roll72_mean"] = df["aqi"].rolling(72,  min_periods=24).mean()
    df["aqi_roll6_std"]   = df["aqi"].rolling(6,   min_periods=2).std()
    df["aqi_roll24_std"]  = df["aqi"].rolling(24,  min_periods=2).std()

    # ── Diff / rate features ──────────────────────────────────────
    df["aqi_change_rate"] = df["aqi"].diff() / df["aqi"].shift(1).replace(0, np.nan)
    df["aqi_diff1"]       = df["aqi"].diff(1)
    df["aqi_diff6"]       = df["aqi"].diff(6)
    df["aqi_diff24"]      = df["aqi"].diff(24)

    # ── PM2.5 lags ────────────────────────────────────────────────
    df["pm25_lag1"]       = df["pm25"].shift(1)
    df["pm25_lag24"]      = df["pm25"].shift(24)
    df["pm25_roll6_mean"] = df["pm25"].rolling(6, min_periods=1).mean()

    # ── Derived weather ───────────────────────────────────────────
    df["temp_humidity"]  = df["temp"] * df["humidity"] / 100
    df["pressure_diff"]  = df["pressure"].diff(1)
    df["pm25_wind"]      = df["pm25"] / (df["wind"] + 0.1)
    df["dew_depression"] = df["temp"] - df["dew_point"]

    # ── Targets ───────────────────────────────────────────────────
    df["target_1h"]  = df["aqi"].shift(-1)
    df["target_24h"] = df["aqi"].shift(-24)
    df["target_48h"] = df["aqi"].shift(-48)
    df["target_72h"] = df["aqi"].shift(-72)

    # Drop rows where we can't have all targets or the deepest lag.
    # lag168 is now the binding constraint at the start (168 rows lost),
    # target_72h at the end (72 rows lost). Total: ~240 rows → ~1944 output.
    df = df.dropna(subset=[
        "target_24h", "target_48h", "target_72h",
        "aqi_lag168", "aqi_roll24_mean", "pm25_lag24"
    ])

    # Fill remaining NaN in feature columns with median
    # (affects early-window lags, aqi_roll6_std at start, pressure_diff row 0)
    skip_fill = {
        "timestamp",
        "target_1h", "target_24h", "target_48h", "target_72h"
    }
    for col in df.columns:
        if col in skip_fill:
            continue
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(0.0 if pd.isna(med) else med)

    nan_count = df.isna().sum().sum()
    print(f"NaN remaining after fill: {nan_count}")

    # ── Types ─────────────────────────────────────────────────────
    df["aqi"]         = df["aqi"].round().astype("int64")
    df["hour"]        = df["hour"].astype("int64")
    df["day_of_week"] = df["day_of_week"].astype("int64")
    df["month"]       = df["month"].astype("int64")
    df["is_weekend"]  = df["is_weekend"].astype("int64")
    # Keep timestamp as datetime64[ns, UTC] — Hopsworks event_time requires
    # a proper TIMESTAMP type, not a string. Do NOT cast to str here.
    df["timestamp"]   = pd.to_datetime(df["timestamp"], utc=True)
    return df


def store_features(df):
    """
    Write features to MongoDB Atlas (replaces Hopsworks Feature Store).
    Uses upsert on timestamp — safe to re-run.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mongo_store import store_df

    print("Connecting to MongoDB Atlas...")
    store_df(df)
    print(f"Upserted {len(df)} rows × {len(df.columns)} columns → MongoDB")


if __name__ == "__main__":
    print("=== Backfill Pipeline v6 ===\n")

    # ── 180-day backfill window ───────────────────────────────────────
    today      = datetime.now(timezone.utc)
    start      = today - timedelta(days=180)
    chunk_days = 30

    print(f"Backfill window: {start.date()} → {today.date()} (180 days)\n")

    all_rows = []
    cur = start
    while cur < today:
        end = min(cur + timedelta(days=chunk_days), today)
        print(f"Fetching {cur.date()} → {end.date()}...", end=" ", flush=True)
        rows = fetch_chunk(cur, end)
        all_rows.extend(rows)
        print(f"{len(rows)} readings")
        cur = end + timedelta(days=1)
        time.sleep(1)

    print(f"\nTotal raw rows: {len(all_rows)}")
    if not all_rows:
        print("No data fetched — aborting.")
        raise SystemExit(1)

    df = pd.DataFrame(all_rows)

    print(f"\nData quality check:")
    print(f"AQI missing:   {df['aqi'].isna().sum()}/{len(df)}")
    print(f"PM2.5 missing: {df['pm25'].isna().sum()}/{len(df)}")
    print(f"Temp missing:  {df['temp'].isna().sum()}/{len(df)}")
    aqi_clean = df["aqi"].dropna()
    print(f"AQI range:     {aqi_clean.min():.0f} to {aqi_clean.max():.0f}")

    df = compute_features(df)

    print(f"\nAfter feature engineering:")
    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Columns: {sorted(df.columns.tolist())}")
    aqi_clean2 = df["aqi"]
    print(f"AQI range: {aqi_clean2.min()} to {aqi_clean2.max()}")
    print(f"Any NaN remaining: {df.isna().any().any()}")

    # Save locally BEFORE inserting to Hopsworks.
    # This is an emergency fallback: if the Hopsworks offline store is empty
    # due to a stuck materialization job, training_pipeline.py can load from
    # this CSV directly instead of waiting for Hopsworks to recover.
    os.makedirs("data", exist_ok=True)
    local_path = "data/backfill_local.csv"
    df.to_csv(local_path, index=False)
    print(f"\nSaved locally to {local_path} (emergency fallback for training)")

    store_features(df)
    print("\n=== Backfill Complete ===")