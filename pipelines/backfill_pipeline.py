import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
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
        (150.5,250.4, 201,  300),
        (250.5,350.4, 301,  400),
        (350.5,500.4, 401,  500),
    ]
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            return round(
                ((aqi_hi-aqi_lo)/(bp_hi-bp_lo))
                * (pm25-bp_lo) + aqi_lo
            )
    return 500

def fetch_pollution_day(date, ow_token):
    start = int(date.replace(
        hour=0, minute=0, second=0,
        tzinfo=timezone.utc).timestamp())
    end = int(date.replace(
        hour=23, minute=59, second=59,
        tzinfo=timezone.utc).timestamp())
    url = (
        f"http://api.openweathermap.org/data/2.5/"
        f"air_pollution/history"
        f"?lat={LAT}&lon={LON}"
        f"&start={start}&end={end}&appid={ow_token}"
    )
    r = requests.get(url).json()
    if "list" not in r:
        return {}
    result = {}
    for entry in r["list"]:
        c = entry["components"]
        pm25 = float(c.get("pm2_5", 0))
        result[entry["dt"]] = {
            "aqi":  float(pm25_to_aqi(pm25)),
            "pm25": pm25,
            "pm10": float(c.get("pm10", 0)),
            "o3":   float(c.get("o3",   0)),
            "no2":  float(c.get("no2",  0)),
            "so2":  float(c.get("so2",  0)),
            "co":   float(c.get("co",   0)),
        }
    return result

def fetch_weather_day(date, ow_token):
    """Fetch hourly weather using One Call history API"""
    ts = int(date.replace(
        hour=12, minute=0, second=0,
        tzinfo=timezone.utc).timestamp())
    url = (
        f"https://history.openweathermap.org/data/2.5/"
        f"history/city?lat={LAT}&lon={LON}"
        f"&type=hour&start={int(date.replace(hour=0,minute=0,second=0,tzinfo=timezone.utc).timestamp())}"
        f"&end={int(date.replace(hour=23,minute=59,second=59,tzinfo=timezone.utc).timestamp())}"
        f"&appid={ow_token}&units=metric"
    )
    r = requests.get(url).json()
    result = {}
    if "list" not in r:
        return result
    for entry in r["list"]:
        result[entry["dt"]] = {
            "temp":     float(entry.get("main",{}).get("temp", 0)),
            "humidity": float(entry.get("main",{}).get("humidity", 0)),
            "wind":     float(entry.get("wind",{}).get("speed", 0)),
        }
    return result

def fetch_day(date, ow_token):
    pollution = fetch_pollution_day(date, ow_token)
    weather   = fetch_weather_day(date, ow_token)
    rows = []
    for ts, p in pollution.items():
        w = weather.get(ts, {"temp":0.0,
                              "humidity":0.0,
                              "wind":0.0})
        rows.append({
            "timestamp": datetime.fromtimestamp(
                ts, tz=timezone.utc).isoformat(),
            **p, **w
        })
    return rows

def compute_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[df["aqi"] > 5].copy()

    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = (
        df["day_of_week"].isin([5,6])).astype(int)

    df["aqi_lag1"]  = df["aqi"].shift(1)
    df["aqi_lag2"]  = df["aqi"].shift(2)
    df["aqi_lag24"] = df["aqi"].shift(24)

    df["aqi_roll6_mean"]  = df["aqi"].rolling(6).mean()
    df["aqi_roll24_mean"] = df["aqi"].rolling(24).mean()
    df["aqi_roll6_std"]   = (
        df["aqi"].rolling(6).std().fillna(0))
    df["aqi_change_rate"] = (
        df["aqi"].diff() / df["aqi"].shift(1)
    ).fillna(0)

    df["target_24h"] = df["aqi"].shift(-24)
    df["target_48h"] = df["aqi"].shift(-48)
    df["target_72h"] = df["aqi"].shift(-72)

    df = df.dropna(subset=[
        "target_24h","target_48h","target_72h",
        "aqi_lag24","aqi_roll24_mean"
    ])
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
        description="Hourly AQI + weather for Karachi",
        online_enabled=False
    )
    fg.insert(df, write_options={"wait_for_job":False})
    print(f"Stored {len(df)} rows")

if __name__ == "__main__":
    print("=== Backfill Pipeline (365 days) ===")
    ow_token = os.getenv("OPENWEATHER_API_KEY")

    all_rows = []
    today = datetime.now(timezone.utc)

    for days_ago in range(365, 0, -1):
        target_date = today - timedelta(days=days_ago)
        print(
            f"Fetching {target_date.date()} "
            f"({366-days_ago}/365)...", end=" ")
        rows = fetch_day(target_date, ow_token)
        all_rows.extend(rows)
        print(f"{len(rows)} readings")
        time.sleep(0.5)

    print(f"\nTotal raw rows: {len(all_rows)}")
    df = pd.DataFrame(all_rows)

    print(f"AQI range: {df['aqi'].min():.0f} "
          f"to {df['aqi'].max():.0f}")
    print(f"Weather filled: "
          f"{(df['temp']!=0).sum()} rows have temp")

    df = compute_features(df)
    print(f"Rows after engineering: {len(df)}")
    store_features(df)
    print("\n=== Backfill Complete ===")