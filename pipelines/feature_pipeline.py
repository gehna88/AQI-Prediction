import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
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
    aq_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        f"&current=pm2_5,pm10,carbon_monoxide,"
        f"nitrogen_dioxide,sulphur_dioxide,ozone,"
        f"dust,ammonia,european_aqi,us_aqi"
    )
    wx_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&current=temperature_2m,relative_humidity_2m,"
        f"wind_speed_10m,wind_direction_10m,"
        f"wind_gusts_10m,precipitation,"
        f"surface_pressure,cloud_cover,"
        f"apparent_temperature,dew_point_2m,"
        f"shortwave_radiation,visibility"
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

    # use NaN instead of 0 for missing values
    aqi_val = float(pm25_to_aqi(pm25)) \
              if not np.isnan(pm25) else np.nan

    return {
        "aqi":          aqi_val,
        "pm25":         pm25,
        "pm10":         s(aq, "pm10"),
        "o3":           s(aq, "ozone"),
        "no2":          s(aq, "nitrogen_dioxide"),
        "so2":          s(aq, "sulphur_dioxide"),
        "co":           s(aq, "carbon_monoxide"),
        "dust":         s(aq, "dust"),
        "ammonia":      s(aq, "ammonia"),
        "european_aqi": s(aq, "european_aqi"),
        "us_aqi":       s(aq, "us_aqi"),
        "temp":         temp,
        "humidity":     hum,
        "wind":         wind,
        "wind_dir":     wdir,
        "wind_gusts":   s(wx, "wind_gusts_10m"),
        "precipitation":s(wx, "precipitation"),
        "pressure":     pres,
        "cloud_cover":  s(wx, "cloud_cover"),
        "visibility":   s(wx, "visibility"),
        "dew_point":    dew,
        "apparent_temp":s(wx, "apparent_temperature"),
        "soil_temp":    np.nan,
        "solar_rad":    sol,
        # derived
        "temp_humidity":  temp * hum / 100
                          if not (np.isnan(temp)
                          or np.isnan(hum)) else np.nan,
        "wind_chill":     temp - wind * 0.7
                          if not (np.isnan(temp)
                          or np.isnan(wind)) else np.nan,
        "wind_dir_sin":   float(np.sin(np.radians(wdir)))
                          if not np.isnan(wdir) else np.nan,
        "wind_dir_cos":   float(np.cos(np.radians(wdir)))
                          if not np.isnan(wdir) else np.nan,
        "pm25_wind":      pm25 / (wind + 0.1)
                          if not (np.isnan(pm25)
                          or np.isnan(wind)) else np.nan,
        "dew_depression": temp - dew
                          if not (np.isnan(temp)
                          or np.isnan(dew)) else np.nan,
        "is_daytime":     int(sol > 10)
                          if not np.isnan(sol) else 0,
        "pressure_diff":  np.nan,
    }
def store_features(df):
    # fix column types to match Hopsworks schema
    df["aqi"]         = df["aqi"].astype(float)
    df["hour"]        = df["hour"].astype(int)
    df["day_of_week"] = df["day_of_week"].astype(int)
    df["month"]       = df["month"].astype(int)
    df["is_weekend"]  = df["is_weekend"].astype(int)
    df["is_daytime"]  = df["is_daytime"].astype(int)

    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="Hourly AQI 58 features Karachi",
        online_enabled=False
    )
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"Stored {len(df)} rows, "
          f"{len(df.columns)} columns")
    

if __name__ == "__main__":
    print("=== AQI Feature Pipeline ===")
    data = fetch_current()
    now  = datetime.now(timezone.utc).isoformat()
    ts   = pd.Timestamp(now)

    print(f"AQI={data['aqi']}  "
          f"PM2.5={data['pm25']}  "
          f"Temp={data['temp']}°C  "
          f"Humidity={data['humidity']}%  "
          f"Wind={data['wind']}m/s")

    df = pd.DataFrame([{
        "timestamp":       now,
        **data,
        "hour":            ts.hour,
        "day_of_week":     ts.dayofweek,
        "month":           ts.month,
        "is_weekend":      int(ts.dayofweek in [5, 6]),
        "hour_sin":        np.sin(2*np.pi*ts.hour/24),
        "hour_cos":        np.cos(2*np.pi*ts.hour/24),
        "month_sin":       np.sin(2*np.pi*ts.month/12),
        "month_cos":       np.cos(2*np.pi*ts.month/12),
        "aqi_lag1":        np.nan,
        "aqi_lag2":        np.nan,
        "aqi_lag3":        np.nan,
        "aqi_lag6":        np.nan,
        "aqi_lag12":       np.nan,
        "aqi_lag24":       np.nan,
        "aqi_lag48":       np.nan,
        "aqi_roll3_mean":  np.nan,
        "aqi_roll6_mean":  np.nan,
        "aqi_roll12_mean": np.nan,
        "aqi_roll24_mean": np.nan,
        "aqi_roll6_std":   np.nan,
        "aqi_roll24_std":  np.nan,
        "aqi_roll24_max":  np.nan,
        "aqi_roll24_min":  np.nan,
        "aqi_change_rate": np.nan,
        "aqi_diff1":       np.nan,
        "aqi_diff6":       np.nan,
        "aqi_diff24":      np.nan,
        "pm25_lag1":       np.nan,
        "pm25_lag24":      np.nan,
        "pm25_roll6_mean": np.nan,
        "target_24h":      np.nan,
        "target_48h":      np.nan,
        "target_72h":      np.nan,
    }])

    # forward fill NaN from Hopsworks history
    # (Hopsworks will handle this at read time)
    df["timestamp"] = df["timestamp"].astype(str)
    store_features(df)
    print("=== Pipeline Complete ===")