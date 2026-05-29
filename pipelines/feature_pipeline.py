import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # reads your .env file automatically

# ── STEP 1: FETCH RAW DATA FROM AQICN ──────────────
def fetch_aqi_data():
    token = os.getenv("AQICN_TOKEN")
    url = f"https://api.waqi.info/feed/karachi/?token={token}"
    
    response = requests.get(url)
    data = response.json()
    
    if data["status"] != "ok":
        raise Exception(f"API error: {data}")
    
    d = data["data"]
    iaqi = d.get("iaqi", {})  # individual air quality index values
    
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "aqi": float(d["aqi"]),
        "pm25": float(iaqi.get("pm25", {}).get("v", None) or 0),
        "pm10": float(iaqi.get("pm10", {}).get("v", None) or 0),
        "o3": float(iaqi.get("o3", {}).get("v", None) or 0),
        "no2": float(iaqi.get("no2", {}).get("v", None) or 0),
        "so2": float(iaqi.get("so2", {}).get("v", None) or 0),
        "co": float(iaqi.get("co", {}).get("v", None) or 0),
        "temp": float(iaqi.get("t", {}).get("v", None) or 0),
        "humidity": float(iaqi.get("h", {}).get("v", None) or 0),
        "wind": float(iaqi.get("w", {}).get("v", None) or 0),
    }
    
    print(f"Fetched: AQI={row['aqi']} at {row['timestamp']}")
    return row


#if __name__ == "__main__":
  #  row = fetch_aqi_data()
   # print(row)

# ── STEP 2: COMPUTE FEATURES ──────────────────────
def compute_features(df):
    """
    Takes a DataFrame of raw readings and adds
    all the computed columns the model needs.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Time features — tells the model WHEN the reading is
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Mon, 6=Sun
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Lag features — tells the model what AQI was recently
    # (shift(1) means "the previous row")
    df["aqi_lag1"]  = df["aqi"].shift(1)   # 1 hour ago
    df["aqi_lag2"]  = df["aqi"].shift(2)   # 2 hours ago
    df["aqi_lag24"] = df["aqi"].shift(24)  # 24 hours ago (same time yesterday)
    
    # Rolling stats — average/variability over recent window
    df["aqi_roll6_mean"]  = df["aqi"].rolling(window=6).mean()
    df["aqi_roll24_mean"] = df["aqi"].rolling(window=24).mean()
    df["aqi_roll6_std"]   = df["aqi"].rolling(window=6).std()
    
    # AQI change rate — how fast is air quality changing?
    df["aqi_change_rate"] = df["aqi"].diff() / df["aqi"].shift(1)
    
    # Target variable — what we WANT to predict
    # shift(-24) means "the value 24 rows in the future"
    df["target_24h"] = df["aqi"].shift(-24)  # AQI 24 hours from now
    df["target_48h"] = df["aqi"].shift(-48)
    df["target_72h"] = df["aqi"].shift(-72)
    
    # Drop rows that have NaN (empty) values — they appear
    # at the start (not enough history yet) and end (no future data yet)
    df = df.dropna()
    
    print(f"Feature engineering done. Rows: {len(df)}, Columns: {len(df.columns)}")
    return df

import hopsworks

# ── STEP 3: STORE IN HOPSWORKS FEATURE STORE ──────
def store_features(df):
    """
    Connects to Hopsworks and inserts the DataFrame
    into a Feature Group (think: a database table).
    Creates the table automatically on first run.
    """
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    
    fs = project.get_feature_store()  # get the data storage area
    
    # A Feature Group is like a table in a database
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],          # unique ID for each row
        description="Hourly AQI features for Karachi",
        online_enabled=True                 # allows fast real-time reads
    )
    
    # Convert timestamp column to the right type
    df["timestamp"] = df["timestamp"].astype(str)
    
    # Insert! Hopsworks handles duplicates automatically
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"Stored {len(df)} rows in Hopsworks feature store")


# ── MAIN: wire it all together ────────────────────
if __name__ == "__main__":
    print("=== AQI Feature Pipeline Starting ===")
    
    raw = fetch_aqi_data()
    df  = pd.DataFrame([raw])
    
    df["timestamp"]       = pd.to_datetime(df["timestamp"])
    df["hour"]            = df["timestamp"].dt.hour
    df["day_of_week"]     = df["timestamp"].dt.dayofweek
    df["month"]           = df["timestamp"].dt.month
    df["is_weekend"]      = df["day_of_week"].isin([5, 6]).astype(int)
    df["aqi_lag1"]        = 0.0
    df["aqi_lag2"]        = 0.0
    df["aqi_lag24"]       = 0.0
    df["aqi_roll6_mean"]  = 0.0
    df["aqi_roll24_mean"] = 0.0
    df["aqi_roll6_std"]   = 0.0
    df["aqi_change_rate"] = 0.0
    df["target_24h"]      = 0.0
    df["target_48h"]      = 0.0
    df["target_72h"]      = 0.0
    df["timestamp"]       = df["timestamp"].astype(str)
    
    store_features(df)
    
    print("=== Pipeline Complete ===")
    print("Check eu-west.cloud.hopsworks.ai → your project → Feature Store → aqi_features")