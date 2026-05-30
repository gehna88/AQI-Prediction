import os
import numpy as np
import pandas as pd
import joblib
import hopsworks
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             r2_score)
from xgboost import XGBRegressor

load_dotenv()

FEATURES = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "dust", "ammonia", "european_aqi", "us_aqi",
    "temp", "humidity", "wind", "wind_dir",
    "wind_gusts", "precipitation", "pressure",
    "cloud_cover", "visibility", "dew_point",
    "apparent_temp", "soil_temp", "solar_rad",
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "aqi_lag1", "aqi_lag2", "aqi_lag3",
    "aqi_lag6", "aqi_lag12", "aqi_lag24", "aqi_lag48",
    "aqi_roll3_mean", "aqi_roll6_mean",
    "aqi_roll12_mean", "aqi_roll24_mean",
    "aqi_roll6_std", "aqi_roll24_std",
    "aqi_roll24_max", "aqi_roll24_min",
    "aqi_change_rate", "aqi_diff1",
    "aqi_diff6", "aqi_diff24",
    "pm25_lag1", "pm25_lag24", "pm25_roll6_mean",
    "temp_humidity", "wind_chill", "pressure_diff",
    "wind_dir_sin", "wind_dir_cos",
    "pm25_wind", "dew_depression", "is_daytime",
]

TARGETS = ["target_24h", "target_48h", "target_72h"]

def load_features():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    fs  = project.get_feature_store()
    fg  = fs.get_feature_group("aqi_features", version=1)
    df  = fg.read()
    print(f"Loaded {len(df)} rows, "
          f"{len(df.columns)} columns")
    return df, project

def prepare_data(df, target_col):
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES
                 if f not in df.columns]
    if missing:
        print(f"  Missing features: {missing}")

    for col in available:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df = df[df[target_col].notna()].copy()
    df = df[df[target_col] > 0].copy()
    df = df.dropna(subset=available)

    # 80/20 time-based split
    split   = int(len(df) * 0.8)
    X_train = df[available].iloc[:split]
    X_test  = df[available].iloc[split:]
    y_train = df[target_col].iloc[:split]
    y_test  = df[target_col].iloc[split:]

    print(f"  Train: {len(X_train)} rows  "
          f"({df['timestamp'].iloc[0][:10]} → "
          f"{df['timestamp'].iloc[split-1][:10]})")
    print(f"  Test:  {len(X_test)} rows  "
          f"({df['timestamp'].iloc[split][:10]} → "
          f"{df['timestamp'].iloc[-1][:10]})")
    print(f"  Train AQI mean={y_train.mean():.1f}  "
          f"Test AQI mean={y_test.mean():.1f}")

    return X_train, X_test, y_train, y_test, available

def train_models(X_train, X_test,
                 y_train, y_test, horizon):
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Ridge": (Ridge(alpha=1.0), True),
        "RandomForest": (
            RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            ), False),
        "XGBoost": (
            XGBRegressor(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                verbosity=0
            ), False),
    }

    results = {}
    print(f"\n  ── {horizon} ──")
    for name, (model, scaled) in models.items():
        Xtr = X_train_s if scaled else X_train.values
        Xte = X_test_s  if scaled else X_test.values
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        results[name] = {
            "model":  model,
            "rmse":   rmse,
            "mae":    mae,
            "r2":     r2,
            "scaled": scaled,
            "scaler": scaler,
            "preds":  preds,
        }
        print(f"    {name:15s}  "
              f"RMSE={rmse:.2f}  "
              f"MAE={mae:.2f}  "
              f"R²={r2:.3f}")

    best_name = min(
        results,
        key=lambda k: results[k]["rmse"])
    print(f"  → Best: {best_name}  "
          f"R²={results[best_name]['r2']:.3f}")

    # baseline: predict mean of training set
    baseline_preds = np.full(
        len(y_test), y_train.mean())
    baseline_rmse  = np.sqrt(
        mean_squared_error(y_test, baseline_preds))
    print(f"  → Baseline (mean):  "
          f"RMSE={baseline_rmse:.2f}  R²=0.000")
    improvement = ((baseline_rmse - results[best_name]["rmse"])
                   / baseline_rmse * 100)
    print(f"  → Model improvement over baseline: "
          f"{improvement:.1f}%")

    return results[best_name], best_name

def save_models(horizon_results, project):
    mr = project.get_model_registry()
    for horizon, (result, best_name) in \
            horizon_results.items():
        fname  = f"model_{horizon}.pkl"
        sfname = f"scaler_{horizon}.pkl"
        joblib.dump(result["model"],  fname)
        joblib.dump(result["scaler"], sfname)

        model_obj = mr.python.create_model(
            name=f"aqi_model_{horizon}",
            metrics={
                "rmse": round(result["rmse"], 3),
                "mae":  round(result["mae"],  3),
                "r2":   round(result["r2"],   3),
            },
            description=(
                f"{best_name} — "
                f"Karachi AQI {horizon} forecast"
            )
        )
        model_obj.save(fname)
        print(f"  Saved aqi_model_{horizon}  "
              f"R²={result['r2']:.3f}")

if __name__ == "__main__":
    print("=== Training Pipeline ===\n")
    df, project = load_features()

    horizon_results = {}
    print("── Training models ────────────────────────")

    for target in TARGETS:
        horizon = target.replace("target_", "")
        if target not in df.columns:
            print(f"Skipping {target}")
            continue

        corr = df["aqi"].corr(df[target])
        print(f"\n  {horizon}  "
              f"corr(aqi,target)={corr:.3f}")

        X_train, X_test, \
        y_train, y_test, \
        available = prepare_data(df, target)

        result, best_name = train_models(
            X_train, X_test,
            y_train, y_test,
            horizon
        )
        horizon_results[horizon] = (result, best_name)

    print("\n── Summary ────────────────────────────────")
    for h, (r, name) in horizon_results.items():
        print(f"  {h:6s}  {name:15s}  "
              f"RMSE={r['rmse']:.2f}  "
              f"R²={r['r2']:.3f}")

    print("\n── Saving to Hopsworks ────────────────────")
    save_models(horizon_results, project)
    print("\n=== Training Pipeline Complete ===")