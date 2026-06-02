"""
training_pipeline.py  –  v4
Changes vs v3:
  1. DROPPED zero-importance features from FEATURES list:
       soil_temp, ammonia, visibility, is_daytime,
       wind_chill, wind_dir (raw degrees),
       aqi_roll24_max, aqi_roll24_min

  2. ADDED weather forecast features to FEATURES list:
       temp_forecast_24h/48h/72h
       wind_forecast_24h/48h/72h
       precip_forecast_24h/48h/72h
       pressure_forecast_24h/48h/72h
     Historical backfill rows have NaN for these (filled with median
     at training time).  Inference rows have real forecast values from
     Open-Meteo.  After a few weeks of inference data accumulating,
     retrain and these features will carry real signal for 48h/72h.

  3. All v3 fixes retained: deseasonalized targets, LSTM, version
     auto-increment, single save() call via temp directory.
"""

import os
import shutil
import tempfile
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

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("  [WARN] TensorFlow not installed — LSTM will be skipped.")

load_dotenv()

FEATURES = [
    # ── Pollutants ──────────────────────────────────────
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "dust",
    # ammonia dropped (zero importance)
    "european_aqi", "us_aqi",

    # ── Current weather ─────────────────────────────────
    "temp", "humidity", "wind",
    # wind_dir raw dropped (encoded as sin/cos below)
    "wind_gusts", "precipitation", "pressure",
    "cloud_cover",
    # visibility dropped (zero importance)
    "dew_point", "apparent_temp",
    # soil_temp dropped (zero importance)
    "solar_rad",

    # ── Time ────────────────────────────────────────────
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",

    # ── AQI lags ────────────────────────────────────────
    "aqi_lag1", "aqi_lag2", "aqi_lag3",
    "aqi_lag6", "aqi_lag12", "aqi_lag24", "aqi_lag48",

    # ── Rolling stats (min/max dropped — redundant) ─────
    "aqi_roll3_mean", "aqi_roll6_mean",
    "aqi_roll12_mean", "aqi_roll24_mean",
    "aqi_roll6_std", "aqi_roll24_std",
    # aqi_roll24_max, aqi_roll24_min dropped

    # ── Diff / rate ──────────────────────────────────────
    "aqi_change_rate", "aqi_diff1", "aqi_diff6", "aqi_diff24",

    # ── PM2.5 lags ───────────────────────────────────────
    "pm25_lag1", "pm25_lag24", "pm25_roll6_mean",

    # ── Derived weather ──────────────────────────────────
    "temp_humidity",
    # wind_chill dropped (linear combo of temp+wind)
    "pressure_diff",
    "wind_dir_sin", "wind_dir_cos",   # kept
    "pm25_wind", "dew_depression",
    # is_daytime dropped (zero importance)

    # ── Seasonal (computed in training pipeline) ─────────
    "monthly_mean_aqi", "monthly_std_aqi",

    # ── Weather forecasts (NaN in backfill, live at inference) ──
    "temp_forecast_24h",     "wind_forecast_24h",
    "precip_forecast_24h",   "pressure_forecast_24h",
    "temp_forecast_48h",     "wind_forecast_48h",
    "precip_forecast_48h",   "pressure_forecast_48h",
    "temp_forecast_72h",     "wind_forecast_72h",
    "precip_forecast_72h",   "pressure_forecast_72h",
]

TARGETS = ["target_1h", "target_24h", "target_48h", "target_72h"]


def load_features():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    fs  = project.get_feature_store()
    fg  = fs.get_feature_group("aqi_features", version=1)
    df  = fg.read()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df, project


def add_seasonal_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["month_num"]  = df["timestamp"].dt.month
    stats = (
        df.groupby("month_num")["aqi"]
        .agg(monthly_mean_aqi="mean", monthly_std_aqi="std")
        .reset_index()
    )
    df = df.merge(stats, on="month_num", how="left")
    df = df.drop(columns=["month_num"])
    df["timestamp"] = df["timestamp"].astype(str)
    return df, stats


def deseasonalize_target(df, target_col, monthly_stats):
    df = df.copy()
    df["_ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["_m"]  = df["_ts"].dt.month
    mean_map  = dict(zip(monthly_stats["month_num"],
                         monthly_stats["monthly_mean_aqi"]))
    df["_tmean"] = df["_m"].map(mean_map)
    df[f"{target_col}_deseason"] = df[target_col] - df["_tmean"]
    return df.drop(columns=["_ts", "_m", "_tmean"])


def prepare_data(df, target_col, monthly_stats):
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = deseasonalize_target(df, target_col, monthly_stats)
    deseason_col = f"{target_col}_deseason"

    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  Missing features (will be skipped): {missing}")

    for col in available:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df = df[df[deseason_col].notna()].dropna(subset=available)

    split       = int(len(df) * 0.8)
    X_train     = df[available].iloc[:split]
    X_test      = df[available].iloc[split:]
    y_train     = df[deseason_col].iloc[:split]
    y_test      = df[deseason_col].iloc[split:]
    y_test_raw  = df[target_col].iloc[split:]
    y_train_raw = df[target_col].iloc[:split]

    diff = abs(y_train_raw.mean() - y_test_raw.mean())
    print(f"  Train: {len(X_train)} rows  "
          f"({str(df['timestamp'].iloc[0])[:10]} → "
          f"{str(df['timestamp'].iloc[split-1])[:10]})")
    print(f"  Test:  {len(X_test)} rows  "
          f"({str(df['timestamp'].iloc[split])[:10]} → "
          f"{str(df['timestamp'].iloc[-1])[:10]})")
    print(f"  Train AQI mean={y_train_raw.mean():.1f}  "
          f"Test AQI mean={y_test_raw.mean():.1f}  "
          f"Diff={diff:.1f}")
    print(f"  Deseasonalized — "
          f"Train mean={y_train.mean():.1f}  "
          f"Test mean={y_test.mean():.1f}  (should be ~0)")
    if diff > 10:
        print(f"  [WARN] Distribution shift={diff:.1f} — "
              f"R² will be suppressed on test set")

    return X_train, X_test, y_train, y_test, y_test_raw, available


def build_lstm(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(1, input_dim)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_models(X_train, X_test,
                 y_train, y_test,
                 y_test_raw, horizon, monthly_stats):

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    X_test_months  = X_test["month"].values
    monthly_mean_map = dict(zip(monthly_stats["month_num"],
                                monthly_stats["monthly_mean_aqi"]))

    def reseason(preds_ds, months):
        means = np.array([monthly_mean_map.get(int(m), 0)
                          for m in months])
        return preds_ds + means

    candidates = {
        "Ridge": (Ridge(alpha=1.0), True),
        "RandomForest": (RandomForestRegressor(
            n_estimators=300, max_depth=10,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        ), False),
        "XGBoost": (XGBRegressor(
            n_estimators=500, learning_rate=0.02,
            max_depth=5, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.5,
            reg_lambda=2.0, random_state=42, verbosity=0
        ), False),
    }

    results = {}
    print(f"\n  ── {horizon} ──")

    for name, (model, scaled) in candidates.items():
        Xtr = X_train_s if scaled else X_train.values
        Xte = X_test_s  if scaled else X_test.values
        model.fit(Xtr, y_train)
        preds_ds  = model.predict(Xte)
        preds_raw = reseason(preds_ds, X_test_months)
        rmse = np.sqrt(mean_squared_error(y_test_raw, preds_raw))
        mae  = mean_absolute_error(y_test_raw, preds_raw)
        r2   = r2_score(y_test_raw, preds_raw)
        r2ds = r2_score(y_test, preds_ds)
        results[name] = dict(model=model, rmse=rmse, mae=mae,
                             r2=r2, r2_ds=r2ds, scaled=scaled,
                             scaler=scaler, is_lstm=False)
        print(f"    {name:15s}  RMSE={rmse:.2f}  MAE={mae:.2f}  "
              f"R²={r2:.3f}  R²_ds={r2ds:.3f}")

    if LSTM_AVAILABLE:
        try:
            idim = X_train_s.shape[1]
            lstm = build_lstm(idim)
            early = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5,
                restore_best_weights=True, verbose=0)
            lstm.fit(
                X_train_s.reshape(-1, 1, idim), y_train.values,
                validation_split=0.1, epochs=50,
                batch_size=256, callbacks=[early], verbose=0)
            preds_ds  = lstm.predict(
                X_test_s.reshape(-1, 1, idim), verbose=0).flatten()
            preds_raw = reseason(preds_ds, X_test_months)
            rmse = np.sqrt(mean_squared_error(y_test_raw, preds_raw))
            mae  = mean_absolute_error(y_test_raw, preds_raw)
            r2   = r2_score(y_test_raw, preds_raw)
            r2ds = r2_score(y_test, preds_ds)
            results["LSTM"] = dict(model=lstm, rmse=rmse, mae=mae,
                                   r2=r2, r2_ds=r2ds, scaled=True,
                                   scaler=scaler, is_lstm=True)
            print(f"    {'LSTM':15s}  RMSE={rmse:.2f}  MAE={mae:.2f}  "
                  f"R²={r2:.3f}  R²_ds={r2ds:.3f}")
        except Exception as e:
            print(f"    LSTM failed: {e}")

    best_name = max(results, key=lambda k: results[k]["r2"])
    best      = results[best_name]
    print(f"  → Best: {best_name}  R²={best['r2']:.3f}")

    baseline_rmse = np.sqrt(mean_squared_error(
        y_test_raw,
        np.full(len(y_test_raw), y_test_raw.mean())))
    print(f"  → Baseline RMSE={baseline_rmse:.2f}  "
          f"Improvement={((baseline_rmse-best['rmse'])/baseline_rmse*100):.1f}%")

    return best, best_name


def save_models(horizon_results, project, monthly_stats):
    mr = project.get_model_registry()

    for horizon, (result, best_name) in horizon_results.items():
        tmp_dir = tempfile.mkdtemp(prefix=f"aqi_{horizon}_")
        try:
            model_fname  = os.path.join(tmp_dir, f"model_{horizon}.pkl")
            scaler_fname = os.path.join(tmp_dir, f"scaler_{horizon}.pkl")
            stats_fname  = os.path.join(tmp_dir, "monthly_stats.pkl")

            if result.get("is_lstm"):
                lstm_path = os.path.join(
                    tmp_dir, f"lstm_model_{horizon}.keras")
                result["model"].save(lstm_path)
                joblib.dump({"type": "lstm",
                             "path": f"lstm_model_{horizon}.keras"},
                            model_fname)
            else:
                joblib.dump(result["model"], model_fname)

            joblib.dump(result["scaler"], scaler_fname)
            joblib.dump(monthly_stats,    stats_fname)

            model_name = f"aqi_model_{horizon}"
            try:
                existing = mr.get_models(name=model_name)
                version  = (max(m.version for m in existing) + 1
                            if existing else 1)
            except Exception:
                version = 1

            model_obj = mr.python.create_model(
                name=model_name,
                version=version,
                metrics=dict(rmse=round(result["rmse"], 3),
                             mae=round(result["mae"],  3),
                             r2=round(result["r2"],   3)),
                description=(f"{best_name} — Karachi AQI {horizon} "
                             f"(deseasonalized, forecast features)")
            )
            model_obj.save(tmp_dir)
            print(f"  Saved aqi_model_{horizon} v{version}  "
                  f"R²={result['r2']:.3f}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=== Training Pipeline v4 ===\n")
    df, project = load_features()

    print(f"Date range: {df['timestamp'].min()} to "
          f"{df['timestamp'].max()}")
    print(f"AQI range: {df['aqi'].min()} to {df['aqi'].max()}  "
          f"mean={df['aqi'].mean():.1f}")

    if "target_1h" not in df.columns:
        print("\n  [INFO] Computing target_1h from aqi column.")
        df = df.sort_values("timestamp").copy()
        df["target_1h"] = df["aqi"].shift(-1)

    df, monthly_stats = add_seasonal_features(df)
    print("\n  Monthly AQI stats:")
    print(monthly_stats.to_string(index=False))

    horizon_results = {}
    print("\n── Training models ────────────────────────")

    for target in TARGETS:
        horizon = target.replace("target_", "")
        if target not in df.columns:
            print(f"Skipping {target} — not in feature store")
            continue

        corr = df["aqi"].corr(df[target])
        print(f"\n  {horizon}  corr(aqi,target)={corr:.3f}")

        X_train, X_test, y_train, y_test, y_test_raw, available = \
            prepare_data(df, target, monthly_stats)

        result, best_name = train_models(
            X_train, X_test, y_train, y_test,
            y_test_raw, horizon, monthly_stats)
        horizon_results[horizon] = (result, best_name)

    print("\n── Summary ────────────────────────────────")
    for h, (r, name) in horizon_results.items():
        print(f"  {h:6s}  {name:15s}  "
              f"RMSE={r['rmse']:.2f}  R²={r['r2']:.3f}")

    print("\n── Saving to Hopsworks ────────────────────")
    save_models(horizon_results, project, monthly_stats)
    print("\n=== Training Pipeline Complete ===")