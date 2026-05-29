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
    "temp", "humidity", "wind",
    "hour", "day_of_week", "month", "is_weekend",
    "aqi_lag1", "aqi_lag2", "aqi_lag24",
    "aqi_roll6_mean", "aqi_roll24_mean",
    "aqi_roll6_std", "aqi_change_rate"
]
TARGET = "target_24h"

# ── STEP 1: LOAD ─────────────────────────────────────
def load_features():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read()
    print(f"Loaded {len(df)} rows")
    return df, project

# ── STEP 2: PREPARE ──────────────────────────────────
def prepare_data(df):
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # clean outliers
    df = df[(df["aqi"] > 5) & (df["aqi"] < 400)]

    # keep only rows with valid 24h target
    df = df[df[TARGET] > 0].copy()
    df = df.dropna(subset=FEATURES + [TARGET])

    print(f"Clean rows: {len(df)}")
    print(f"AQI range: {df['aqi'].min():.0f} "
          f"to {df['aqi'].max():.0f}")
    print(f"Target range: {df[TARGET].min():.0f} "
          f"to {df[TARGET].max():.0f}")
    print(f"Correlation aqi→target_24h: "
          f"{df['aqi'].corr(df[TARGET]):.3f}")

    split = int(len(df) * 0.8)
    X_train = df[FEATURES].iloc[:split]
    X_test  = df[FEATURES].iloc[split:]
    y_train = df[TARGET].iloc[:split]
    y_test  = df[TARGET].iloc[split:]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

# ── STEP 3A: SKLEARN MODELS ──────────────────────────
def train_sklearn_models(X_train, X_test,
                         y_train, y_test):
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_train)
    Xte_s  = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        ),
    }

    results = {}
    print("\n── Sklearn Models ────────────────────────")
    for name, model in models.items():
        Xtr = Xtr_s if name == "Ridge" else X_train.values
        Xte = Xte_s if name == "Ridge" else X_test.values
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        results[name] = {
            "model": model, "rmse": rmse,
            "mae": mae, "r2": r2
        }
        print(f"{name:15s} → RMSE={rmse:.2f}  "
              f"MAE={mae:.2f}  R²={r2:.3f}")
    return results

# ── STEP 3B: LSTM MODEL ──────────────────────────────
def train_lstm(X_train, X_test, y_train, y_test):
    print("\n── LSTM Model ────────────────────────────")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            LSTM, Dense, Dropout, BatchNormalization)
        from tensorflow.keras.callbacks import (
            EarlyStopping, ReduceLROnPlateau)
        from tensorflow.keras.optimizers import Adam

        scaler = StandardScaler()
        Xtr_s  = scaler.fit_transform(X_train)
        Xte_s  = scaler.transform(X_test)

        # reshape for LSTM: (samples, timesteps, features)
        Xtr_3d = Xtr_s.reshape(
            Xtr_s.shape[0], 1, Xtr_s.shape[1])
        Xte_3d = Xte_s.reshape(
            Xte_s.shape[0], 1, Xte_s.shape[1])

        model = Sequential([
            LSTM(128, input_shape=(1, X_train.shape[1]),
                 return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )

        callbacks = [
            EarlyStopping(patience=10,
                          restore_best_weights=True,
                          verbose=1),
            ReduceLROnPlateau(patience=5,
                              factor=0.5, verbose=1)
        ]

        model.fit(
            Xtr_3d, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0
        )

        preds = model.predict(Xte_3d).flatten()
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)

        print(f"{'LSTM':15s} → RMSE={rmse:.2f}  "
              f"MAE={mae:.2f}  R²={r2:.3f}")

        # save lstm separately
        model.save("lstm_model.keras")
        joblib.dump(scaler, "lstm_scaler.pkl")

        return {
            "LSTM": {
                "model": model, "rmse": rmse,
                "mae": mae, "r2": r2
            }
        }

    except ImportError:
        print("TensorFlow not installed — skipping LSTM")
        print("Install with: pip install tensorflow")
        return {}

# ── STEP 4: SAVE BEST MODEL ──────────────────────────
def save_best_model(all_results, project):
    best_name = min(all_results,
                    key=lambda k: all_results[k]["rmse"])
    best      = all_results[best_name]

    print(f"\n── Winner: {best_name} ───────────────────")
    print(f"RMSE={best['rmse']:.2f}  "
          f"MAE={best['mae']:.2f}  "
          f"R²={best['r2']:.3f}")

    # save the best model
    if best_name == "LSTM":
        # already saved as lstm_model.keras
        model_file = "lstm_model.keras"
    else:
        joblib.dump(best["model"], "model.pkl")
        model_file = "model.pkl"

    mr = project.get_model_registry()
    aqi_model = mr.python.create_model(
        name="aqi_model",
        metrics={
            "rmse": round(best["rmse"], 3),
            "mae":  round(best["mae"],  3),
            "r2":   round(best["r2"],   3),
        },
        description=f"{best_name} — Karachi AQI 24h"
    )
    aqi_model.save(model_file)
    print("Model saved to Hopsworks registry!")
    return best_name

# ── MAIN ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Training Pipeline Starting ===\n")

    # install tensorflow if missing
    try:
        import tensorflow
    except ImportError:
        print("Installing TensorFlow...")
        os.system("pip install tensorflow")

    df, project = load_features()
    X_train, X_test, y_train, y_test = prepare_data(df)

    # train all models
    all_results = {}
    all_results.update(
        train_sklearn_models(
            X_train, X_test, y_train, y_test))
    all_results.update(
        train_lstm(
            X_train, X_test, y_train, y_test))

    # print full comparison
    print("\n── Full Comparison ───────────────────────")
    for name, r in sorted(
            all_results.items(),
            key=lambda x: x[1]["rmse"]):
        print(f"{name:15s} → RMSE={r['rmse']:.2f}  "
              f"MAE={r['mae']:.2f}  R²={r['r2']:.3f}")

    best_name = save_best_model(all_results, project)
    print(f"\n=== Done — best model: {best_name} ===")