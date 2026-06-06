"""
training_pipeline.py  –  v6

WHAT CHANGED vs v5 and WHY:
=============================================================================

CHANGE 1 — Differenced targets (highest impact)
  v5: target_resid = target_Xh - aqi_trend(T)
      At reconstruction: predicted_aqi = residual + trend(T)
      Problem: trend(T) is the 7-day mean ENDING at T, but the target is
      AQI at T+48. If trend is still shifting upward in that window,
      trend(T) underestimates the true level → systematic offset → r2 < 0
      even though r2_resid was positive (model was learning correctly).

  v6: target_Xh_diff = target_Xh - aqi_current (at time T, NOT at T+X)
      At reconstruction: predicted_aqi = aqi_now + predicted_diff
      aqi_now is always exactly known at inference. No trend needed.
      No lookup table. No offset error. Cleaner stationarity for LSTM.
  
  This is the correct formulation: the model predicts HOW MUCH AQI will
  change over the next X hours, not what the absolute level will be.
  The strong negative correlation (r≈-0.43) between aqi_now and the diff
  (mean reversion: high AQI tends to fall) becomes the primary signal.

CHANGE 2 — Extended lag features in FEATURES list
  Added: aqi_lag72, aqi_lag96, aqi_lag120, aqi_lag168
  Added: aqi_roll48_mean, aqi_roll72_mean
  These were added to backfill and feature_pipeline schemas (v6).
  Weekly lag (168h) captures same-hour-last-week traffic patterns.
  Long rolling means capture mean-reversion: if roll48_mean >> aqi_now,
  AQI is likely to rise back toward baseline (and vice versa).
  aqi_roll168_mean skipped — redundant with aqi_trend.

CHANGE 3 — Ridge alpha tuned per horizon via TimeSeriesSplit CV
  v5 used Ridge(alpha=1.0) for all horizons.
  v6 searches alpha in [0.01, 0.1, 1, 10, 50, 100, 500] using
  TimeSeriesSplit(n_splits=5). This finds the right regularisation
  strength for each horizon separately:
    1h:  likely alpha → small (strong clean signal)
    24h: likely alpha → 1-10
    48h/72h: likely alpha → 10-100 (weak noisy signal needs more shrinkage)
  TimeSeriesSplit is used for alpha search only. Final reported metrics
  still come from the single 80/20 temporal split (reproducible baseline).

CHANGE 4 — RandomForest hyperparameters improved
  n_estimators 300 → 500, max_depth 10 → 12.
  min_samples_leaf stays at 3 (prevents overfitting on 1944 rows).
  Depth 12 is a compromise: deeper than 10 to find interactions,
  shallower than the suggested 15 which would overfit on this dataset size.

CHANGE 5 — aqi_trend removed as primary deseasonalizer, kept as feature
  With differenced targets, aqi_trend is no longer needed for target
  construction. It is still included in FEATURES because it provides
  a smooth representation of the current level baseline — useful context
  for the model even if it no longer plays a structural role.
=============================================================================
"""

import os
import sys
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

try:
    import confluent_kafka  # noqa: F401
except ImportError:
    print(
        "\n[ERROR] confluent_kafka not installed.\n"
        "Fix: pip install confluent-kafka\n"
    )
    sys.exit(1)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("  [WARN] TensorFlow not installed — LSTM will be skipped.")

load_dotenv()

# ── Feature list — all columns present in the v6 feature group schema ──
FEATURES = [
    # ── Pollutants ──────────────────────────────────────────────────
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "dust", "european_aqi", "us_aqi",

    # ── Current weather ─────────────────────────────────────────────
    "temp", "humidity", "wind", "wind_gusts",
    "precipitation", "pressure", "cloud_cover",
    "dew_point", "apparent_temp",
    # solar_rad removed: OpenWeather free tier never provides it, so every
    # live inference row has solar_rad=NaN, imputed to median. The model
    # was trained on real backfill values → systematic mismatch at inference
    # → spurious SHAP values (+1.7 for 24h, +0.98 for 48h). Removed.

    # ── Time ────────────────────────────────────────────────────────
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",

    # ── AQI lags (original set) ──────────────────────────────────────
    "aqi_lag1", "aqi_lag2", "aqi_lag3",
    "aqi_lag6", "aqi_lag12", "aqi_lag24", "aqi_lag48",

    # ── AQI lags (extended — added in v6) ───────────────────────────
    # lag72/96/120: 3/4/5-day lags for persistence at longer horizons
    # lag168: same hour last week — captures weekly traffic/activity cycle
    "aqi_lag72", "aqi_lag96", "aqi_lag120", "aqi_lag168",

    # ── Rolling stats ───────────────────────────────────────────────
    "aqi_roll3_mean", "aqi_roll6_mean",
    "aqi_roll12_mean", "aqi_roll24_mean",
    # roll48/72: 2-day/3-day smooth means for mean-reversion signal
    "aqi_roll48_mean", "aqi_roll72_mean",
    "aqi_roll6_std", "aqi_roll24_std",

    # ── Diff / rate — most important for differenced targets ─────────
    # corr(aqi_diff1, target_48h_diff) ≈ -0.49: strong mean-reversion
    "aqi_change_rate", "aqi_diff1", "aqi_diff6", "aqi_diff24",

    # ── PM2.5 lags ───────────────────────────────────────────────────
    "pm25_lag1", "pm25_lag24", "pm25_roll6_mean",

    # ── Derived weather ──────────────────────────────────────────────
    "temp_humidity", "pressure_diff",
    "wind_dir_sin", "wind_dir_cos",
    "pm25_wind", "dew_depression",

    # ── Trend baseline (kept as feature, no longer used for deseason) ─
    "aqi_trend",
]

# Horizons and their differenced target column names
# diff targets = target_Xh - aqi_current
# At inference: predicted_aqi = aqi_now + model.predict(features)
HORIZONS = {
    "1h":  ("target_1h",  "target_1h_diff"),
    "24h": ("target_24h", "target_24h_diff"),
    "48h": ("target_48h", "target_48h_diff"),
    "72h": ("target_72h", "target_72h_diff"),
}

TREND_WINDOW_HOURS = 168  # 7-day rolling mean


def load_features():
    """Load all feature rows from MongoDB Atlas."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from mongo_store import read_df

    print("Loading features from MongoDB Atlas...")
    df = read_df()
    if df is None or len(df) == 0:
        raise RuntimeError("MongoDB is empty — run backfill_pipeline.py first.")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Still connect to Hopsworks for model registry
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    return df, project


def parse_timestamps(df):
    """ISO8601 parse handles both '2026-03-07 00:00:00+00:00' and
    '2026-06-03 08:29:51.227071+00:00' (microseconds from feature_pipeline)."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="ISO8601", utc=True)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    return df


def compute_trend(df):
    """Rolling 7-day AQI trend — used as a context feature, not for target
    construction (that role was replaced by differenced targets in v6)."""
    df = df.copy()
    df["aqi_trend"] = (
        df["aqi"]
        .rolling(TREND_WINDOW_HOURS, min_periods=24)
        .mean()
    )
    df["aqi_trend"] = df["aqi_trend"].fillna(df["aqi"].expanding().mean())
    return df


def build_diff_targets(df):
    """
    Construct differenced targets: target_Xh_diff = aqi[T+X] - aqi[T].
    Model learns to predict the CHANGE in AQI, not the absolute level.
    Reconstruction at inference: predicted_aqi = aqi_now + predicted_diff.

    Why this is better than rolling-trend deseasonalization:
      - aqi_now is exactly known at prediction time (zero offset error)
      - The residuals are zero-mean by construction (no lookup table needed)
      - More stationary signal = better for LSTM
      - Strong mean-reversion signal: corr(aqi_now, target_48h_diff) ≈ -0.43
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    for horizon, (raw_col, diff_col) in HORIZONS.items():
        if raw_col in df.columns:
            df[diff_col] = df[raw_col] - df["aqi"]
    return df


def prepare_data(df, raw_target_col, diff_target_col):
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  [WARN] Missing from schema (skipped): {missing}")

    for col in available:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Keep only rows where both raw target and diff target exist
    df = df[df[diff_target_col].notna() & df[raw_target_col].notna()]
    df = df.dropna(subset=available)

    split       = int(len(df) * 0.8)
    X_train     = df[available].iloc[:split]
    X_test      = df[available].iloc[split:]
    y_train     = df[diff_target_col].iloc[:split]    # train on diff
    y_test      = df[diff_target_col].iloc[split:]    # eval on diff
    y_test_raw  = df[raw_target_col].iloc[split:]     # final eval on absolute AQI
    y_train_raw = df[raw_target_col].iloc[:split]
    aqi_test    = df["aqi"].iloc[split:]              # aqi_now for reconstruction

    train_start = str(df["timestamp"].iloc[0])[:10]
    train_end   = str(df["timestamp"].iloc[split - 1])[:10]
    test_start  = str(df["timestamp"].iloc[split])[:10]
    test_end    = str(df["timestamp"].iloc[-1])[:10]
    mean_diff   = abs(y_train_raw.mean() - y_test_raw.mean())

    print(f"  Train: {len(X_train)} rows  ({train_start} → {train_end})")
    print(f"  Test:  {len(X_test)} rows  ({test_start} → {test_end})")
    print(f"  Train AQI mean={y_train_raw.mean():.1f}  "
          f"Test AQI mean={y_test_raw.mean():.1f}  "
          f"Shift={mean_diff:.1f}")
    print(f"  Diff target — train mean={y_train.mean():.2f}  "
          f"test mean={y_test.mean():.2f}  (both ~0 expected)")

    return (X_train, X_test,
            y_train, y_test,
            y_test_raw, aqi_test, available)


def tune_ridge_alpha(X_train, y_train):
    """
    Find the best Ridge alpha for this horizon using TimeSeriesSplit CV.
    CV is on the TRAINING set only — no test-set contamination.

    n_splits=3 instead of 5:
      With 1555 train rows and 5 splits, fold 1 had only 258 rows —
      too small for the noisy 24h/48h/72h diff signal. CV always picked
      alpha=500 (near-zero prediction) on tiny folds. With n_splits=3,
      fold 1 has 388 rows (50% more), giving a more reliable alpha estimate.

    Alpha range starts at 0.001:
      Ensures the search can pick a low alpha for the clean 1h signal
      where aggressive regularisation was actively hurting R²_diff.
    """
    alphas     = [0.001, 0.01, 0.1, 1, 10, 50, 100, 500]
    tscv       = TimeSeriesSplit(n_splits=3)
    best_alpha = 1.0
    best_score = -np.inf

    X_arr = X_train.values if hasattr(X_train, "values") else X_train
    y_arr = y_train.values if hasattr(y_train, "values") else y_train

    alpha_scores = {}
    for alpha in alphas:
        fold_scores = []
        for tr_idx, val_idx in tscv.split(X_arr):
            Xtr, Xval = X_arr[tr_idx], X_arr[val_idx]
            ytr, yval = y_arr[tr_idx], y_arr[val_idx]
            sc = StandardScaler()
            Xtr_s  = sc.fit_transform(Xtr)
            Xval_s = sc.transform(Xval)
            m = Ridge(alpha=alpha)
            m.fit(Xtr_s, ytr)
            preds = m.predict(Xval_s)
            fold_scores.append(r2_score(yval, preds))
        mean_score = np.mean(fold_scores)
        alpha_scores[alpha] = round(mean_score, 3)
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha

    scores_str = "  ".join(f"a={a}:{s:+.3f}" for a, s in alpha_scores.items())
    print(f"    Ridge CV: {scores_str}")
    print(f"    Ridge CV best alpha={best_alpha}  CV R²={best_score:.3f}")
    return best_alpha


def make_sequences(X_arr, y_arr, seq_len):
    """
    Slide a window of seq_len over X to create LSTM sequences.
    Each output sample: X[i-seq_len+1 : i+1] → y[i]
    Returns Xs shape (N-seq_len+1, seq_len, features), ys shape (N-seq_len+1,)
    """
    Xs, ys = [], []
    for i in range(seq_len - 1, len(X_arr)):
        Xs.append(X_arr[i - seq_len + 1 : i + 1])
        ys.append(y_arr[i])
    return np.array(Xs), np.array(ys)


def build_lstm(seq_len, input_dim):
    """
    LSTM that sees a real temporal sequence of seq_len timesteps.
    seq_len=24 means the model looks back 24 hours of feature history.
    This is qualitatively different from seq_len=1, which has no recurrence.
    """
    model = keras.Sequential([
        layers.Input(shape=(seq_len, input_dim)),
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
                 y_test_raw, aqi_test, horizon):
    """
    Train all models on differenced targets.
    Reconstruction: predicted_aqi = aqi_now + predicted_diff
    aqi_now is exact → no offset error possible.
    Reports R²_diff (on change prediction) and R² (on absolute AQI).
    """
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    aqi_arr   = aqi_test.values

    def reconstruct(preds_diff):
        return aqi_arr + preds_diff

    # ── Ridge with tuned alpha ───────────────────────────────────────
    best_alpha = tune_ridge_alpha(X_train, y_train)
    ridge      = Ridge(alpha=best_alpha)

    candidates = {
        f"Ridge(a={best_alpha})": (ridge, True),
        "RandomForest": (RandomForestRegressor(
            n_estimators=500,
            max_depth=12,        # compromise between v5's 10 and suggested 15
            min_samples_leaf=3,  # keep at 3: prevents overfit on ~1550 train rows
            random_state=42,
            n_jobs=-1,
        ), False),
        "XGBoost": (XGBRegressor(
            n_estimators=500, learning_rate=0.02,
            max_depth=5, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.5,
            reg_lambda=2.0, random_state=42, verbosity=0,
        ), False),
    }

    results = {}
    print(f"\n  ── {horizon} ──")

    for name, (model, scaled) in candidates.items():
        Xtr = X_train_s if scaled else X_train.values
        Xte = X_test_s  if scaled else X_test.values
        model.fit(Xtr, y_train)
        preds_diff = model.predict(Xte)
        preds_raw  = reconstruct(preds_diff)

        rmse     = np.sqrt(mean_squared_error(y_test_raw, preds_raw))
        mae      = mean_absolute_error(y_test_raw, preds_raw)
        r2_abs   = r2_score(y_test_raw, preds_raw)
        r2_diff  = r2_score(y_test, preds_diff)
        results[name] = dict(
            model=model, rmse=rmse, mae=mae,
            r2=r2_abs, r2_diff=r2_diff,
            scaled=scaled, scaler=scaler, is_lstm=False,
            alpha=best_alpha if "Ridge" in name else None,
        )
        print(f"    {name:20s}  RMSE={rmse:.2f}  MAE={mae:.2f}  "
              f"R²={r2_abs:.3f}  R²_diff={r2_diff:.3f}")

    # ── LSTM with proper temporal sequences ─────────────────────────
    # seq_len=24: the LSTM sees the last 24 hours of feature history
    # for each prediction. This is the minimum meaningful window for
    # hourly AQI (one full diurnal cycle). With seq_len=1 (old code),
    # the LSTM had no temporal sequence to recur over — it was just an MLP.
    if LSTM_AVAILABLE:
        try:
            SEQ_LEN = 24
            idim    = X_train_s.shape[1]

            Xs_train, ys_train = make_sequences(
                X_train_s, y_train.values, SEQ_LEN)
            Xs_test,  ys_test  = make_sequences(
                X_test_s,  y_test.values,  SEQ_LEN)

            # aqi_test for reconstruction must also be trimmed to match
            # the sequence output (first SEQ_LEN-1 rows are consumed by window)
            aqi_arr_seq = aqi_arr[SEQ_LEN - 1:]

            model = build_lstm(SEQ_LEN, idim)
            early = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5,
                restore_best_weights=True, verbose=0)
            model.fit(
                Xs_train, ys_train,
                validation_split=0.1, epochs=50,
                batch_size=64, callbacks=[early], verbose=0)

            preds_diff = model.predict(Xs_test, verbose=0).flatten()
            preds_raw  = aqi_arr_seq + preds_diff

            # y_test_raw must also be trimmed to match
            y_test_raw_seq = y_test_raw.values[SEQ_LEN - 1:]

            rmse    = np.sqrt(mean_squared_error(y_test_raw_seq, preds_raw))
            mae     = mean_absolute_error(y_test_raw_seq, preds_raw)
            r2_abs  = r2_score(y_test_raw_seq, preds_raw)
            r2_diff = r2_score(ys_test, preds_diff)
            results["LSTM"] = dict(
                model=model, rmse=rmse, mae=mae,
                r2=r2_abs, r2_diff=r2_diff,
                scaled=True, scaler=scaler, is_lstm=True,
                seq_len=SEQ_LEN,
            )
            print(f"    {'LSTM(seq=24)':20s}  RMSE={rmse:.2f}  MAE={mae:.2f}  "
                  f"R²={r2_abs:.3f}  R²_diff={r2_diff:.3f}")
        except Exception as e:
            print(f"    LSTM failed: {e}")

    best_name = max(results, key=lambda k: results[k]["r2"])
    best      = results[best_name]

    baseline_rmse = np.sqrt(mean_squared_error(
        y_test_raw, np.full(len(y_test_raw), y_test_raw.mean())))
    improvement = (baseline_rmse - best["rmse"]) / baseline_rmse * 100
    print(f"  → Best: {best_name}  "
          f"R²={best['r2']:.3f}  R²_diff={best['r2_diff']:.3f}")
    print(f"  → Baseline RMSE={baseline_rmse:.2f}  "
          f"Improvement={improvement:.1f}%")

    # Return ALL results so save_models can build the ensemble
    return results, best_name


def compute_ensemble_weights(results_dict, horizon):
    """
    Compute R²-proportional weights for ensemble blending.
    Only include models with positive R² (negative R² models hurt the ensemble).
    Returns a dict: {model_name: weight} that sums to 1.0.

    Rationale per horizon:
      1h:  RF near-perfect → RF dominates; LSTM slightly negative → exclude
      24h: RF best → RF×0.6 + LSTM×0.4 (when LSTM R² is positive)
      48h: LSTM best → LSTM×0.6 + Ridge×0.4
      72h: LSTM dominates → LSTM×0.65 + Ridge×0.35
    """
    # Filter to models with positive R²
    positive = {k: v for k, v in results_dict.items() if v["r2"] > 0}
    if not positive:
        # Fall back to the single best by R²_diff
        best_k = max(results_dict, key=lambda k: results_dict[k]["r2_diff"])
        return {best_k: 1.0}
    if len(positive) == 1:
        return {list(positive.keys())[0]: 1.0}

    # R²-proportional weights among positive-R² models
    total_r2 = sum(v["r2"] for v in positive.values())
    weights = {k: v["r2"] / total_r2 for k, v in positive.items()}

    # Print ensemble composition
    parts = "  ".join(f"{k}×{w:.2f}" for k, w in weights.items())
    print(f"    Ensemble ({horizon}): {parts}")
    return weights


def save_models(horizon_results, project):
    """
    Save all trained models for each horizon plus ensemble weights.
    The model bundle contains:
      - model_{horizon}.pkl         : best single model (sklearn/XGBoost)
      - lstm_model_{horizon}.keras  : LSTM model (if LSTM was trained)
      - all_{horizon}.pkl           : all model results (for ensemble)
      - ensemble_weights_{horizon}.pkl : {model_name: weight}
      - scaler_{horizon}.pkl        : StandardScaler fitted on training data
      - meta_{horizon}.pkl          : reconstruction metadata
    """
    mr = project.get_model_registry()

    for horizon, (all_results, best_name) in horizon_results.items():
        # horizon_results[h] = (all_results_dict, best_name)
        # all_results_dict = {name: {model, rmse, mae, r2, r2_diff, scaler, ...}}
        best_result = all_results[best_name]
        tmp_dir     = tempfile.mkdtemp(prefix=f"aqi_{horizon}_")
        try:
            # ── Save best model ────────────────────────────────────────
            model_fname  = os.path.join(tmp_dir, f"model_{horizon}.pkl")
            scaler_fname = os.path.join(tmp_dir, f"scaler_{horizon}.pkl")

            if best_result.get("is_lstm"):
                lstm_path = os.path.join(tmp_dir, f"lstm_model_{horizon}.keras")
                best_result["model"].save(lstm_path)
                joblib.dump({"type": "lstm",
                             "path": f"lstm_model_{horizon}.keras"},
                            model_fname)
            else:
                joblib.dump(best_result["model"], model_fname)

            joblib.dump(best_result["scaler"], scaler_fname)

            # ── Save ALL models for ensemble ───────────────────────────
            all_pkl = os.path.join(tmp_dir, f"all_{horizon}.pkl")
            ensemble_bundle = {}
            for name, res in all_results.items():
                entry = {
                    "r2":      res["r2"],
                    "r2_diff": res["r2_diff"],
                    "rmse":    res["rmse"],
                    "scaled":  res["scaled"],
                    "is_lstm": res.get("is_lstm", False),
                }
                if res.get("is_lstm"):
                    lstm_path_e = os.path.join(
                        tmp_dir, f"lstm_model_{horizon}_{name}.keras")
                    res["model"].save(lstm_path_e)
                    entry["model"] = {"type": "lstm",
                                      "path": f"lstm_model_{horizon}_{name}.keras"}
                else:
                    entry["model"] = res["model"]
                ensemble_bundle[name] = entry
            joblib.dump(ensemble_bundle, all_pkl)

            # ── Compute and save ensemble weights ──────────────────────
            weights = compute_ensemble_weights(all_results, horizon)
            joblib.dump(weights,
                        os.path.join(tmp_dir,
                                     f"ensemble_weights_{horizon}.pkl"))

            # ── Save metadata ──────────────────────────────────────────
            meta = {
                "approach":         "differenced_target",
                "reconstruct":      "predicted_aqi = aqi_now + model.predict(features)",
                "horizon":          horizon,
                "best_model":       best_name,
                "ensemble_weights": weights,
            }
            joblib.dump(meta, os.path.join(tmp_dir, f"meta_{horizon}.pkl"))

            # ── Register in Hopsworks ──────────────────────────────────
            model_name = f"aqi_model_{horizon}"
            try:
                existing = mr.get_models(name=model_name)
                version  = (max(m.version for m in existing) + 1
                            if existing else 1)
            except Exception:
                version = 1

            safe_name = best_name.encode("ascii", errors="replace").decode("ascii")
            safe_name = safe_name.replace("?", "a")
            model_obj = mr.python.create_model(
                name=model_name,
                version=version,
                metrics=dict(
                    rmse=round(best_result["rmse"], 3),
                    mae=round(best_result["mae"],  3),
                    r2=round(best_result["r2"],   3),
                    r2_diff=round(best_result["r2_diff"], 3),
                ),
                description=(
                    f"{safe_name} + ensemble -- Karachi AQI {horizon} "
                    f"(differenced-target, v6)"
                )
            )
            model_obj.save(tmp_dir)
            print(f"  Saved aqi_model_{horizon} v{version}  "
                  f"R²={best_result['r2']:.3f}  "
                  f"ensemble_weights={weights}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=== Training Pipeline v6 ===\n")

    df, project = load_features()
    df = parse_timestamps(df)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"AQI range:  {df['aqi'].min()} to {df['aqi'].max()}  "
          f"mean={df['aqi'].mean():.1f}")

    # Compute rolling trend (kept as a feature, not used for deseasonalisation)
    df = compute_trend(df)

    # Verify all expected features are present
    missing_from_schema = [f for f in FEATURES if f not in df.columns]
    if missing_from_schema:
        print(f"\n  [WARN] These FEATURES are absent from the feature group:")
        for f in missing_from_schema:
            print(f"    {f}")
        print("  Run backfill_pipeline.py to regenerate with v6 schema.")
    else:
        print(f"\n  All {len(FEATURES)} features present ✓")

    # Build differenced targets
    df = build_diff_targets(df)

    horizon_results = {}
    print("\n── Training models ─────────────────────────────")

    for horizon, (raw_col, diff_col) in HORIZONS.items():
        if raw_col not in df.columns:
            print(f"\n  Skipping {horizon} — {raw_col} not found")
            continue
        if diff_col not in df.columns:
            print(f"\n  Skipping {horizon} — {diff_col} not built")
            continue

        corr_raw  = df["aqi"].corr(df[raw_col].dropna())
        corr_diff = df["aqi"].corr(df[diff_col].dropna())
        print(f"\n  {horizon}  corr(aqi, raw_target)={corr_raw:.3f}  "
              f"corr(aqi, diff_target)={corr_diff:.3f}")

        (X_train, X_test,
         y_train, y_test,
         y_test_raw, aqi_test, available) = prepare_data(
            df, raw_col, diff_col)

        result, best_name = train_models(
            X_train, X_test,
            y_train, y_test,
            y_test_raw, aqi_test, horizon)

        horizon_results[horizon] = (result, best_name)

    print("\n── Summary ─────────────────────────────────────")
    print(f"  {'Horizon':8s}  {'Best model':22s}  {'RMSE':>8}  "
          f"{'R²':>7}  {'R²_diff':>8}")
    print(f"  {'-'*8}  {'-'*22}  {'-'*8}  {'-'*7}  {'-'*8}")
    for h, (all_res, name) in horizon_results.items():
        r = all_res[name]
        print(f"  {h:8s}  {name:22s}  {r['rmse']:8.2f}  "
              f"{r['r2']:7.3f}  {r['r2_diff']:8.3f}")

    print("\n── Saving to Hopsworks ─────────────────────────")
    save_models(horizon_results, project)
    print("\n=== Training Pipeline Complete ===")