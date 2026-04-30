import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import holidays
from sklearn.metrics import r2_score

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from src.artifacts import save_feature_columns, save_preprocessing_config

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
EXTERNAL_DATA_DIR = ROOT_DIR / "data" / "external"
MODELS_DIR = ROOT_DIR / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "baseline"


def rmspe(y_true, y_pred):
    mask = y_true > 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


print("=" * 60)
print("BASELINE: XGBoost with weather and public holiday, without lags")
print("=" * 60)

print("\n1. Loading data...")
train = pd.read_csv(RAW_DATA_DIR / "train.csv", parse_dates=["Date"], low_memory=False)
store = pd.read_csv(RAW_DATA_DIR / "store.csv", low_memory=False)
store_states = pd.read_csv(RAW_DATA_DIR / "store_states.csv")
weather = pd.read_csv(EXTERNAL_DATA_DIR / "weather_daytime.csv", parse_dates=["Date"])

store = store.merge(store_states, on="Store", how="left")
df = train.merge(store, on="Store", how="left")
df = df[df["Open"] != 0].copy()
df["Date"] = pd.to_datetime(df["Date"])

print("2. Creating baseline features...")
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day
df["dayofweek"] = df["Date"].dt.dayofweek
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["IsMonthEnd"] = (df["day"] >= 25).astype(int)
df["IsFriday"] = (df["dayofweek"] == 4).astype(int)

df = pd.get_dummies(
    df,
    columns=["StateHoliday", "StoreType", "Assortment", "PromoInterval"],
    dummy_na=True,
)

df = df.merge(weather, on=["Date", "State"], how="left")

de_holidays = holidays.Germany(years=[2013, 2014, 2015])
df["PublicHoliday"] = df["Date"].apply(lambda d: 1 if d in de_holidays else 0)

df["TempRange"] = df["temperature_max"] - df["temperature_mean"]
df["IsRainy"] = (df["precipitation_sum"] > 5).astype(int)
df["IsSunny"] = (df["sunshine_sum"] > 21600).astype(int)

df = df.fillna(0)

print("3. Splitting train / valid...")
train_mask = df["Date"] < pd.Timestamp("2015-06-01")
valid_mask = df["Date"] >= pd.Timestamp("2015-06-01")

drop_cols = ["Date", "Customers", "State"]
if "Id" in df.columns:
    drop_cols.append("Id")
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

X_train = df[train_mask].drop("Sales", axis=1)
y_train = df[train_mask]["Sales"]
X_valid = df[valid_mask].drop("Sales", axis=1)
y_valid = df[valid_mask]["Sales"]

print(f"   Train: {X_train.shape}")
print(f"   Valid: {X_valid.shape}")

print("\n4. Training XGBoost baseline...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

params = {
    "max_depth": 5,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "seed": 42,
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=3000,
    evals=[(dtrain, "train"), (dvalid, "valid")],
    early_stopping_rounds=50,
    verbose_eval=100,
)

print("\n5. Prediction...")
pred = model.predict(dvalid)
pred = np.clip(pred, 0, None)

rmspe_val = rmspe(y_valid, pred)
mae_val = mae(y_valid, pred)
r2_val = r2_score(y_valid, pred)

print("\n" + "=" * 50)
print("BASELINE METRICS")
print("=" * 50)
print(f"RMSPE: {rmspe_val:.4f}")
print(f"MAE:   {mae_val:.0f} EUR")
print(f"R2:    {r2_val:.4f}")
print("=" * 50)

model_path = MODELS_DIR / "baseline_xgboost.json"
model.save_model(str(model_path))
feature_columns_path = save_feature_columns(
    MODELS_DIR,
    MODEL_NAME,
    X_train.columns.tolist(),
)
preprocessing_config_path = save_preprocessing_config(
    MODELS_DIR,
    MODEL_NAME,
    {
        "target": "Sales",
        "model": {
            "type": "xgboost.Booster",
            "params": params,
            "num_boost_round": 3000,
            "early_stopping_rounds": 50,
            "best_iteration": int(model.best_iteration),
        },
        "data": {
            "raw_files": ["train.csv", "store.csv", "store_states.csv"],
            "external_files": ["weather_daytime.csv"],
        },
        "split": {
            "train_before": "2015-06-01",
            "valid_from": "2015-06-01",
        },
        "drop_columns": drop_cols,
        "date_features": [
            "year",
            "month",
            "day",
            "dayofweek",
            "WeekOfYear",
            "IsMonthEnd",
            "IsFriday",
        ],
        "one_hot_columns": ["StateHoliday", "StoreType", "Assortment", "PromoInterval"],
        "external_features": {
            "weather": True,
            "google_trends": False,
            "unemployment": False,
        },
        "holiday_features": ["PublicHoliday"],
        "lag_features": [],
        "fillna": 0,
        "metrics": {
            "rmspe": float(rmspe_val),
            "mae": float(mae_val),
            "r2": float(r2_val),
        },
    },
)

print(f"\nOK: Model saved: {model_path}")
print(f"OK: Feature columns saved: {feature_columns_path}")
print(f"OK: Preprocessing config saved: {preprocessing_config_path}")
