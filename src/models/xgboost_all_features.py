import pandas as pd
import numpy as np
import xgboost as xgb
import holidays
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from src.artifacts import save_feature_columns, save_preprocessing_config

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
EXTERNAL_DATA_DIR = ROOT_DIR / "data" / "external"
MODELS_DIR = ROOT_DIR / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("1. Загрузка всех данных...")
train = pd.read_csv(RAW_DATA_DIR / 'train.csv', parse_dates=['Date'], low_memory=False)
store = pd.read_csv(RAW_DATA_DIR / 'store.csv', low_memory=False)
store_states = pd.read_csv(RAW_DATA_DIR / 'store_states.csv')
weather = pd.read_csv(EXTERNAL_DATA_DIR / 'weather_daytime.csv', parse_dates=['Date'])
googletrend = pd.read_csv(EXTERNAL_DATA_DIR / 'googletrend_self.csv', parse_dates=['Date'])
unemployment = pd.read_csv(EXTERNAL_DATA_DIR / 'unemployment_germany.csv', parse_dates=['Date'])

store = store.merge(store_states, on='Store', how='left')
df = train.merge(store, on='Store', how='left')
df = df[df['Open'] != 0].copy()
df['Date'] = pd.to_datetime(df['Date'])

# Признаки из даты
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['IsMonthEnd'] = (df['day'] >= 25).astype(int)
df['IsFriday'] = (df['dayofweek'] == 4).astype(int)

# One-hot
df = pd.get_dummies(
    df, columns=['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'],
    dummy_na=True
)

# Погода
df = df.merge(weather, on=['Date', 'State'], how='left')

# Google Trends
df = df.merge(googletrend, on=['Date', 'State'], how='left')
df = df.rename(columns={'trend': 'GoogleTrend'})

# Безработица
df['YearMonth'] = df['Date'].dt.to_period('M')
unemployment['YearMonth'] = unemployment['Date'].dt.to_period('M')
df = df.merge(unemployment[['YearMonth', 'UnemploymentRate']], on='YearMonth', how='left')
df = df.drop(columns=['YearMonth'])

# Праздники
de_holidays = holidays.Germany(years=[2013, 2014, 2015])
df['PublicHoliday'] = df['Date'].apply(lambda d: 1 if d in de_holidays else 0)
df['BeforeHoliday'] = df['Date'].apply(lambda d: 1 if (d + pd.Timedelta(days=1)) in de_holidays else 0)
df['AfterHoliday'] = df['Date'].apply(lambda d: 1 if (d - pd.Timedelta(days=1)) in de_holidays else 0)

# Производные погодные
df['TempRange'] = df['temperature_max'] - df['temperature_mean']
df['IsRainy'] = (df['precipitation_sum'] > 5).astype(int)
df['IsSunny'] = (df['sunshine_sum'] > 21600).astype(int)
df['IsHot'] = (df['temperature_max'] > 30).astype(int)
df['IsCold'] = (df['temperature_mean'] < -5).astype(int)

# Взаимодействия
df['Promo_Temperature'] = df['Promo'] * df['temperature_mean']
df['Promo_Sunny'] = df['Promo'] * df['IsSunny']
df['Promo_Rainy'] = df['Promo'] * df['IsRainy']

# ========== ЛАГИ ==========
df = df.sort_values(['Store', 'Date'])
df['Sales_Lag1'] = df.groupby('Store')['Sales'].shift(1)
df['Sales_Lag7'] = df.groupby('Store')['Sales'].shift(7)
df['Sales_Rolling7'] = df.groupby('Store')['Sales'].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean()
)

df = df.fillna(0)

# Разделение
train_mask = df['Date'] < pd.Timestamp('2015-06-01')
valid_mask = df['Date'] >= pd.Timestamp('2015-06-01')

drop_cols = ['Date', 'Customers', 'State']
if 'Id' in df.columns:
    drop_cols.append('Id')
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

X_train = df[train_mask].drop('Sales', axis=1)
y_train = df[train_mask]['Sales']
X_valid = df[valid_mask].drop('Sales', axis=1)
y_valid = df[valid_mask]['Sales']

print(f"Train: {X_train.shape}, Valid: {X_valid.shape}")

# Обучение
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

params = {
    'max_depth': 5, 'learning_rate': 0.05,
    'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42
}

model = xgb.train(params, dtrain, num_boost_round=3000,
                  evals=[(dtrain, 'train'), (dvalid, 'valid')],
                  early_stopping_rounds=50, verbose_eval=100)

pred = model.predict(dvalid)
pred = np.clip(pred, 0, None)


def rmspe(y_true, y_pred):
    mask = y_true > 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))


baseline_rmspe = 0.1720
rmspe_val = rmspe(y_valid, pred)
improvement = (baseline_rmspe - rmspe_val) / baseline_rmspe * 100

print("\n" + "=" * 60)
print(f"RMSPE (ВСЕ признаки): {rmspe_val:.4f}")
print(f"Baseline RMSPE:       {baseline_rmspe:.4f}")
print(f"Улучшение:            {improvement:.2f}%")
print("=" * 60)

model_path = MODELS_DIR / "xgboost_all_features.json"
model.save_model(str(model_path))
feature_columns_path = save_feature_columns(
    MODELS_DIR,
    "xgboost_all_features",
    X_train.columns.tolist()
)
preprocessing_config_path = save_preprocessing_config(
    MODELS_DIR,
    "xgboost_all_features",
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
            "external_files": [
                "weather_daytime.csv",
                "googletrend_self.csv",
                "unemployment_germany.csv",
            ],
        },
        "split": {
            "train_before": "2015-06-01",
            "valid_from": "2015-06-01",
        },
        "drop_columns": drop_cols,
        "date_features": [
            "year", "month", "day", "dayofweek", "WeekOfYear",
            "IsMonthEnd", "IsFriday",
        ],
        "one_hot_columns": ["StateHoliday", "StoreType", "Assortment", "PromoInterval"],
        "external_features": {
            "weather": True,
            "google_trends": True,
            "unemployment": True,
        },
        "holiday_features": ["PublicHoliday", "BeforeHoliday", "AfterHoliday"],
        "lag_features": ["Sales_Lag1", "Sales_Lag7", "Sales_Rolling7"],
        "fillna": 0,
        "metrics": {
            "rmspe": float(rmspe_val),
        },
    }
)

print(f"\nOK: Модель сохранена: {model_path}")
print(f"OK: Колонки признаков сохранены: {feature_columns_path}")
print(f"OK: Конфиг предобработки сохранён: {preprocessing_config_path}")
