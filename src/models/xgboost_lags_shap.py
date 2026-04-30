import pandas as pd
import numpy as np
import xgboost as xgb
import holidays
import shap
import matplotlib
from pathlib import Path
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from src.artifacts import save_feature_columns, save_preprocessing_config

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
EXTERNAL_DATA_DIR = ROOT_DIR / "data" / "external"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
MODELS_DIR = ROOT_DIR / "outputs" / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("1. Загрузка данных...")
train = pd.read_csv(RAW_DATA_DIR / 'train.csv', parse_dates=['Date'], low_memory=False)
store = pd.read_csv(RAW_DATA_DIR / 'store.csv', low_memory=False)
store_states = pd.read_csv(RAW_DATA_DIR / 'store_states.csv')
weather = pd.read_csv(EXTERNAL_DATA_DIR / 'weather_daytime.csv', parse_dates=['Date'])

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

# One-hot
df = pd.get_dummies(
    df, columns=['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'],
    dummy_na=True
)

# Погода
df = df.merge(weather, on=['Date', 'State'], how='left')

# Праздники
de_holidays = holidays.Germany(years=[2013, 2014, 2015])
df['PublicHoliday'] = df['Date'].apply(lambda d: 1 if d in de_holidays else 0)

# Производные погодные
df['TempRange'] = df['temperature_max'] - df['temperature_mean']
df['IsRainy'] = (df['precipitation_sum'] > 5).astype(int)
df['IsSunny'] = (df['sunshine_sum'] > 21600).astype(int)

# ========== ЛАГОВЫЕ ПРИЗНАКИ ==========
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

print(f"\nRMSPE (погода + праздники + лаги): {rmspe_val:.4f}")
print(f"Baseline RMSPE: {baseline_rmspe:.4f}")
print(f"Улучшение: {improvement:.2f}%")


# ========== МЕТРИКИ ==========
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)


mae_val = mae(y_valid, pred)
r2_val = r2(y_valid, pred)

print(f"\nМЕТРИКИ XGBoost:")
print(f"   RMSPE: {rmspe_val:.4f}")
print(f"   MAE:   {mae_val:.0f} €")
print(f"   R2:    {r2_val:.4f}")

model_path = MODELS_DIR / "xgboost_lags_shap.json"
model.save_model(str(model_path))
feature_columns_path = save_feature_columns(
    MODELS_DIR,
    "xgboost_lags_shap",
    X_train.columns.tolist()
)
preprocessing_config_path = save_preprocessing_config(
    MODELS_DIR,
    "xgboost_lags_shap",
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
        "date_features": ["year", "month", "day", "dayofweek", "WeekOfYear"],
        "one_hot_columns": ["StateHoliday", "StoreType", "Assortment", "PromoInterval"],
        "external_features": {
            "weather": True,
            "google_trends": False,
            "unemployment": False,
        },
        "holiday_features": ["PublicHoliday"],
        "lag_features": ["Sales_Lag1", "Sales_Lag7", "Sales_Rolling7"],
        "fillna": 0,
        "metrics": {
            "rmspe": float(rmspe_val),
            "mae": float(mae_val),
            "r2": float(r2_val),
        },
    }
)

print(f"\nOK: Модель сохранена: {model_path}")
print(f"OK: Колонки признаков сохранены: {feature_columns_path}")
print(f"OK: Конфиг предобработки сохранён: {preprocessing_config_path}")

# ========== SHAP-АНАЛИЗ ==========
print("\n2. SHAP-анализ...")

# Берём выборку для ускорения
sample_size = min(500, len(X_valid))
sample_idx = np.random.choice(len(X_valid), sample_size, replace=False)
X_sample = X_valid.iloc[sample_idx]

# XGBoost can calculate SHAP contributions directly. This avoids a
# shap/xgboost version mismatch around parsing Booster base_score.
dsample = xgb.DMatrix(X_sample, feature_names=X_train.columns.tolist())
shap_contribs = model.predict(dsample, pred_contribs=True)
shap_values = shap_contribs[:, :-1]  # Last column is the bias term.

# SHAP summary plot (показывает влияние каждого признака)
print("   Строим SHAP summary plot...")
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
plt.title('SHAP — Влияние признаков на прогноз XGBoost', fontsize=14)
plt.tight_layout()
shap_summary_path = FIGURES_DIR / 'shap_summary.png'
plt.savefig(shap_summary_path, dpi=150)
plt.close()
print(f"   Сохранено: {shap_summary_path}")

# SHAP bar plot (средняя важность)
print("   Строим SHAP bar plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False, max_display=15)
plt.title('SHAP — Средняя важность признаков XGBoost', fontsize=14)
plt.tight_layout()
shap_bar_path = FIGURES_DIR / 'shap_bar.png'
plt.savefig(shap_bar_path, dpi=150)
plt.close()
print(f"   Сохранено: {shap_bar_path}")

print("\nOK: Всё готово!")
