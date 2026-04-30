import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
import holidays
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

print("=" * 60)
print("МОДЕЛЬ 3: Linear Regression (линейная регрессия)")
print("=" * 60)

# ========== 1. ЗАГРУЗКА ==========
print("\n1. Загрузка данных...")
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

# ========== 2. ПРИЗНАКИ ==========
print("2. Создание признаков...")

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['IsMonthEnd'] = (df['day'] >= 25).astype(int)
df['IsFriday'] = (df['dayofweek'] == 4).astype(int)

df = pd.get_dummies(
    df, columns=['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'],
    dummy_na=True
)

df = df.merge(weather, on=['Date', 'State'], how='left')
df = df.merge(googletrend, on=['Date', 'State'], how='left')
df = df.rename(columns={'trend': 'GoogleTrend'})

df['YearMonth'] = df['Date'].dt.to_period('M')
unemployment['YearMonth'] = unemployment['Date'].dt.to_period('M')
df = df.merge(unemployment[['YearMonth', 'UnemploymentRate']], on='YearMonth', how='left')
df = df.drop(columns=['YearMonth'])

de_holidays = holidays.Germany(years=[2013, 2014, 2015])
df['PublicHoliday'] = df['Date'].apply(lambda d: 1 if d in de_holidays else 0)
df['BeforeHoliday'] = df['Date'].apply(lambda d: 1 if (d + pd.Timedelta(days=1)) in de_holidays else 0)
df['AfterHoliday'] = df['Date'].apply(lambda d: 1 if (d - pd.Timedelta(days=1)) in de_holidays else 0)

df['TempRange'] = df['temperature_max'] - df['temperature_mean']
df['IsRainy'] = (df['precipitation_sum'] > 5).astype(int)
df['IsSunny'] = (df['sunshine_sum'] > 21600).astype(int)
df['IsHot'] = (df['temperature_max'] > 30).astype(int)
df['IsCold'] = (df['temperature_mean'] < -5).astype(int)
df['Promo_Temperature'] = df['Promo'] * df['temperature_mean']
df['Promo_Sunny'] = df['Promo'] * df['IsSunny']
df['Promo_Rainy'] = df['Promo'] * df['IsRainy']

df = df.sort_values(['Store', 'Date'])
df['Sales_Lag1'] = df.groupby('Store')['Sales'].shift(1)
df['Sales_Lag7'] = df.groupby('Store')['Sales'].shift(7)
df['Sales_Rolling7'] = df.groupby('Store')['Sales'].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean()
)

df = df.fillna(0)

# ========== 3. РАЗДЕЛЕНИЕ ==========
print("3. Разделение на train / valid...")
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

print(f"   Train: {X_train.shape}")
print(f"   Valid: {X_valid.shape}")

# ========== 4. ОБУЧЕНИЕ ==========
print("\n4. Обучение Linear Regression...")
model_lr = LinearRegression(n_jobs=-1)
model_lr.fit(X_train, y_train)

# ========== 5. ПРЕДСКАЗАНИЕ ==========
print("5. Предсказание...")
pred_lr = model_lr.predict(X_valid)
pred_lr = np.clip(pred_lr, 0, None)  # убираем отрицательные прогнозы

# ========== 6. ВСЕ МЕТРИКИ ==========
from sklearn.metrics import r2_score


def rmspe(y_true, y_pred):
    mask = y_true > 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


rmspe_lr = rmspe(y_valid, pred_lr)
mae_lr = mae(y_valid, pred_lr)
r2_lr = r2_score(y_valid, pred_lr)

print("\n" + "=" * 50)
print("МЕТРИКИ Linear Regression")
print("=" * 50)
print(f"RMSPE: {rmspe_lr:.4f}")
print(f"MAE:   {mae_lr:.0f} €")
print(f"R2:    {r2_lr:.4f}")
print("=" * 50)

model_path = MODELS_DIR / "linear_regression.pkl"
joblib.dump(model_lr, model_path)
feature_columns_path = save_feature_columns(
    MODELS_DIR,
    "linear_regression",
    X_train.columns.tolist()
)
preprocessing_config_path = save_preprocessing_config(
    MODELS_DIR,
    "linear_regression",
    {
        "target": "Sales",
        "model": {
            "type": "LinearRegression",
            "params": {"n_jobs": -1},
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
            "rmspe": float(rmspe_lr),
            "mae": float(mae_lr),
            "r2": float(r2_lr),
        },
    }
)

print(f"\nOK: Модель сохранена: {model_path}")
print(f"OK: Колонки признаков сохранены: {feature_columns_path}")
print(f"OK: Конфиг предобработки сохранён: {preprocessing_config_path}")

# ========== 7. КОЭФФИЦИЕНТЫ (важность) ==========
print("\n6. Топ-15 коэффициентов...")
coef = pd.Series(np.abs(model_lr.coef_), index=X_train.columns).sort_values(ascending=False)

print("\nТоп-15 по модулю коэффициентов:")
for name, val in coef.head(15).items():
    print(f"   {name:<35} {val:.0f} €")

plt.figure(figsize=(10, 6))
plt.barh(range(15), coef.head(15).values[::-1])
plt.yticks(range(15), coef.head(15).index[::-1])
plt.title('Linear Regression — Топ-15 |коэффициентов|', fontsize=14)
plt.tight_layout()
importance_path = FIGURES_DIR / 'importance_lr.png'
plt.savefig(importance_path, dpi=150)
plt.close()

print(f"\nOK: График сохранён: {importance_path}")
