from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from holidays.countries import Germany
from sklearn.metrics import r2_score

from src.artifacts import save_feature_columns, save_preprocessing_config


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
EXTERNAL_DATA_DIR = ROOT_DIR / "data" / "external"
MODELS_DIR = ROOT_DIR / "outputs" / "models"

TRAIN_CUTOFF = pd.Timestamp("2015-06-01")
DATE_FEATURES = [
    "year",
    "month",
    "day",
    "dayofweek",
    "WeekOfYear",
    "IsMonthEnd",
    "IsFriday",
]
ONE_HOT_COLUMNS = ["StateHoliday", "StoreType", "Assortment", "PromoInterval"]
DROP_COLUMNS = ["Date", "Customers", "State"]


@dataclass(frozen=True)
class TrainValidData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    drop_columns: list[str]


@dataclass(frozen=True)
class FeatureVariant:
    name: str
    weather: bool = False
    google_trends: bool = False
    unemployment: bool = False
    holidays: bool = False
    promo_weather: bool = False
    lags: bool = False

    def enabled_groups(self) -> list[str]:
        return [
            name
            for name in [
                "weather",
                "google_trends",
                "unemployment",
                "holidays",
                "promo_weather",
                "lags",
            ]
            if getattr(self, name)
        ]


def make_feature_variant(
    *,
    weather: bool = False,
    google_trends: bool = False,
    unemployment: bool = False,
    holidays: bool = False,
    promo_weather: bool = False,
    lags: bool = False,
) -> FeatureVariant:
    if promo_weather and not weather:
        raise ValueError("promo_weather requires weather=True")

    enabled = [
        name
        for name, is_enabled in [
            ("weather", weather),
            ("google_trends", google_trends),
            ("unemployment", unemployment),
            ("holidays", holidays),
            ("promo_weather", promo_weather),
            ("lags", lags),
        ]
        if is_enabled
    ]
    name = "base" if not enabled else "__".join(enabled)
    return FeatureVariant(
        name=name,
        weather=weather,
        google_trends=google_trends,
        unemployment=unemployment,
        holidays=holidays,
        promo_weather=promo_weather,
        lags=lags,
    )


def iter_feature_variants() -> list[FeatureVariant]:
    variants = []
    for (
        weather,
        google_trends,
        unemployment,
        holidays,
        promo_weather,
        lags,
    ) in product([False, True], repeat=6):
        if promo_weather and not weather:
            continue

        variants.append(
            make_feature_variant(
                weather=weather,
                google_trends=google_trends,
                unemployment=unemployment,
                holidays=holidays,
                promo_weather=promo_weather,
                lags=lags,
            )
        )

    return variants


def load_base_frame() -> pd.DataFrame:
    train = pd.read_csv(
        RAW_DATA_DIR / "train.csv",
        parse_dates=["Date"],
        low_memory=False,
    )
    store = pd.read_csv(RAW_DATA_DIR / "store.csv", low_memory=False)
    store_states = pd.read_csv(RAW_DATA_DIR / "store_states.csv")

    store = store.merge(store_states, on="Store", how="left")
    df = train.merge(store, on="Store", how="left")
    df = df[df["Open"] != 0].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsMonthEnd"] = (df["day"] >= 25).astype(int)
    df["IsFriday"] = (df["dayofweek"] == 4).astype(int)
    return df


def add_one_hot_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=ONE_HOT_COLUMNS, dummy_na=True)


def add_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    weather = pd.read_csv(
        EXTERNAL_DATA_DIR / "weather_daytime.csv",
        parse_dates=["Date"],
    )
    return df.merge(weather, on=["Date", "State"], how="left")


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TempRange"] = df["temperature_max"] - df["temperature_mean"]
    df["IsRainy"] = (df["precipitation_sum"] > 5).astype(int)
    df["IsSunny"] = (df["sunshine_sum"] > 21600).astype(int)
    return df


def add_full_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_weather_features(df)
    df["IsHot"] = (df["temperature_max"] > 30).astype(int)
    df["IsCold"] = (df["temperature_mean"] < -5).astype(int)
    return df


def _holiday_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    years = sorted(df["Date"].dt.year.unique().tolist())
    return pd.DatetimeIndex(pd.to_datetime(list(Germany(years=years).keys())))


def add_public_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    holiday_dates = _holiday_dates(df)
    df = df.copy()
    df["PublicHoliday"] = df["Date"].dt.normalize().isin(holiday_dates).astype(int)
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    holiday_dates = _holiday_dates(df)
    normalized_date = df["Date"].dt.normalize()
    df = df.copy()
    df["PublicHoliday"] = normalized_date.isin(holiday_dates).astype(int)
    df["BeforeHoliday"] = (
        normalized_date + pd.Timedelta(days=1)
    ).isin(holiday_dates).astype(int)
    df["AfterHoliday"] = (
        normalized_date - pd.Timedelta(days=1)
    ).isin(holiday_dates).astype(int)
    return df


def add_google_trends_feature(df: pd.DataFrame) -> pd.DataFrame:
    googletrend = pd.read_csv(
        EXTERNAL_DATA_DIR / "googletrend_self.csv",
        parse_dates=["Date"],
    )
    df = df.merge(googletrend, on=["Date", "State"], how="left")
    return df.rename(columns={"trend": "GoogleTrend"})


def add_unemployment_feature(df: pd.DataFrame) -> pd.DataFrame:
    unemployment = pd.read_csv(
        EXTERNAL_DATA_DIR / "unemployment_germany.csv",
        parse_dates=["Date"],
    )
    df = df.copy()
    df["YearMonth"] = df["Date"].dt.to_period("M")
    unemployment["YearMonth"] = unemployment["Date"].dt.to_period("M")
    df = df.merge(
        unemployment[["YearMonth", "UnemploymentRate"]],
        on="YearMonth",
        how="left",
    )
    return df.drop(columns=["YearMonth"])


def add_promo_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Promo_Temperature"] = df["Promo"] * df["temperature_mean"]
    df["Promo_Sunny"] = df["Promo"] * df["IsSunny"]
    df["Promo_Rainy"] = df["Promo"] * df["IsRainy"]
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Store", "Date"]).copy()
    df["Sales_Lag1"] = df.groupby("Store")["Sales"].shift(1)
    df["Sales_Lag7"] = df.groupby("Store")["Sales"].shift(7)
    df["Sales_Rolling7"] = df.groupby("Store")["Sales"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    return df


def prepare_baseline_data() -> TrainValidData:
    df = load_base_frame()
    df = add_date_features(df)
    df = add_one_hot_features(df)
    df = add_weather_data(df)
    df = add_public_holiday_feature(df)
    df = add_weather_features(df)
    return split_train_valid(df)


def prepare_all_features_data() -> TrainValidData:
    variant = make_feature_variant(
        weather=True,
        google_trends=True,
        unemployment=True,
        holidays=True,
        promo_weather=True,
        lags=True,
    )
    return prepare_feature_variant_data(variant)


def prepare_feature_variant_data(variant: FeatureVariant) -> TrainValidData:
    df = load_base_frame()
    df = add_date_features(df)
    df = add_one_hot_features(df)

    if variant.weather:
        df = add_weather_data(df)

    if variant.google_trends:
        df = add_google_trends_feature(df)

    if variant.unemployment:
        df = add_unemployment_feature(df)

    if variant.holidays:
        df = add_holiday_features(df)

    if variant.weather:
        df = add_full_weather_features(df)

    if variant.promo_weather:
        df = add_promo_weather_features(df)

    if variant.lags:
        df = add_lag_features(df)

    return split_train_valid(df)


def feature_variant_metadata(variant: FeatureVariant) -> dict[str, Any]:
    external_files = []
    if variant.weather:
        external_files.append("weather_daytime.csv")
    if variant.google_trends:
        external_files.append("googletrend_self.csv")
    if variant.unemployment:
        external_files.append("unemployment_germany.csv")

    return {
        "feature_variant": variant.name,
        "external_files": external_files,
        "external_features": {
            "weather": variant.weather,
            "google_trends": variant.google_trends,
            "unemployment": variant.unemployment,
            "promo_weather_interactions": variant.promo_weather,
        },
        "holiday_features": (
            ["PublicHoliday", "BeforeHoliday", "AfterHoliday"]
            if variant.holidays
            else []
        ),
        "lag_features": (
            ["Sales_Lag1", "Sales_Lag7", "Sales_Rolling7"] if variant.lags else []
        ),
    }


def split_train_valid(df: pd.DataFrame) -> TrainValidData:
    df = df.fillna(0)
    train_mask = df["Date"] < TRAIN_CUTOFF
    valid_mask = df["Date"] >= TRAIN_CUTOFF

    drop_columns = DROP_COLUMNS.copy()
    if "Id" in df.columns:
        drop_columns.append("Id")

    df = df.drop(columns=[col for col in drop_columns if col in df.columns])
    X_train = df.loc[train_mask].drop("Sales", axis=1)
    y_train = df.loc[train_mask, "Sales"]
    X_valid = df.loc[valid_mask].drop("Sales", axis=1)
    y_valid = df.loc[valid_mask, "Sales"]

    return TrainValidData(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        drop_columns=drop_columns,
    )


def rmspe(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    mask = y_true_array > 0
    return float(
        np.sqrt(
            np.mean(
                ((y_true_array[mask] - y_pred_array[mask]) / y_true_array[mask]) ** 2
            )
        )
    )


def mae(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def save_xgboost_artifacts(
    *,
    model: Any,
    model_name: str,
    model_file: str,
    feature_columns: list[str],
    drop_columns: list[str],
    params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
    external_files: list[str],
    external_features: dict[str, bool],
    holiday_features: list[str],
    lag_features: list[str],
    metrics: dict[str, float],
    feature_variant: str | None = None,
) -> tuple[Path, Path, Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / model_file
    model.save_model(str(model_path))

    feature_columns_path = save_feature_columns(
        MODELS_DIR,
        model_name,
        feature_columns,
    )
    preprocessing_config_path = save_preprocessing_config(
        MODELS_DIR,
        model_name,
        {
            "target": "Sales",
            "feature_variant": feature_variant,
            "model": {
                "type": "xgboost.Booster",
                "params": params,
                "num_boost_round": num_boost_round,
                "early_stopping_rounds": early_stopping_rounds,
                "best_iteration": int(model.best_iteration),
            },
            "data": {
                "raw_files": ["train.csv", "store.csv", "store_states.csv"],
                "external_files": external_files,
            },
            "split": {
                "train_before": str(TRAIN_CUTOFF.date()),
                "valid_from": str(TRAIN_CUTOFF.date()),
            },
            "drop_columns": drop_columns,
            "date_features": DATE_FEATURES,
            "one_hot_columns": ONE_HOT_COLUMNS,
            "external_features": external_features,
            "holiday_features": holiday_features,
            "lag_features": lag_features,
            "fillna": 0,
            "metrics": metrics,
        },
    )

    return model_path, feature_columns_path, preprocessing_config_path


def save_sklearn_artifacts(
    *,
    model_name: str,
    feature_columns: list[str],
    drop_columns: list[str],
    model_type: str,
    model_params: dict[str, Any],
    external_files: list[str],
    external_features: dict[str, bool],
    holiday_features: list[str],
    lag_features: list[str],
    metrics: dict[str, float],
    feature_variant: str | None = None,
) -> tuple[Path, Path]:
    feature_columns_path = save_feature_columns(
        MODELS_DIR,
        model_name,
        feature_columns,
    )
    preprocessing_config_path = save_preprocessing_config(
        MODELS_DIR,
        model_name,
        {
            "target": "Sales",
            "feature_variant": feature_variant,
            "model": {
                "type": model_type,
                "params": model_params,
            },
            "data": {
                "raw_files": ["train.csv", "store.csv", "store_states.csv"],
                "external_files": external_files,
            },
            "split": {
                "train_before": str(TRAIN_CUTOFF.date()),
                "valid_from": str(TRAIN_CUTOFF.date()),
            },
            "drop_columns": drop_columns,
            "date_features": DATE_FEATURES,
            "one_hot_columns": ONE_HOT_COLUMNS,
            "external_features": external_features,
            "holiday_features": holiday_features,
            "lag_features": lag_features,
            "fillna": 0,
            "metrics": metrics,
        },
    )

    return feature_columns_path, preprocessing_config_path
