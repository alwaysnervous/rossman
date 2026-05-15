from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

common = import_module("src.models.common")


REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
CSV_PATH = REPORTS_DIR / "baseline_metrics.csv"
JSON_PATH = REPORTS_DIR / "baseline_metrics.json"


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    return {
        "rmspe": common.rmspe(y_true, np.asarray(y_pred)),
        "mae": common.mae(y_true, np.asarray(y_pred)),
        "r2": common.r2(y_true, np.asarray(y_pred)),
    }


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["Date"] < common.TRAIN_CUTOFF].copy()
    valid = df[df["Date"] >= common.TRAIN_CUTOFF].copy()
    return train, valid


def fill_with_fallback(
    values: pd.Series,
    fallback: pd.Series | float,
    global_mean: float,
) -> pd.Series:
    return values.fillna(fallback).fillna(global_mean)


def predict_store_mean(train: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    global_mean = float(train["Sales"].mean())
    store_mean = train.groupby("Store")["Sales"].mean()
    pred = valid["Store"].map(store_mean)
    return fill_with_fallback(pred, global_mean, global_mean)


def predict_store_dayofweek_mean(train: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    global_mean = float(train["Sales"].mean())
    store_mean = train.groupby("Store")["Sales"].mean()
    store_day_mean = train.groupby(["Store", "DayOfWeek"])["Sales"].mean()

    index = pd.MultiIndex.from_frame(valid[["Store", "DayOfWeek"]])
    pred = pd.Series(store_day_mean.reindex(index).to_numpy(), index=valid.index)
    fallback = valid["Store"].map(store_mean)
    return fill_with_fallback(pred, fallback, global_mean)


def build_baseline_rows() -> list[dict[str, float | str]]:
    df = common.load_base_frame()
    df_with_lags = common.add_lag_features(df)
    train, valid = split_frame(df_with_lags)
    global_mean = float(train["Sales"].mean())

    predictions = {
        "Naive Lag-1": valid["Sales_Lag1"].fillna(global_mean),
        "Naive Lag-7": valid["Sales_Lag7"].fillna(global_mean),
        "Store Mean": predict_store_mean(train, valid),
        "Store + DayOfWeek Mean": predict_store_dayofweek_mean(train, valid),
    }

    rows = []
    for model_name, pred in predictions.items():
        metrics = calculate_metrics(valid["Sales"], pred)
        rows.append(
            {
                "model": model_name,
                "rmspe": metrics["rmspe"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )

    return rows


def main() -> None:
    rows = build_baseline_rows()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame(rows).sort_values("rmspe")
    result.to_csv(CSV_PATH, index=False)
    result.to_json(JSON_PATH, orient="records", force_ascii=False, indent=2)

    print("Базовые прогнозы:")
    print(
        result.to_string(
            index=False,
            formatters={
                "rmspe": "{:.4f}".format,
                "mae": "{:.0f}".format,
                "r2": "{:.4f}".format,
            },
        )
    )
    print(f"\nCSV: {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
