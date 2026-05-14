from argparse import ArgumentParser
from dataclasses import asdict
from importlib import import_module
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

common = import_module("src.models.common")


REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
DEFAULT_CSV_PATH = REPORTS_DIR / "feature_variant_metrics.csv"
DEFAULT_JSON_PATH = REPORTS_DIR / "feature_variant_metrics.json"


def parse_args():
    parser = ArgumentParser(
        description=(
            "Перебор вариантов признаков для Linear Regression, "
            "Random Forest и XGBoost "
            "с расчётом RMSPE, MAE и R2."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["linear_regression", "random_forest", "xgboost"],
        default=["linear_regression", "random_forest", "xgboost"],
        help="Какие модели запускать.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Имена конкретных вариантов. Список имён можно увидеть через --dry-run.",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Ограничить число вариантов признаков. Удобно для быстрой проверки.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать список экспериментов без обучения моделей.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять CSV/JSON с результатами.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Куда сохранить таблицу метрик.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help="Куда сохранить подробный JSON с метриками и колонками.",
    )
    parser.add_argument("--rf-n-estimators", type=int, default=100)
    parser.add_argument("--rf-max-depth", type=int, default=15)
    parser.add_argument("--rf-n-jobs", type=int, default=-1)
    parser.add_argument("--lr-n-jobs", type=int, default=-1)
    parser.add_argument("--xgb-num-boost-round", type=int, default=3000)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=50)
    parser.add_argument(
        "--xgb-verbose-eval",
        type=int,
        default=0,
        help="0 отключает подробный лог XGBoost; 100 печатает каждые 100 итераций.",
    )
    return parser.parse_args()


def select_variants(args) -> list:
    variants = common.iter_feature_variants()
    if args.variants:
        by_name = {variant.name: variant for variant in variants}
        unknown = sorted(set(args.variants) - set(by_name))
        if unknown:
            raise ValueError(f"Неизвестные варианты признаков: {unknown}")
        variants = [by_name[name] for name in args.variants]

    if args.max_experiments is not None:
        variants = variants[: args.max_experiments]

    return variants


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmspe": common.rmspe(y_true, y_pred),
        "mae": common.mae(y_true, y_pred),
        "r2": common.r2(y_true, y_pred),
    }


def train_linear_regression(data, args):
    params = {"n_jobs": args.lr_n_jobs}
    model = LinearRegression(**params)
    model.fit(data.X_train, data.y_train)
    pred = np.clip(model.predict(data.X_valid), 0, None)
    return calculate_metrics(data.y_valid, pred), params


def train_random_forest(data, args):
    params = {
        "n_estimators": args.rf_n_estimators,
        "max_depth": args.rf_max_depth,
        "random_state": 42,
        "n_jobs": args.rf_n_jobs,
    }
    model = RandomForestRegressor(**params)
    model.fit(data.X_train, data.y_train)
    pred = np.clip(model.predict(data.X_valid), 0, None)
    return calculate_metrics(data.y_valid, pred), params


def train_xgboost(data, args):
    params = {
        "max_depth": 5,
        "learning_rate": 0.05,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
    }
    dtrain = xgb.DMatrix(data.X_train, label=data.y_train)
    dvalid = xgb.DMatrix(data.X_valid, label=data.y_valid)
    verbose_eval = args.xgb_verbose_eval if args.xgb_verbose_eval > 0 else False
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.xgb_num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=args.xgb_early_stopping_rounds,
        verbose_eval=verbose_eval,
    )
    pred = np.clip(model.predict(dvalid), 0, None)
    params_with_training = {
        **params,
        "num_boost_round": args.xgb_num_boost_round,
        "early_stopping_rounds": args.xgb_early_stopping_rounds,
        "best_iteration": int(model.best_iteration),
    }
    return calculate_metrics(data.y_valid, pred), params_with_training


def result_key(result: dict) -> tuple[str, str]:
    return result["model"], result["variant"]


def read_existing_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []

    return pd.read_csv(csv_path).to_dict("records")


def read_existing_details(json_path: Path) -> list[dict]:
    if not json_path.exists():
        return []

    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_results(rows: list[dict], details: list[dict], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    current_keys = {result_key(row) for row in rows}
    existing_rows = [
        row for row in read_existing_rows(csv_path) if result_key(row) not in current_keys
    ]
    existing_details = [
        detail
        for detail in read_existing_details(json_path)
        if result_key(detail) not in current_keys
    ]

    merged_rows = existing_rows + rows
    merged_details = existing_details + details

    pd.DataFrame(merged_rows).sort_values(["model", "rmspe"]).to_csv(
        csv_path,
        index=False,
    )
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(merged_details, f, ensure_ascii=False, indent=2)
        f.write("\n")


def print_dry_run(models: list[str], variants: list) -> None:
    print("Модели:", ", ".join(models))
    print(f"Вариантов признаков: {len(variants)}")
    print(f"Всего запусков: {len(models) * len(variants)}")
    print("\nВарианты:")
    for i, variant in enumerate(variants, start=1):
        groups = ", ".join(variant.enabled_groups()) or "только базовые признаки"
        print(f"{i:02d}. {variant.name}: {groups}")


def main() -> None:
    args = parse_args()
    variants = select_variants(args)

    if args.dry_run:
        print_dry_run(args.models, variants)
        return

    rows = []
    details = []
    total_runs = len(args.models) * len(variants)
    run_number = 0

    print("=" * 80)
    print("Перебор вариантов признаков: Linear Regression, Random Forest и XGBoost")
    print("=" * 80)
    print(f"Вариантов признаков: {len(variants)}")
    print(f"Всего запусков: {total_runs}")

    for variant in variants:
        print("\n" + "-" * 80)
        print(f"Вариант признаков: {variant.name}")
        print("Группы:", ", ".join(variant.enabled_groups()) or "только базовые признаки")
        data = common.prepare_feature_variant_data(variant)
        metadata = common.feature_variant_metadata(variant)
        print(f"Train: {data.X_train.shape}, Valid: {data.X_valid.shape}")

        for model_name in args.models:
            run_number += 1
            started_at = perf_counter()
            print(f"\n[{run_number}/{total_runs}] Модель: {model_name}")

            if model_name == "linear_regression":
                metrics, params = train_linear_regression(data, args)
            elif model_name == "random_forest":
                metrics, params = train_random_forest(data, args)
            else:
                metrics, params = train_xgboost(data, args)

            elapsed_seconds = perf_counter() - started_at
            row = {
                "model": model_name,
                "variant": variant.name,
                "n_features": data.X_train.shape[1],
                "train_rows": data.X_train.shape[0],
                "valid_rows": data.X_valid.shape[0],
                "weather": variant.weather,
                "google_trends": variant.google_trends,
                "unemployment": variant.unemployment,
                "holidays": variant.holidays,
                "promo_weather": variant.promo_weather,
                "lags": variant.lags,
                "rmspe": metrics["rmspe"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "elapsed_seconds": elapsed_seconds,
            }
            rows.append(row)
            details.append(
                {
                    **row,
                    "variant_config": asdict(variant),
                    "model_params": params,
                    "feature_columns": data.X_train.columns.tolist(),
                    "drop_columns": data.drop_columns,
                    **metadata,
                }
            )

            print(
                "RMSPE: {rmspe:.4f} | MAE: {mae:.0f} EUR | R2: {r2:.4f} | "
                "time: {time:.1f}s".format(
                    rmspe=metrics["rmspe"],
                    mae=metrics["mae"],
                    r2=metrics["r2"],
                    time=elapsed_seconds,
                )
            )

            if not args.no_save:
                save_results(rows, details, args.output_csv, args.output_json)

    if not args.no_save:
        print("\n" + "=" * 80)
        print(f"OK: CSV сохранён: {args.output_csv}")
        print(f"OK: JSON сохранён: {args.output_json}")

    best = pd.DataFrame(rows).sort_values("rmspe").head(10)
    print("\nТоп-10 экспериментов по RMSPE:")
    print(best[["model", "variant", "n_features", "rmspe", "mae", "r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
