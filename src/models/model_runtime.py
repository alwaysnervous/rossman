from importlib import import_module
from pathlib import Path
import sys

import matplotlib


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

matplotlib.use("Agg")
plt = import_module("matplotlib.pyplot")
common = import_module("src.models.common")

FIGURES_DIR = ROOT_DIR / "outputs" / "figures"


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    common.MODELS_DIR.mkdir(parents=True, exist_ok=True)


def prepare_logged_variant_data(variant):
    print("\n1. Загрузка данных...")
    print(f"   Вариант признаков: {variant.name}")
    print("2. Создание признаков...")
    data = common.prepare_feature_variant_data(variant)

    print("3. Разделение на train / valid...")
    print(f"   Train: {data.X_train.shape}")
    print(f"   Valid: {data.X_valid.shape}")

    return data


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmspe": common.rmspe(y_true, y_pred),
        "mae": common.mae(y_true, y_pred),
        "r2": common.r2(y_true, y_pred),
    }


def print_metrics_block(model_label: str, metrics: dict[str, float]) -> None:
    print("\n" + "=" * 50)
    print(f"МЕТРИКИ {model_label}")
    print("=" * 50)
    print(f"RMSPE: {metrics['rmspe']:.4f}")
    print(f"MAE:   {metrics['mae']:.0f} €")
    print(f"R2:    {metrics['r2']:.4f}")
    print("=" * 50)


def save_sklearn_model_outputs(
    *,
    model,
    model_file: str,
    model_name: str,
    model_type: str,
    model_params: dict,
    variant,
    data,
    metrics: dict[str, float],
):
    joblib = import_module("joblib")
    model_path = common.MODELS_DIR / model_file
    joblib.dump(model, model_path)

    metadata = common.feature_variant_metadata(variant)
    feature_columns_path, preprocessing_config_path = common.save_sklearn_artifacts(
        model_name=model_name,
        feature_columns=data.X_train.columns.tolist(),
        drop_columns=data.drop_columns,
        model_type=model_type,
        model_params=model_params,
        external_files=metadata["external_files"],
        external_features=metadata["external_features"],
        holiday_features=metadata["holiday_features"],
        lag_features=metadata["lag_features"],
        metrics=metrics,
        feature_variant=metadata["feature_variant"],
    )

    print(f"\nOK: Модель сохранена: {model_path}")
    print(f"OK: Колонки признаков сохранены: {feature_columns_path}")
    print(f"OK: Конфиг предобработки сохранён: {preprocessing_config_path}")


def print_top_values(series, header: str, value_format: str) -> None:
    print(f"\n{header}:")
    for name, val in series.head(15).items():
        print(f"   {name:<35} {value_format.format(val)}")


def save_top_bar_plot(
    series,
    *,
    title: str | None,
    output_filename: str,
    xlabel: str | None = None,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.barh(range(15), series.head(15).values[::-1])
    plt.yticks(range(15), series.head(15).index[::-1])
    if title:
        plt.title(title, fontsize=14)
    if xlabel:
        plt.xlabel(xlabel)
    plt.tight_layout()
    importance_path = FIGURES_DIR / output_filename
    plt.savefig(importance_path, dpi=150)
    plt.close()

    print(f"\nOK: График сохранён: {importance_path}")


__all__ = [
    "FIGURES_DIR",
    "ROOT_DIR",
    "common",
    "ensure_output_dirs",
    "matplotlib",
    "plt",
    "prepare_logged_variant_data",
    "print_metrics_block",
    "print_top_values",
    "save_sklearn_model_outputs",
    "save_top_bar_plot",
]
