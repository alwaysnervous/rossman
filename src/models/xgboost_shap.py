import numpy as np
import shap
import xgboost as xgb
from matplotlib.text import Text

import model_runtime as runtime


MODEL_NAME = "xgboost_shap_holidays_lags"
MODEL_FILE = "xgboost_shap_holidays_lags.json"
VARIANT = runtime.common.make_feature_variant(holidays=True, lags=True)
PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "seed": 42,
}
NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 50
SHAP_SAMPLE_SIZE = 500


def localize_shap_summary_plot() -> None:
    runtime.plt.xlabel("SHAP-значение")
    figure = runtime.plt.gcf()
    for text in figure.findobj(match=Text):
        if text.get_text() == "High":
            text.set_text("Высокое")
        elif text.get_text() == "Low":
            text.set_text("Низкое")

    if len(figure.axes) > 1:
        colorbar_axis = figure.axes[-1]
        colorbar_axis.set_ylabel("Значение признака")
        colorbar_axis.set_yticks(colorbar_axis.get_yticks())
        colorbar_axis.set_yticklabels(["Низкое", "Высокое"])


def train_xgboost_shap() -> None:
    runtime.ensure_output_dirs()

    print("1. Загрузка данных...")
    print(f"Вариант признаков: {VARIANT.name}")
    data = runtime.common.prepare_feature_variant_data(VARIANT)
    print(f"Train: {data.X_train.shape}, Valid: {data.X_valid.shape}")

    print("\n2. Обучение XGBoost...")
    dtrain = xgb.DMatrix(data.X_train, label=data.y_train)
    dvalid = xgb.DMatrix(data.X_valid, label=data.y_valid)

    model = xgb.train(
        PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=100,
    )

    print("\n3. Предсказание и метрики...")
    pred = np.clip(model.predict(dvalid), 0, None)
    rmspe_val = runtime.common.rmspe(data.y_valid, pred)
    mae_val = runtime.common.mae(data.y_valid, pred)
    r2_val = runtime.common.r2(data.y_valid, pred)

    print("\nМЕТРИКИ XGBoost:")
    print(f"   RMSPE: {rmspe_val:.4f}")
    print(f"   MAE:   {mae_val:.0f} EUR")
    print(f"   R2:    {r2_val:.4f}")

    metadata = runtime.common.feature_variant_metadata(VARIANT)
    model_path, feature_columns_path, preprocessing_config_path = (
        runtime.common.save_xgboost_artifacts(
            model=model,
            model_name=MODEL_NAME,
            model_file=MODEL_FILE,
            feature_columns=data.X_train.columns.tolist(),
            drop_columns=data.drop_columns,
            params=PARAMS,
            num_boost_round=NUM_BOOST_ROUND,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            external_files=metadata["external_files"],
            external_features=metadata["external_features"],
            holiday_features=metadata["holiday_features"],
            lag_features=metadata["lag_features"],
            metrics={
                "rmspe": rmspe_val,
                "mae": mae_val,
                "r2": r2_val,
            },
            feature_variant=metadata["feature_variant"],
        )
    )

    print(f"\nOK: модель сохранена: {model_path}")
    print(f"OK: колонки признаков сохранены: {feature_columns_path}")
    print(f"OK: конфиг предобработки сохранён: {preprocessing_config_path}")

    print("\n4. SHAP-анализ...")
    rng = np.random.default_rng(42)
    sample_size = min(SHAP_SAMPLE_SIZE, len(data.X_valid))
    sample_idx = rng.choice(len(data.X_valid), sample_size, replace=False)
    X_sample = data.X_valid.iloc[sample_idx]

    dsample = xgb.DMatrix(X_sample, feature_names=data.X_train.columns.tolist())
    shap_contribs = model.predict(dsample, pred_contribs=True)
    shap_values = shap_contribs[:, :-1]

    print("   Строим SHAP summary plot...")
    runtime.plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    localize_shap_summary_plot()
    runtime.plt.tight_layout()
    shap_summary_path = runtime.FIGURES_DIR / "shap_summary_holidays_lags.png"
    runtime.plt.savefig(shap_summary_path, dpi=150)
    runtime.plt.close()
    print(f"   Сохранено: {shap_summary_path}")

    print("   Строим SHAP bar plot...")
    runtime.plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    runtime.plt.xlabel("Среднее абсолютное SHAP-значение")
    runtime.plt.tight_layout()
    shap_bar_path = runtime.FIGURES_DIR / "shap_bar_holidays_lags.png"
    runtime.plt.savefig(shap_bar_path, dpi=150)
    runtime.plt.close()
    print(f"   Сохранено: {shap_bar_path}")

    print("\nOK: всё готово!")


if __name__ == "__main__":
    train_xgboost_shap()
