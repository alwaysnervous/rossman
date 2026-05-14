import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import model_runtime as runtime


MODEL_NAME = "random_forest"
MODEL_FILE = "random_forest.pkl"
VARIANT = runtime.common.make_feature_variant(weather=True, holidays=True, lags=True)
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "random_state": 42,
    "n_jobs": -1,
}


def train_random_forest() -> None:
    runtime.ensure_output_dirs()

    print("=" * 60)
    print("МОДЕЛЬ 2: Random Forest (случайный лес)")
    print("=" * 60)

    data = runtime.prepare_logged_variant_data(VARIANT)

    print("\n4. Обучение Random Forest...")
    print("   Параметры: n_estimators=100, max_depth=15")
    model_rf = RandomForestRegressor(**MODEL_PARAMS, verbose=1)
    model_rf.fit(data.X_train, data.y_train)

    print("\n5. Предсказание...")
    pred_rf = np.clip(model_rf.predict(data.X_valid), 0, None)

    metrics = runtime.regression_metrics(data.y_valid, pred_rf)
    runtime.print_metrics_block("Random Forest", metrics)
    runtime.save_sklearn_model_outputs(
        model=model_rf, model_file=MODEL_FILE, model_name=MODEL_NAME,
        model_type="RandomForestRegressor", model_params=MODEL_PARAMS,
        variant=VARIANT, data=data, metrics=metrics,
    )

    print("\n6. График важности признаков...")
    imp_rf = pd.Series(
        model_rf.feature_importances_,
        index=data.X_train.columns,
    ).sort_values(ascending=False)

    # runtime.print_top_values(imp_rf, "Топ-15 важных признаков", "{:.4f}")
    runtime.save_top_bar_plot(
        imp_rf,
        title=None,
        output_filename="importance_rf.png",
        xlabel="Относительная важность признака",
    )


if __name__ == "__main__":
    train_random_forest()
