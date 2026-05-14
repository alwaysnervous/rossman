import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import model_runtime as runtime


MODEL_NAME = "linear_regression"
MODEL_FILE = "linear_regression.pkl"
MODEL_PARAMS = {"n_jobs": -1}
VARIANT = runtime.common.make_feature_variant(
    weather=True,
    unemployment=True,
    holidays=True,
    lags=True,
)


def train_linear_regression() -> None:
    runtime.ensure_output_dirs()

    print("=" * 60)
    print("МОДЕЛЬ 3: Linear Regression (линейная регрессия)")
    print("=" * 60)

    data = runtime.prepare_logged_variant_data(VARIANT)

    print("\n4. Обучение Linear Regression...")
    model_lr = LinearRegression(**MODEL_PARAMS)
    model_lr.fit(data.X_train, data.y_train)

    print("5. Предсказание...")
    pred_lr = model_lr.predict(data.X_valid)
    pred_lr = np.clip(pred_lr, 0, None)

    metrics = runtime.regression_metrics(data.y_valid, pred_lr)
    runtime.print_metrics_block("Linear Regression", metrics)
    runtime.save_sklearn_model_outputs(
        model=model_lr, model_file=MODEL_FILE, model_name=MODEL_NAME,
        model_type="LinearRegression", model_params=MODEL_PARAMS,
        variant=VARIANT, data=data, metrics=metrics,
    )

    print("\n6. Топ-15 коэффициентов...")
    coef = pd.Series(
        np.abs(model_lr.coef_),
        index=data.X_train.columns,
    ).sort_values(ascending=False)

    # runtime.print_top_values(coef, "Топ-15 по модулю коэффициентов", "{:.0f} €")
    runtime.save_top_bar_plot(
        coef,
        title=None,
        output_filename="importance_lr.png",
        xlabel="Модуль коэффициента признака"
    )


if __name__ == "__main__":
    train_linear_regression()
