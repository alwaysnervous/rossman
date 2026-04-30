# Rossmann Sales Forecasting

Проект для прогнозирования продаж Rossmann: подготовка внешних данных, обучение моделей и анализ важности признаков.

## Структура

```text
Sem/
├─ data/
│  ├─ raw/          # исходные данные Rossmann
│  ├─ external/     # погода, Google Trends, безработица
│  └─ cache/        # локальные кеши API
├─ src/
│  ├─ data/         # скрипты загрузки внешних данных
│  └─ models/       # обучение моделей
├─ outputs/
│  ├─ figures/      # графики
│  ├─ models/       # сохранённые модели, если понадобятся
│  └─ reports/      # таблицы и отчёты
├─ notebooks/
├─ requirements.txt
└─ README.md
```

## Установка

```powershell
python -m pip install -r requirements.txt
```

## Запуск моделей

Запускать можно из корня проекта `Sem`:

```powershell
python src\models\baseline.py
python src\models\linear_regression.py
python src\models\random_forest.py
python src\models\xgboost_lags_shap.py
python src\models\xgboost_all_features.py
python src\models\xgboost_full_features.py
```

Графики сохраняются в `outputs/figures/`, отчёты в `outputs/reports/`.

После обучения модели сохраняются в `outputs/models/`:

```text
outputs/models/
├─ baseline_xgboost.json
├─ linear_regression.pkl
├─ random_forest.pkl
├─ xgboost_lags_shap.json
├─ xgboost_all_features.json
├─ xgboost_full_features.json
├─ feature_columns.json
└─ preprocessing_config.json
```

Файлы `.pkl` добавлены в `.gitignore`, потому что `random_forest.pkl` получается большим и обычный GitHub не принимает файлы больше 100 МБ. JSON-файлы можно хранить в репозитории.

## Обновление внешних данных

```powershell
python src\data\weather.py
python src\data\google_trends_parsing.py
python src\data\unemployment.py
```

Скрипты сохраняют результаты в `data/external/`.
