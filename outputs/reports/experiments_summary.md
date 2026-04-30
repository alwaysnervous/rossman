# Описание экспериментов

## Общие условия

Во всех экспериментах использовался датасет Rossmann с целевой переменной `Sales`.
Из обучающей выборки были исключены строки, где магазин был закрыт (`Open == 0`), так как продажи в эти дни равны нулю и не отражают обычный спрос.

Разделение на обучающую и валидационную выборки было выполнено по времени:

- train: даты до `2015-06-01`;
- valid: даты начиная с `2015-06-01`.

Пропуски после объединения таблиц заполнялись нулями (`fillna(0)`). Категориальные признаки `StateHoliday`, `StoreType`, `Assortment`, `PromoInterval` кодировались через one-hot encoding.

Во всех моделях использовались базовые признаки магазина и календаря:

- признаки магазина: `Store`, `DayOfWeek`, `Open`, `Promo`, `SchoolHoliday`, `CompetitionDistance`, `Promo2` и связанные с ними поля;
- календарные признаки: `year`, `month`, `day`, `dayofweek`, `WeekOfYear`;
- для части экспериментов дополнительно: `IsMonthEnd`, `IsFriday`.

Для сравнения использовался baseline с `RMSPE = 0.1720`.

## Baseline. XGBoost: погода + публичный праздник, без лагов

Скрипт: `src/models/baseline.py`

Модель: `xgboost.Booster`

Параметры:

- `max_depth = 5`;
- `learning_rate = 0.05`;
- `objective = reg:squarederror`;
- `eval_metric = rmse`;
- `num_boost_round = 3000`;
- `early_stopping_rounds = 50`;
- `best_iteration = 2999`.

Размерность:

- train: `(785781, 44)`;
- valid: `(58611, 44)`.

Учитываемые внешние и дополнительные признаки:

- погода: `temperature_mean`, `temperature_max`, `precipitation_sum`, `sunshine_sum`;
- производные погодные признаки: `TempRange`, `IsRainy`, `IsSunny`;
- праздник: `PublicHoliday`.

Не учитывались:

- Google Trends;
- уровень безработицы;
- лаговые признаки продаж;
- признаки `BeforeHoliday`, `AfterHoliday`;
- взаимодействия промо и погоды.

Метрики:

| Метрика | Значение |
|---|---:|
| RMSPE | 0.1720 |
| MAE | 774 € |
| R2 | 0.8812 |

## Эксперимент 1. Linear Regression

Скрипт: `src/models/linear_regression.py`

Модель: `LinearRegression(n_jobs=-1)`

Размерность:

- train: `(785781, 56)`;
- valid: `(58611, 56)`.

Учитываемые внешние и дополнительные признаки:

- погода по федеральной земле и дате: `temperature_mean`, `temperature_max`, `precipitation_sum`, `sunshine_sum`;
- производные погодные признаки: `TempRange`, `IsRainy`, `IsSunny`, `IsHot`, `IsCold`;
- Google Trends: `GoogleTrend`;
- уровень безработицы в Германии: `UnemploymentRate`;
- праздники: `PublicHoliday`, `BeforeHoliday`, `AfterHoliday`;
- взаимодействия промо и погоды: `Promo_Temperature`, `Promo_Sunny`, `Promo_Rainy`;
- лаговые признаки продаж: `Sales_Lag1`, `Sales_Lag7`, `Sales_Rolling7`.

Метрики:

| Метрика | Значение |
|---|---:|
| RMSPE | 0.2081 |
| MAE | 1016 € |
| R2 | 0.7654 |

Итог: линейная модель показала результат хуже baseline. Вероятная причина — зависимость продаж от признаков нелинейная, а линейная регрессия плохо описывает взаимодействия между промо, календарём, магазином и лагами продаж.

## Эксперимент 2. Random Forest

Скрипт: `src/models/random_forest.py`

Модель: `RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)`

Размерность:

- train: `(785781, 56)`;
- valid: `(58611, 56)`.

Учитываемые внешние и дополнительные признаки:

- погода: `temperature_mean`, `temperature_max`, `precipitation_sum`, `sunshine_sum`;
- производные погодные признаки: `TempRange`, `IsRainy`, `IsSunny`, `IsHot`, `IsCold`;
- Google Trends: `GoogleTrend`;
- уровень безработицы: `UnemploymentRate`;
- праздники: `PublicHoliday`, `BeforeHoliday`, `AfterHoliday`;
- взаимодействия промо и погоды: `Promo_Temperature`, `Promo_Sunny`, `Promo_Rainy`;
- лаговые признаки продаж: `Sales_Lag1`, `Sales_Lag7`, `Sales_Rolling7`.

Метрики:

| Метрика | Значение |
|---|---:|
| RMSPE | 0.1445 |
| MAE | 687 € |
| R2 | 0.8972 |

Итог: Random Forest показал лучший результат среди всех проведённых экспериментов. Самым важным признаком оказался `Sales_Lag1`, затем `Sales_Rolling7`, `Promo`, календарные признаки и `Sales_Lag7`.

## Эксперимент 3. XGBoost: погода + праздники + лаги

Скрипт: `src/models/xgboost_lags_shap.py`

Модель: `xgboost.Booster`

Параметры:

- `max_depth = 5`;
- `learning_rate = 0.05`;
- `objective = reg:squarederror`;
- `eval_metric = rmse`;
- `num_boost_round = 3000`;
- `early_stopping_rounds = 50`;
- `best_iteration = 537`.

Размерность:

- train: `(785781, 45)`;
- valid: `(58611, 45)`.

Учитываемые внешние и дополнительные признаки:

- погода: `temperature_mean`, `temperature_max`, `precipitation_sum`, `sunshine_sum`;
- производные погодные признаки: `TempRange`, `IsRainy`, `IsSunny`;
- праздник: `PublicHoliday`;
- лаговые признаки продаж: `Sales_Lag1`, `Sales_Lag7`, `Sales_Rolling7`.

Не учитывались:

- Google Trends;
- уровень безработицы;
- признаки `BeforeHoliday`, `AfterHoliday`;
- взаимодействия промо и погоды.

Метрики:

| Метрика | Значение |
|---|---:|
| RMSPE | 0.1537 |
| MAE | 735 € |
| R2 | 0.8849 |
| Улучшение относительно baseline | 10.64% |

Итог: добавление погодных признаков, праздников и лагов заметно улучшило качество относительно baseline. Также для этой модели был построен SHAP-анализ.

## Эксперимент 4. XGBoost: внешние признаки без лагов

Скрипт: `src/models/xgboost_full_features.py`

Модель: `xgboost.Booster`

Параметры:

- `max_depth = 5`;
- `learning_rate = 0.05`;
- `objective = reg:squarederror`;
- `eval_metric = rmse`;
- `num_boost_round = 3000`;
- `early_stopping_rounds = 50`;
- `best_iteration = 2999`.

Размерность:

- train: `(785781, 53)`;
- valid: `(58611, 53)`.

Учитываемые внешние и дополнительные признаки:

- погода: `temperature_mean`, `temperature_max`, `precipitation_sum`, `sunshine_sum`;
- производные погодные признаки: `TempRange`, `IsRainy`, `IsSunny`, `IsHot`, `IsCold`;
- Google Trends: `GoogleTrend`;
- уровень безработицы: `UnemploymentRate`;
- праздники: `PublicHoliday`, `BeforeHoliday`, `AfterHoliday`;
- взаимодействия промо и погоды: `Promo_Temperature`, `Promo_Sunny`, `Promo_Rainy`.

Не учитывались:

- лаговые признаки продаж.

Метрики:

| Метрика | Значение |
|---|---:|
| RMSPE | 0.1689 |
| MAE | 771 € |
| Улучшение относительно baseline | 1.83% |

Итог: внешние признаки без лагов улучшили baseline, но качество оказалось хуже, чем у моделей с лаговыми признаками. Это показывает, что история продаж является самым сильным источником информации.

## Эксперимент 5. XGBoost: все признаки

Скрипт: `src/models/xgboost_all_features.py`

Модель: `xgboost.Booster`

Параметры:

- `max_depth = 5`;
- `learning_rate = 0.05`;
- `objective = reg:squarederror`;
- `eval_metric = rmse`;
- `num_boost_round = 3000`;
- `early_stopping_rounds = 50`;
- `best_iteration = 557`.

Размерность:

- train: `(785781, 56)`;
- valid: `(58611, 56)`.

Учитываемые внешние и дополнительные признаки:

- погода: `temperature_mean`, `temperature_max`, `precipitation_sum`, `sunshine_sum`;
- производные погодные признаки: `TempRange`, `IsRainy`, `IsSunny`, `IsHot`, `IsCold`;
- Google Trends: `GoogleTrend`;
- уровень безработицы: `UnemploymentRate`;
- праздники: `PublicHoliday`, `BeforeHoliday`, `AfterHoliday`;
- взаимодействия промо и погоды: `Promo_Temperature`, `Promo_Sunny`, `Promo_Rainy`;
- лаговые признаки продаж: `Sales_Lag1`, `Sales_Lag7`, `Sales_Rolling7`.

Метрики:

| Метрика | Значение |
|---|---:|
| RMSPE | 0.1517 |
| Улучшение относительно baseline | 11.78% |

Итог: это лучший XGBoost-эксперимент по RMSPE. Он немного лучше версии с погодой, праздниками и лагами, но хуже Random Forest.

## Сводная таблица

| Эксперимент | Модель | Внешние признаки | Лаги | RMSPE | MAE | R2 |
|---|---|---|---|---:|---:|---:|
| Baseline | XGBoost | погода, публичный праздник | нет | 0.1720 | 774 € | 0.8812 |
| Linear Regression | Linear Regression | погода, Google Trends, безработица, праздники | да | 0.2081 | 1016 € | 0.7654 |
| Random Forest | Random Forest | погода, Google Trends, безработица, праздники | да | 0.1445 | 687 € | 0.8972 |
| XGBoost: погода + праздники + лаги | XGBoost | погода, праздник | да | 0.1537 | 735 € | 0.8849 |
| XGBoost: внешние признаки без лагов | XGBoost | погода, Google Trends, безработица, праздники | нет | 0.1689 | 771 € | не считался |
| XGBoost: все признаки | XGBoost | погода, Google Trends, безработица, праздники | да | 0.1517 | не считался | не считался |

## Общий вывод

Лучшее качество показал Random Forest:

- `RMSPE = 0.1445`;
- `MAE = 687 €`;
- `R2 = 0.8972`.

Среди XGBoost-моделей лучший результат дал эксперимент со всеми признаками:

- `RMSPE = 0.1517`.

Главный вывод по признакам: лаговые признаки продаж (`Sales_Lag1`, `Sales_Rolling7`, `Sales_Lag7`) дают наиболее сильный вклад в качество. Внешние признаки вроде погоды, Google Trends и безработицы улучшают модель, но без истории продаж дают менее сильный эффект.

Важно: лаговые признаки были рассчитаны на всём временном ряду до разделения на train и valid. Для реального прогноза на `test.csv` нужно следить, чтобы лаги строились только из доступной на момент прогноза истории, иначе можно получить утечку информации из будущего.
