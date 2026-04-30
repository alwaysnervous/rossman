import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
EXTERNAL_DATA_DIR = ROOT_DIR / "data" / "external"
CACHE_DIR = ROOT_DIR / "data" / "cache"
EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Координаты столиц федеральных земель
state_capitals = {
    'BW': {'lat': 48.78, 'lon': 9.18},   # Stuttgart
    'BY': {'lat': 48.14, 'lon': 11.58},  # München
    'BE': {'lat': 52.52, 'lon': 13.40},  # Berlin
    'BB': {'lat': 52.41, 'lon': 13.06},  # Potsdam
    'HB': {'lat': 53.08, 'lon': 8.80},   # Bremen
    'HH': {'lat': 53.55, 'lon': 10.00},  # Hamburg
    'HE': {'lat': 50.08, 'lon': 8.24},   # Wiesbaden
    'MV': {'lat': 53.63, 'lon': 11.41},  # Schwerin
    'NI': {'lat': 52.37, 'lon': 9.74},   # Hannover
    'NW': {'lat': 51.22, 'lon': 6.79},   # Düsseldorf
    'RP': {'lat': 50.00, 'lon': 8.27},   # Mainz
    'SL': {'lat': 49.23, 'lon': 7.00},   # Saarbrücken
    'SN': {'lat': 51.05, 'lon': 13.74},  # Dresden
    'ST': {'lat': 52.13, 'lon': 11.62},  # Magdeburg
    'SH': {'lat': 54.32, 'lon': 10.14},  # Kiel
    'TH': {'lat': 50.98, 'lon': 11.03}   # Erfurt
}

def fetch_daytime_weather(lat, lon):
    """Скачивает почасовые данные и агрегирует только за 9:00-20:00."""
    cache_session = requests_cache.CachedSession(str(CACHE_DIR / 'weather_cache'), expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": "2013-01-01", "end_date": "2015-07-31",
        "hourly": ["temperature_2m", "precipitation", "sunshine_duration"],
        "timezone": "Europe/Berlin"  # Важно: теперь время местное!
    }
    response = openmeteo.weather_api(url, params=params)[0]
    hourly = response.Hourly()

    # Создаём DataFrame с почасовыми данными
    dates = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit='s'),
        periods=len(hourly.Variables(0).ValuesAsNumpy()),
        freq='h'
    )
    df_hourly = pd.DataFrame({
        'datetime': dates,
        'temperature': hourly.Variables(0).ValuesAsNumpy(),
        'precipitation': hourly.Variables(1).ValuesAsNumpy(),
        'sunshine': hourly.Variables(2).ValuesAsNumpy()
    })
    df_hourly['date'] = df_hourly['datetime'].dt.date
    df_hourly['hour'] = df_hourly['datetime'].dt.hour

    # Оставляем только рабочие часы магазинов (9:00 - 20:00)
    work_hours = df_hourly[(df_hourly['hour'] >= 9) & (df_hourly['hour'] <= 20)]

    # Агрегируем по дням
    daily = work_hours.groupby('date').agg({
        'temperature': ['mean', 'max'],           # средняя и максимальная дневная температура
        'precipitation': 'sum',                   # сумма осадков за день
        'sunshine': 'sum'                         # сумма солнечного сияния за день
    })
    daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
    daily = daily.reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily.rename(columns={'date': 'Date'}, inplace=True)

    return daily


print("Начинаю загрузку ДНЕВНОЙ погоды для 16 федеральных земель...")
all_weather = []

for state, coords in state_capitals.items():
    print(f"  Загружаю {state}...")
    df_state = fetch_daytime_weather(coords['lat'], coords['lon'])
    df_state['State'] = state
    all_weather.append(df_state)
    time.sleep(0.5)

final_weather = pd.concat(all_weather, ignore_index=True)
weather_path = EXTERNAL_DATA_DIR / 'weather_daytime.csv'
final_weather.to_csv(weather_path, index=False)
print(f"OK: Готово! Дневная погода сохранена в {weather_path}")
print("Колонки:", list(final_weather.columns))
