import pandas as pd
import time
from pytrends.request import TrendReq
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
EXTERNAL_DATA_DIR = ROOT_DIR / "data" / "external"
EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

states = {
    'BW': 'DE-BW', 'BY': 'DE-BY', 'BE': 'DE-BE', 'BB': 'DE-BB',
    'HB': 'DE-HB', 'HH': 'DE-HH', 'HE': 'DE-HE', 'MV': 'DE-MV',
    'NI': 'DE-NI', 'NW': 'DE-NW', 'RP': 'DE-RP', 'SL': 'DE-SL',
    'SN': 'DE-SN', 'ST': 'DE-ST', 'SH': 'DE-SH', 'TH': 'DE-TH'
}


def fetch_trends_for_state(state_code, keyword='rossmann', max_retries=3):
    for attempt in range(max_retries):
        try:
            pytrends = TrendReq(hl='de-DE', tz=360, timeout=(10, 25))
            pytrends.build_payload(
                kw_list=[keyword],
                cat=0,
                timeframe='2013-01-01 2015-07-31',
                geo=state_code,
                gprop=''
            )
            trends_df = pytrends.interest_over_time()

            if not trends_df.empty:
                trends_df = trends_df.drop(columns=['isPartial'])
                trends_df = trends_df.reset_index()
                trends_df = trends_df.rename(columns={'date': 'Date', keyword: 'GoogleTrend'})
                trends_df['State'] = state_code.split('-')[1]
                return trends_df
            else:
                print(f"   WARN: Пустой ответ для {state_code}")
                return pd.DataFrame()

        except Exception as e:
            print(f"   ERROR: Попытка {attempt + 1}/{max_retries} для {state_code}: {e}")
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)  # 30, 60, 90 секунд
                print(f"   Жду {wait} секунд...")
                time.sleep(wait)
            else:
                print(f"   ERROR: Все попытки для {state_code} исчерпаны")
                return pd.DataFrame()


print("Начинаю сбор Google Trends для всех 16 земель...")
all_trends = []

for state_name, state_code in states.items():
    print(f"Скачиваю данные для {state_name} ({state_code})...")
    state_df = fetch_trends_for_state(state_code)

    if not state_df.empty:
        all_trends.append(state_df)

    # Большая пауза между землями
    print(f"   Пауза 20 секунд...")
    time.sleep(20)

if all_trends:
    final_trends = pd.concat(all_trends, ignore_index=True)
    trends_path = EXTERNAL_DATA_DIR / 'googletrend_self.csv'
    final_trends.to_csv(trends_path, index=False)
    print(f"OK: Готово! Данные сохранены в {trends_path}")
    print(f"   Всего строк: {len(final_trends)}")
else:
    print("ERROR: Не удалось получить ни одной записи.")
