import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
EXTERNAL_DATA_DIR = ROOT_DIR / "data" / "external"
EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Прямая ссылка на CSV с данными FRED
# LRHUTTTTDEM156S — код показателя "Harmonized Unemployment Rate: All Persons for Germany"
# Это официальный показатель уровня безработицы в Германии (месячный, в %)
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=LRHUTTTTDEM156S"

print("Скачиваю уровень безработицы Германии с FRED...")
try:
    df = pd.read_csv(url)
    print(f"OK: Скачано! Размер: {df.shape}")
    print(f"   Колонки: {list(df.columns)}")
    print(f"   Первые строки:")
    print(df.head(10))
    
    # Переименовываем колонки для удобства
    df.columns = ['Date', 'UnemploymentRate']
    
    # Преобразуем дату
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Оставляем только нужный период (2013-2015)
    df = df[(df['Date'] >= '2013-01-01') & (df['Date'] <= '2015-07-31')]
    
    # Сохраняем в CSV
    unemployment_path = EXTERNAL_DATA_DIR / 'unemployment_germany.csv'
    df.to_csv(unemployment_path, index=False)
    print(f"\nOK: Сохранено в {unemployment_path}")
    print(f"   Период: {df['Date'].min().date()} — {df['Date'].max().date()}")
    print(f"   Строк: {len(df)}")
    print(f"   Мин: {df['UnemploymentRate'].min():.1f}%")
    print(f"   Макс: {df['UnemploymentRate'].max():.1f}%")
    print(f"   Среднее: {df['UnemploymentRate'].mean():.1f}%")
    
except Exception as e:
    print(f"ERROR: Ошибка: {e}")
    print("Проверь интернет-соединение или попробуй позже.")
