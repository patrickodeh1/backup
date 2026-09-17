# load_csv.py
import pandas as pd
from sqlalchemy import create_engine
import os

DATABASE_URL = os.environ["DATABASE_URL"].replace("postgres://", "postgresql+psycopg2://", 1)
engine = create_engine(DATABASE_URL)

CSV_PATH = "data/MetroPT3(AirCompressor).csv"
CHUNK_SIZE = 50_000

col_map = {
    "timestamp": "ts",
    "TP2": "tp2", "TP3": "tp3", "H1": "h1",
    "DV_pressure": "dv_pressure", "Reservoirs": "reservoirs",
    "Oil_temperature": "oil_temperature", "Motor_current": "motor_current",
    "COMP": "comp", "DV_eletric": "dv_eletric", "Towers": "towers",
    "MPG": "mpg", "LPS": "lps", "Pressure_switch": "pressure_switch",
    "Oil_level": "oil_level", "Caudal_impulses": "caudal_impulses",
}

total = 0
for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, parse_dates=["timestamp"]):
    chunk = chunk.drop(columns=["Unnamed: 0"])
    chunk = chunk.rename(columns=col_map)
    chunk.to_sql("readings", engine, if_exists="append", index=False, method="multi")
    total += len(chunk)
    print(f"Loaded {total} rows...")

print("Done.")
