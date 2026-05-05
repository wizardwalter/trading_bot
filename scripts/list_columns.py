import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

conn = psycopg2.connect(
    dbname=DB_CONFIG["dbname"],     
    user=DB_CONFIG["user"],       
    password=DB_CONFIG["password"],
    host=DB_CONFIG["host"],
    port=DB_CONFIG["port"]
)

df = pd.read_sql("SELECT * FROM candles_1m LIMIT 5;", conn)
print(df.columns)

conn.close()
