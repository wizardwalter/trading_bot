import psycopg2

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

cur = conn.cursor()
cur.execute("""
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'public';
""")

print("📦 Tables in your DB:")
for row in cur.fetchall():
    print(" -", row[0])

cur.close()
conn.close()