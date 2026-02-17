import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "trading_bot"),
    "user": os.getenv("DB_USER", "walter"),
    "password": os.getenv("DB_PASS", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
