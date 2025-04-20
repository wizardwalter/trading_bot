# Trading Bot
 
## Setup Instructions
 
This bot is built for AI-enhanced trading with PostgreSQL for storing market data and Discord webhooks for trade notifications.
 
### Prerequisites
- macOS with Homebrew installed
- Docker & Docker Compose
- Python 3 (comes with macOS)
- Git
- Discord webhook URL
### Step 1: Clone the Repo
```bash
git clone https://github.com/yourusername/tradingBot.git
cd tradingBot
```
 
### Step 2: Set Up the Virtual Environment
Create and activate a Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
 
### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```
 
### Step 4: Create a `.env` File
Create a `.env` file in the root directory with your Discord webhook and DB credentials:
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
DB_PASS=your_database_password
```
 
### Step 5: Start PostgreSQL with Docker Compose
Ensure you have Docker running, then spin up the DB:
```bash
docker-compose up -d
```
 
### Step 6: Load the Schema
Your schema file should be mapped in `docker-compose.yml`. It will be executed on first DB boot.
 
Example schema for tickers:
```sql
CREATE TABLE IF NOT EXISTS tickers (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(10) NOT NULL UNIQUE
);
```
 
### Step 7: Run the Bot
With your virtual environment activated:
```bash
python main.py
```
 
### Step 8: Monitor CLI & Discord
The bot logs actions to the CLI and sends notifications to your Discord server.
 
---
💡 You can use `cron` or `launchd` to schedule the bot to run between market hours (9:30AM – 4:00PM EST).
 
Happy trading 🚀
