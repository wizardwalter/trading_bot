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
With your virtual environment activated you can trigger a single trading pass:
```bash
python cli/main.py
```
Or launch the continuous loop with optional live execution harness:
```bash
python cli/main.py --loop --interval 60 --status-every 5 --execute
```
 
### Step 8: Monitor CLI & Discord
The bot logs actions to the CLI and sends notifications to your Discord server. Tail `logs/autonomous.log` when running via the helper scripts below.

### Step 9: Autonomous Trading Loop (optional)
Wrapper scripts live under `scripts/` to keep the bot running in the background:
```bash
./scripts/start_autonomous.sh      # start loop (writes logs/autonomous.log)
./scripts/status_autonomous.sh     # tail recent output & show PID
./scripts/stop_autonomous.sh       # shut it down
```
 
### Step 10: Autonomous Training Loop (optional)
Use the training daemon to keep models fresh without supervision:
```bash
./scripts/start_training.sh        # starts services/training_daemon.py
./scripts/status_training.sh       # tail logs/training_daemon.log
./scripts/stop_training.sh         # stop the daemon
```
Environment overrides (set in `.env` or inline) include `TRAINING_INTERVAL_MINUTES`, `TRAINING_JITTER_MINUTES`, `TRAINING_SYMBOL`, `TRAINING_BAR_INTERVAL`, and `TRAINING_PERIOD`.

### Progress Ledger (Crash Recovery)
Both the trading loop and training daemon now emit structured breadcrumbs to `logs/progress.jsonl`, with the latest event mirrored to `logs/progress_latest.json`. Tail the ledger anytime you need to know what the bot was doing before a restart or crash:
```bash
tail -f logs/progress.jsonl
```
`logs/progress_state.json` is reserved for the most recent high-level snapshot. Feel free to extend it with any extra metadata your supervisor stack might need.

---
💡 You can use `cron` or `launchd` to schedule the bot to run between market hours (9:30AM – 4:00PM EST). The daemon scripts can also be dropped into systemd services or supervisor configs for hands-off ops.
 
Happy trading 🚀

---

## 🧠 Memory Jogger (Commands to Remember)

Because we got bigger things to focus on than memorizing flags.

### 🔧 Dev Environment
```bash
python3 -m venv .venv        # Create virtual env
source .venv/bin/activate    # Activate it
deactivate                   # Bounce outta there
```

### 🐍 Python Essentials
```bash
pip install -r requirements.txt  # Install deps
python3 ./cli/main.py            # Run main logic
```

### 🐘 PostgreSQL (local)
```bash
psql -U postgres -d postgres              # Log into DB
\dt                                      # Show tables
\du                                      # Show users
\q                                       # Quit
```

### 🐳 Docker Stuff
```bash
docker-compose up -d         # Fire up the container
docker-compose down          # Shut it down
docker volume ls             # List volumes
docker volume rm <name>      # Remove a volume
```

### 🕒 Cron Sample
```cron
0 11 * * 1-5 /path/to/project/.venv/bin/python /path/to/project/services/yfinance_candles.py >> /tmp/trade.log 2>&1
```

---

## 🔥 Bot Summary (For the Homies)

This ain’t your grandpa’s stock bot. This is a machine-learning-powered, Discord-notifying, PostgreSQL-storing, ultra-slick auto-trader built with love, caffeine, and a little bit of market revenge.

What it does:
- Pulls candles faster than John Wick reloads
- Calculates indicators like a trading Jedi
- Makes real-time decisions (buy/hold/sell) — all without blinking
- Learns from its mistakes and comes back stronger
- Sends you Discord updates so you look like a Wall Street wizard while walking your dog

Future plans:
- Confidence scoring? Yup.
- Ensemble models with voting logic? Already in the works.
- Real-time fire trades that print green candles all day? You bet your last contract.

Let’s build the beast, feed it data, train it up, and let it feast on the market. The game plan is strong, the hustle is real, and the tech is bulletproof (or at least cronproof).

Welcome to the grind. Let’s make some damn money 💰🚀
