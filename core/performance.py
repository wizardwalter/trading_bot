from __future__ import annotations

from dataclasses import dataclass

from data.database import get_connection


@dataclass
class PerfSummary:
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    gross_pnl: float


def compute_performance() -> PerfSummary:
    query = """
    WITH paired AS (
      SELECT
        ticker,
        action,
        price::float8 AS price,
        quantity::float8 AS qty,
        timestamp,
        LEAD(action) OVER (PARTITION BY ticker ORDER BY timestamp) AS next_action,
        LEAD(price::float8) OVER (PARTITION BY ticker ORDER BY timestamp) AS next_price,
        LEAD(quantity::float8) OVER (PARTITION BY ticker ORDER BY timestamp) AS next_qty
      FROM trades
    ),
    closed AS (
      SELECT
        ticker,
        CASE
          WHEN action='buy' AND next_action='sell' THEN (next_price - price) * LEAST(qty, COALESCE(next_qty, qty))
          WHEN action='sell' AND next_action='buy' THEN (price - next_price) * LEAST(qty, COALESCE(next_qty, qty))
          ELSE NULL
        END AS pnl
      FROM paired
    )
    SELECT
      COUNT(*) FILTER (WHERE pnl IS NOT NULL) AS n,
      COUNT(*) FILTER (WHERE pnl > 0) AS wins,
      COUNT(*) FILTER (WHERE pnl < 0) AS losses,
      COALESCE(SUM(pnl), 0) AS gross_pnl
    FROM closed;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            n, wins, losses, gross_pnl = cur.fetchone()

    n = int(n or 0)
    wins = int(wins or 0)
    losses = int(losses or 0)
    gross_pnl = float(gross_pnl or 0.0)
    win_rate = (wins / n) if n else 0.0

    return PerfSummary(
        trade_count=n,
        win_count=wins,
        loss_count=losses,
        win_rate=win_rate,
        gross_pnl=gross_pnl,
    )
