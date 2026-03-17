"""
TradeIQ AI — Performance Database
====================================
Stores every trade, signal, pattern outcome, and parameter state
in a local SQLite database. This is the memory the AI learns from.

Every day adds more data. Over time patterns emerge:
  - Which FVG sizes win most often?
  - What quality scores actually correlate with wins?
  - Which hours of day are most profitable?
  - Which symbols respond best to which strategies?
  - What ATR conditions produce best setups?

The optimizer reads this database each night to tune parameters.
"""

import sqlite3
import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional, Any
from loguru import logger


DB_PATH = Path("learning/tradeiq_memory.db")


class PerformanceDB:
    """
    SQLite database storing all historical trade and signal data.
    Grows every session. Never resets. This is the system's long-term memory.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        """Create all tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript("""
                -- Every completed trade
                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_date    TEXT NOT NULL,
                    symbol          TEXT NOT NULL,
                    direction       TEXT NOT NULL,
                    strategy        TEXT NOT NULL,
                    entry_price     REAL NOT NULL,
                    exit_price      REAL,
                    stop_loss       REAL NOT NULL,
                    target          REAL NOT NULL,
                    shares          REAL NOT NULL,
                    pnl             REAL DEFAULT 0,
                    pnl_pct         REAL DEFAULT 0,
                    outcome         TEXT,           -- 'win', 'loss', 'breakeven'
                    exit_reason     TEXT,           -- 'target_hit', 'stopped_out', 'eod_close'
                    quality_score   REAL,
                    entry_hour      INTEGER,        -- hour of day (9-15)
                    entry_minute    INTEGER,
                    orb_range       REAL,
                    fvg_size        REAL,
                    fvg_size_atr    REAL,           -- fvg_size / atr
                    atr             REAL,
                    market_minute   INTEGER,        -- minutes since 9:30 open
                    signals         TEXT,           -- JSON array of signal names
                    created_at      TEXT DEFAULT (datetime('now'))
                );

                -- Every setup that was detected (including rejected ones)
                CREATE TABLE IF NOT EXISTS setups (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_date    TEXT NOT NULL,
                    symbol          TEXT NOT NULL,
                    strategy        TEXT NOT NULL,
                    direction       TEXT,
                    quality_score   REAL,
                    executed        INTEGER DEFAULT 0,  -- 1 if trade placed, 0 if rejected
                    reject_reason   TEXT,
                    rr_ratio        REAL,
                    signals         TEXT,
                    market_minute   INTEGER,
                    created_at      TEXT DEFAULT (datetime('now'))
                );

                -- Daily session summaries
                CREATE TABLE IF NOT EXISTS sessions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_date    TEXT UNIQUE NOT NULL,
                    total_trades    INTEGER DEFAULT 0,
                    wins            INTEGER DEFAULT 0,
                    losses          INTEGER DEFAULT 0,
                    win_rate        REAL DEFAULT 0,
                    total_pnl       REAL DEFAULT 0,
                    pnl_pct         REAL DEFAULT 0,
                    starting_equity REAL,
                    ending_equity   REAL,
                    kill_switch     INTEGER DEFAULT 0,
                    day_trades_used INTEGER DEFAULT 0,
                    market_conditions TEXT,          -- JSON: vix, trend, volatility
                    paper_mode      INTEGER DEFAULT 1,
                    created_at      TEXT DEFAULT (datetime('now'))
                );

                -- Parameter snapshots — what config was used each day
                CREATE TABLE IF NOT EXISTS parameters (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_date    TEXT NOT NULL,
                    param_name      TEXT NOT NULL,
                    param_value     REAL NOT NULL,
                    source          TEXT DEFAULT 'manual',  -- 'manual', 'optimizer'
                    created_at      TEXT DEFAULT (datetime('now'))
                );

                -- AI optimizer decisions — what it changed and why
                CREATE TABLE IF NOT EXISTS optimizations (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    applied_date    TEXT NOT NULL,
                    param_name      TEXT NOT NULL,
                    old_value       REAL,
                    new_value       REAL,
                    reasoning       TEXT,
                    confidence      REAL,
                    days_of_data    INTEGER,
                    created_at      TEXT DEFAULT (datetime('now'))
                );

                -- Indexes for fast queries
                CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(session_date);
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
                CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(session_date);
            """)
        logger.debug(f"Performance DB initialized: {self.db_path}")

    # ─── Write Methods ────────────────────────────────────────────────────

    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a completed trade with all metadata."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO trades (
                    session_date, symbol, direction, strategy,
                    entry_price, exit_price, stop_loss, target, shares,
                    pnl, pnl_pct, outcome, exit_reason, quality_score,
                    entry_hour, entry_minute, orb_range, fvg_size,
                    fvg_size_atr, atr, market_minute, signals
                ) VALUES (
                    :session_date, :symbol, :direction, :strategy,
                    :entry_price, :exit_price, :stop_loss, :target, :shares,
                    :pnl, :pnl_pct, :outcome, :exit_reason, :quality_score,
                    :entry_hour, :entry_minute, :orb_range, :fvg_size,
                    :fvg_size_atr, :atr, :market_minute, :signals
                )
            """, trade_data)

    def record_setup(self, setup_data: Dict[str, Any]):
        """Record every detected setup, whether executed or not."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO setups (
                    session_date, symbol, strategy, direction,
                    quality_score, executed, reject_reason, rr_ratio,
                    signals, market_minute
                ) VALUES (
                    :session_date, :symbol, :strategy, :direction,
                    :quality_score, :executed, :reject_reason, :rr_ratio,
                    :signals, :market_minute
                )
            """, setup_data)

    def record_session(self, session_data: Dict[str, Any]):
        """Record end-of-day session summary."""
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions (
                    session_date, total_trades, wins, losses, win_rate,
                    total_pnl, pnl_pct, starting_equity, ending_equity,
                    kill_switch, day_trades_used, paper_mode
                ) VALUES (
                    :session_date, :total_trades, :wins, :losses, :win_rate,
                    :total_pnl, :pnl_pct, :starting_equity, :ending_equity,
                    :kill_switch, :day_trades_used, :paper_mode
                )
            """, session_data)

    def record_parameters(self, session_date: str, params: Dict[str, float], source: str = "manual"):
        """Snapshot all parameters used in a session."""
        with self._conn() as conn:
            for name, value in params.items():
                conn.execute("""
                    INSERT INTO parameters (session_date, param_name, param_value, source)
                    VALUES (?, ?, ?, ?)
                """, (session_date, name, value, source))

    def record_optimization(self, applied_date: str, changes: List[Dict]):
        """Record what the optimizer changed."""
        with self._conn() as conn:
            for change in changes:
                conn.execute("""
                    INSERT INTO optimizations (
                        applied_date, param_name, old_value, new_value,
                        reasoning, confidence, days_of_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    applied_date,
                    change["param"], change["old_value"], change["new_value"],
                    change["reasoning"], change.get("confidence", 0.5),
                    change.get("days_of_data", 0),
                ))

    # ─── Read / Analysis Methods ──────────────────────────────────────────

    def get_all_trades(self, min_date: str = None) -> List[Dict]:
        """Get all trades, optionally filtered by date."""
        with self._conn() as conn:
            if min_date:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE session_date >= ? ORDER BY session_date, id",
                    (min_date,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades ORDER BY session_date, id"
                ).fetchall()
            return [dict(r) for r in rows]

    def get_strategy_performance(self) -> List[Dict]:
        """Win rate and P&L by strategy."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT
                    strategy,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins,
                    ROUND(AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) * 100, 1) as win_rate_pct,
                    ROUND(SUM(pnl), 2) as total_pnl,
                    ROUND(AVG(pnl), 2) as avg_pnl,
                    ROUND(AVG(pnl_pct), 2) as avg_pnl_pct
                FROM trades
                GROUP BY strategy
                ORDER BY win_rate_pct DESC
            """).fetchall()
            return [dict(r) for r in rows]

    def get_fvg_size_analysis(self) -> List[Dict]:
        """Performance grouped by FVG size (in ATR multiples)."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT
                    ROUND(fvg_size_atr, 1) as fvg_atr_bucket,
                    COUNT(*) as trades,
                    ROUND(AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) * 100, 1) as win_rate_pct,
                    ROUND(AVG(pnl_pct), 2) as avg_pnl_pct
                FROM trades
                WHERE fvg_size_atr IS NOT NULL AND fvg_size_atr > 0
                GROUP BY fvg_atr_bucket
                HAVING trades >= 3
                ORDER BY fvg_atr_bucket
            """).fetchall()
            return [dict(r) for r in rows]

    def get_quality_score_analysis(self) -> List[Dict]:
        """Performance grouped by quality score bucket."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT
                    ROUND(quality_score * 10) / 10 as q_bucket,
                    COUNT(*) as trades,
                    ROUND(AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) * 100, 1) as win_rate_pct,
                    ROUND(AVG(pnl_pct), 2) as avg_pnl_pct
                FROM trades
                WHERE quality_score IS NOT NULL
                GROUP BY q_bucket
                HAVING trades >= 2
                ORDER BY q_bucket
            """).fetchall()
            return [dict(r) for r in rows]

    def get_hour_analysis(self) -> List[Dict]:
        """Performance by entry hour."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT
                    entry_hour,
                    COUNT(*) as trades,
                    ROUND(AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) * 100, 1) as win_rate_pct,
                    ROUND(AVG(pnl_pct), 2) as avg_pnl_pct
                FROM trades
                WHERE entry_hour IS NOT NULL
                GROUP BY entry_hour
                ORDER BY entry_hour
            """).fetchall()
            return [dict(r) for r in rows]

    def get_symbol_performance(self) -> List[Dict]:
        """Performance by symbol."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT
                    symbol,
                    COUNT(*) as trades,
                    ROUND(AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) * 100, 1) as win_rate_pct,
                    ROUND(SUM(pnl), 2) as total_pnl,
                    ROUND(AVG(pnl_pct), 2) as avg_pnl_pct
                FROM trades
                GROUP BY symbol
                HAVING trades >= 2
                ORDER BY total_pnl DESC
            """).fetchall()
            return [dict(r) for r in rows]

    def get_recent_sessions(self, n: int = 30) -> List[Dict]:
        """Last N session summaries."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY session_date DESC LIMIT ?", (n,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_optimization_history(self) -> List[Dict]:
        """All past optimizer decisions."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM optimizations ORDER BY applied_date DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_current_params(self) -> Dict[str, float]:
        """Most recent parameter values for each param."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT param_name, param_value
                FROM parameters
                WHERE id IN (
                    SELECT MAX(id) FROM parameters GROUP BY param_name
                )
            """).fetchall()
            return {r["param_name"]: r["param_value"] for r in rows}

    def get_stats_summary(self) -> Dict:
        """High-level stats across all history."""
        with self._conn() as conn:
            r = conn.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as total_wins,
                    ROUND(AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END)*100,1) as overall_win_rate,
                    ROUND(SUM(pnl), 2) as total_pnl,
                    COUNT(DISTINCT session_date) as trading_days
                FROM trades
            """).fetchone()
            return dict(r) if r else {}

    def count_trades(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
