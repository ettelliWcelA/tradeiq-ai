"""
TradeIQ AI — Robinhood Data Manager
=====================================
Wraps robin_stocks to provide clean, validated OHLCV data.

Key limitations of Robinhood's API (work around them):
  - Minimum bar interval is 5 minutes (no 1-minute bars)
  - Historical data limited to: 5min/10min/hour/day/week intervals
  - Only 'day' span works with 'extended' bounds
  - Rate limits: be conservative, use caching

Strategy adaptation: We use 5-minute bars for our ORB/FVG logic
(same as the "first candle rule" from the source material).
"""

import time
import pytz
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

import pandas as pd
import numpy as np
from loguru import logger

import robin_stocks.robinhood as rh

EASTERN = pytz.timezone("America/New_York")
MARKET_OPEN  = (9, 30)
MARKET_CLOSE = (16, 0)

# ── AI / Tech Stock Universe ───────────────────────────────────────────────
AI_STOCKS = [
    # Core AI Semis
    "NVDA", "AMD", "AVGO", "TSM", "SMCI", "INTC", "QCOM",
    # Cloud AI / Big Tech
    "MSFT", "GOOGL", "AMZN", "META", "ORCL",
    # Pure-play AI
    "PLTR", "AI", "BBAI", "SOUN", "IONQ",
    # AI Infrastructure
    "DELL", "HPE", "ANET",
]

# Robinhood field name mapping
FIELD_MAP = {
    "begins_at":   "timestamp",
    "open_price":  "open",
    "close_price": "close",
    "high_price":  "high",
    "low_price":   "low",
    "volume":      "volume",
}


class RobinhoodDataManager:
    """
    Market data layer via Robinhood/robin_stocks.
    Provides OHLCV DataFrames, session info, and account data.
    """

    def __init__(self):
        self._cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self._cache_ttl = 60  # seconds before refetch
        self._login_ok = False

    # ─── Authentication ───────────────────────────────────────────────────

    def login(self, username: str, password: str, totp_secret: str = "") -> bool:
        """
        Login to Robinhood with optional 2FA TOTP.
        Returns True on success.
        """
        try:
            if totp_secret:
                import pyotp
                totp = pyotp.TOTP(totp_secret).now()
                result = rh.login(username, password, mfa_code=totp, store_session=True)
            else:
                result = rh.login(username, password, store_session=True)

            if result:
                self._login_ok = True
                logger.info(f"Robinhood login successful for {username}")
                return True
            else:
                logger.error("Robinhood login returned empty result")
                return False
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    def logout(self):
        try:
            rh.logout()
            logger.info("Logged out of Robinhood")
        except Exception:
            pass

    # ─── Market Data ──────────────────────────────────────────────────────

    def get_bars(
        self,
        symbol: str,
        interval: str = "5minute",
        span: str = "day",
        bounds: str = "regular",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a symbol.

        Valid intervals: '5minute', '10minute', 'hour', 'day', 'week'
        Valid spans:     'day', 'week', 'month', '3month', 'year', '5year'
        Valid bounds:    'regular', 'extended', 'trading'
          Note: extended/trading bounds only work with span='day'

        Returns DataFrame indexed by timestamp with O/H/L/C/V columns.
        """
        cache_key = f"{symbol}_{interval}_{span}_{bounds}"

        # Check cache
        if use_cache and cache_key in self._cache:
            df, fetched_at = self._cache[cache_key]
            age = (datetime.now() - fetched_at).total_seconds()
            if age < self._cache_ttl:
                return df

        try:
            raw = rh.get_stock_historicals(
                symbol,
                interval=interval,
                span=span,
                bounds=bounds,
            )

            if not raw or raw == [None]:
                logger.warning(f"No data for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(raw)

            # Rename columns
            df = df.rename(columns=FIELD_MAP)

            # Keep only OHLCV + timestamp
            cols = ["timestamp", "open", "high", "low", "close", "volume"]
            existing = [c for c in cols if c in df.columns]
            df = df[existing].copy()

            # Convert types
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

            # Parse and set timestamp index
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["timestamp"] = df["timestamp"].dt.tz_convert(EASTERN)
            df = df.set_index("timestamp").sort_index()

            # Drop nulls and validate
            df = df.dropna()
            df = self._validate(df, symbol)

            self._cache[cache_key] = (df, datetime.now())
            logger.debug(f"Fetched {len(df)} {interval} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return pd.DataFrame()

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the most recent trade price."""
        try:
            prices = rh.get_latest_price(symbol)
            if prices and prices[0]:
                return float(prices[0])
        except Exception as e:
            logger.error(f"Price fetch failed for {symbol}: {e}")
        return None

    def get_opening_candle(self, symbol: str) -> Optional[pd.Series]:
        """
        Get the 9:30-9:35 AM candle (first 5-min bar of the session).
        The ORB is built from this candle's high and low.
        """
        df = self.get_bars(symbol, interval="5minute", span="day", bounds="regular", use_cache=False)
        if df.empty:
            return None

        now = datetime.now(tz=EASTERN)
        today = now.date()

        # Filter to today only
        today_df = df[df.index.date == today]
        if today_df.empty:
            return None

        # First bar = 9:30 AM bar
        first_bar = today_df.iloc[0]

        # Verify it's actually the 9:30 bar
        bar_time = first_bar.name
        if bar_time.hour != 9 or bar_time.minute != 30:
            logger.warning(f"{symbol}: First bar is not 9:30 ({bar_time})")

        # ORB not complete until 9:35 passes
        orb_complete_time = bar_time.replace(minute=35, second=0)
        if now < orb_complete_time:
            logger.info(f"{symbol}: ORB still forming (wait until 9:35 AM)")
            return None

        logger.info(
            f"{symbol} Opening Candle: "
            f"O={first_bar['open']:.2f} H={first_bar['high']:.2f} "
            f"L={first_bar['low']:.2f} C={first_bar['close']:.2f}"
        )
        return first_bar

    def get_previous_day_levels(self, symbol: str) -> Dict[str, float]:
        """
        Get previous day's H/L/C — used for Box Theory and Failed Level strategies.
        Also returns the prior week high/low for larger structure.
        """
        try:
            df = self.get_bars(symbol, interval="day", span="week")
            if len(df) < 2:
                return {}

            prev = df.iloc[-2]  # Yesterday (today is still forming)
            return {
                "prev_high":  float(prev["high"]),
                "prev_low":   float(prev["low"]),
                "prev_close": float(prev["close"]),
                "prev_open":  float(prev["open"]),
            }
        except Exception as e:
            logger.error(f"Error getting daily levels for {symbol}: {e}")
            return {}

    def get_week_high_low(self, symbol: str) -> Dict[str, float]:
        """Get the rolling 5-day high and low (session range context)."""
        try:
            df = self.get_bars(symbol, interval="day", span="week")
            if df.empty:
                return {}
            return {
                "week_high": float(df["high"].max()),
                "week_low":  float(df["low"].min()),
            }
        except Exception:
            return {}

    def get_premarket_levels(self, symbol: str) -> Dict[str, float]:
        """
        Opt 6: Get today's pre-market high and low (4:00 AM – 9:30 AM ET).
        Uses 5-min bars with extended hours. Returns empty dict if unavailable.
        The pre-market range acts as a directional bias signal:
          - Pre-market high sweep by London session → bearish NY bias
          - Pre-market low sweep → bullish NY bias (buy the false breakdown)
        """
        try:
            df = self.get_bars(symbol, interval="5minute", span="day")
            if df.empty:
                return {}

            import pytz
            from datetime import time as dtime
            eastern = pytz.timezone("America/New_York")

            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC").tz_convert(eastern)
            else:
                df.index = df.index.tz_convert(eastern)

            # Filter to today's pre-market window: 4:00 AM – 9:29 AM ET
            pm_bars = df[
                (df.index.time >= dtime(4, 0)) &
                (df.index.time < dtime(9, 30))
            ]

            if pm_bars.empty:
                return {}

            pm_high = float(pm_bars["high"].max())
            pm_low  = float(pm_bars["low"].min())

            logger.info(f"{symbol}: Pre-market H={pm_high:.2f} L={pm_low:.2f} ({len(pm_bars)} bars)")
            return {
                "premarket_high": pm_high,
                "premarket_low":  pm_low,
                "premarket_bars": len(pm_bars),
            }
        except Exception as e:
            logger.warning(f"Could not fetch pre-market levels for {symbol}: {e}")
            return {}

    def get_atr(self, symbol: str, period: int = 14) -> float:
        """
        Average True Range on 5-min bars.
        Used for position sizing and FVG minimum size validation.
        """
        df = self.get_bars(symbol, interval="5minute", span="day")
        if len(df) < period + 1:
            # Fall back to daily ATR
            df = self.get_bars(symbol, interval="day", span="month")
            if len(df) < period + 1:
                return 0.0

        h, l, c_prev = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if not pd.isna(atr) else 0.0

    # ─── Session Utilities ────────────────────────────────────────────────

    def get_session_bars(self, symbol: str) -> pd.DataFrame:
        """
        Get all 5-minute bars from today's session.
        Filters strictly to regular trading hours (9:30 AM - 4:00 PM ET).
        """
        df = self.get_bars(symbol, interval="5minute", span="day", bounds="regular", use_cache=False)
        if df.empty:
            return df

        today = date.today()
        start = EASTERN.localize(datetime(today.year, today.month, today.day, 9, 30))
        end   = EASTERN.localize(datetime(today.year, today.month, today.day, 16, 0))

        return df[(df.index >= start) & (df.index <= end)]

    def is_market_open(self) -> bool:
        """Check if NYSE is currently open."""
        now = datetime.now(tz=EASTERN)
        if now.weekday() >= 5:
            return False
        open_t  = now.replace(hour=9, minute=30, second=0, microsecond=0)
        close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
        return open_t <= now <= close_t

    def minutes_to_open(self) -> int:
        """Minutes until market opens. 0 if already open."""
        now = datetime.now(tz=EASTERN)
        today = now.date()
        open_dt = EASTERN.localize(datetime(today.year, today.month, today.day, 9, 30))
        if now >= open_dt:
            return 0
        return max(0, int((open_dt - now).total_seconds() / 60))

    # ─── Account ──────────────────────────────────────────────────────────

    def get_account_equity(self) -> float:
        """Return total account equity (portfolio value)."""
        try:
            profile = rh.load_portfolio_profile()
            if profile:
                return float(profile.get("equity", 0) or 0)
        except Exception as e:
            logger.error(f"Error fetching equity: {e}")
        return 0.0

    def get_buying_power(self) -> float:
        """Return available buying power."""
        try:
            profile = rh.load_account_profile()
            if profile:
                return float(profile.get("buying_power", 0) or 0)
        except Exception as e:
            logger.error(f"Error fetching buying power: {e}")
        return 0.0

    def get_open_positions(self) -> List[dict]:
        """Return current open stock positions."""
        try:
            positions = rh.get_open_stock_positions()
            result = []
            for pos in (positions or []):
                if float(pos.get("quantity", 0)) > 0:
                    result.append({
                        "symbol":   rh.get_symbol_by_url(pos["instrument"]),
                        "quantity": float(pos["quantity"]),
                        "avg_cost": float(pos.get("average_buy_price", 0)),
                    })
            return result
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    # ─── Multi-Symbol Batch ────────────────────────────────────────────────

    def prefetch_all(self, symbols: List[str]):
        """
        Prefetch data for all symbols before market open.
        Spaces out requests to avoid rate limiting.
        """
        logger.info(f"Pre-fetching data for {len(symbols)} symbols...")
        for i, sym in enumerate(symbols):
            try:
                self.get_bars(sym, "5minute", "day")
                self.get_previous_day_levels(sym)
                if i % 5 == 4:
                    time.sleep(1)  # Brief pause every 5 requests
            except Exception as e:
                logger.warning(f"Prefetch failed for {sym}: {e}")
        logger.info("Prefetch complete")

    # ─── Validation ───────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Remove bad bars."""
        original = len(df)
        df = df[df["high"] >= df["low"]]
        df = df[(df["open"] >= df["low"]) & (df["open"] <= df["high"])]
        df = df[(df["close"] >= df["low"]) & (df["close"] <= df["high"])]
        df = df[df["volume"] > 0]
        removed = original - len(df)
        if removed:
            logger.warning(f"Removed {removed} bad bars for {symbol}")
        return df
