"""
TradeIQ AI — Robinhood Execution Engine
=========================================
Handles order placement, monitoring, and position management via robin_stocks.

CRITICAL ROBINHOOD LIMITATIONS vs. Alpaca:
  1. NO bracket orders — must place entry and stop separately
  2. NO native stop-limit on stocks (only stop-market)
  3. Fractional shares supported (min $1 notional)
  4. PDT rule: <3 round-trips in 5 days if account < $25k
  5. Market orders only during regular hours (9:30-4:00 ET)

PAPER MODE:
  When PAPER_MODE=true, all orders are SIMULATED — no real orders are sent.
  Everything is logged exactly as if real, allowing you to validate the system
  before committing real money.
"""

import time
from datetime import datetime
from typing import Optional, Tuple
import pytz

from loguru import logger
import robin_stocks.robinhood as rh

from risk.risk_manager import RiskManager, Position
from strategies.unified_strategy import TradeSetup

EASTERN = pytz.timezone("America/New_York")


class ExecutionEngine:

    def __init__(
        self,
        risk_manager: RiskManager,
        paper_mode: bool = True,
    ):
        self.risk = risk_manager
        self.paper = paper_mode
        self._audit: list = []

        mode = "PAPER (simulated — no real orders)" if paper_mode else "⚠️  LIVE TRADING"
        logger.info(f"ExecutionEngine initialized | Mode: {mode}")
        if not paper_mode:
            logger.warning("=" * 60)
            logger.warning("LIVE MODE: REAL MONEY. REAL ORDERS. NO UNDO.")
            logger.warning("=" * 60)

    # ─── Account ──────────────────────────────────────────────────────────

    def get_equity(self) -> float:
        if self.paper:
            return self.risk.account_size
        try:
            profile = rh.load_portfolio_profile()
            return float(profile.get("equity", 0) or 0)
        except Exception as e:
            logger.error(f"Error fetching equity: {e}")
            return self.risk.account_size

    # ─── Execute a Trade Setup ────────────────────────────────────────────

    def execute_setup(self, setup: TradeSetup) -> Optional[Position]:
        """
        Execute a TradeSetup:
        1. Risk manager approval
        2. Calculate shares
        3. Submit market order (or simulate in paper mode)
        4. Immediately place stop-loss order
        5. Record position
        """
        equity = self.get_equity()

        # Risk approval
        approved, reason = self.risk.can_trade(
            setup.symbol, setup.entry_price, setup.stop_loss,
            setup.target_2, equity, setup.quality_score
        )
        if not approved:
            logger.warning(f"Trade REJECTED [{setup.symbol}]: {reason}")
            return None

        shares, dollar_risk = self.risk.calculate_shares(
            setup.entry_price, setup.stop_loss, equity, setup.quality_score
        )
        if shares <= 0:
            logger.warning(f"Zero shares for {setup.symbol} — skipping")
            return None

        logger.info(
            f"Executing: {setup.symbol} {setup.direction.upper()} "
            f"{shares:.4f} shares @ ~${setup.entry_price:.2f} | "
            f"Stop: ${setup.stop_loss:.2f} | Target: ${setup.target_2:.2f} | "
            f"Risk: ${dollar_risk:.2f}"
        )

        # ── Submit Entry Order ─────────────────────────────────────────────
        entry_order_id = None
        actual_entry = setup.entry_price

        if self.paper:
            entry_order_id = f"PAPER-{setup.symbol}-{int(time.time())}"
            logger.info(f"[PAPER] {setup.direction.upper()} {shares:.4f} {setup.symbol} @ MARKET")
        else:
            try:
                if setup.direction == "long":
                    order = rh.order_buy_market(setup.symbol, shares)
                else:
                    order = rh.order_sell_market(setup.symbol, shares)

                if order and "id" in order:
                    entry_order_id = order["id"]
                    # Wait briefly for fill price
                    time.sleep(2)
                    try:
                        filled = rh.get_stock_order_info(entry_order_id)
                        if filled and filled.get("average_price"):
                            actual_entry = float(filled["average_price"])
                    except Exception:
                        pass
                else:
                    logger.error(f"Entry order failed for {setup.symbol}: {order}")
                    return None
            except Exception as e:
                logger.error(f"Entry order exception: {e}")
                return None

        # ── Place Stop Loss Order ──────────────────────────────────────────
        stop_order_id = self._place_stop_order(
            setup.symbol,
            setup.direction,
            shares,
            setup.stop_loss,
        )

        # ── Build Position Record ──────────────────────────────────────────
        position = Position(
            symbol=setup.symbol,
            direction=setup.direction,
            entry_price=actual_entry,
            stop_loss=setup.stop_loss,
            target_1=setup.target_1,
            target_2=setup.target_2,
            shares=shares,
            entry_time=datetime.now(tz=EASTERN),
            strategy=setup.strategy,
            quality_score=setup.quality_score,
            rh_order_id=entry_order_id,
        )

        self.risk.record_open(position)
        self._audit.append({
            "time": datetime.now(tz=EASTERN).isoformat(),
            "action": "OPEN",
            "symbol": setup.symbol,
            "direction": setup.direction,
            "shares": shares,
            "entry": actual_entry,
            "stop": setup.stop_loss,
            "target": setup.target_2,
            "strategy": setup.strategy,
            "quality": setup.quality_score,
            "paper": self.paper,
        })

        logger.info(f"Position opened: {setup.symbol} | Order ID: {entry_order_id}")
        return position

    # ─── Stop Loss Order ──────────────────────────────────────────────────

    def _place_stop_order(
        self, symbol: str, direction: str, shares: float, stop_price: float
    ) -> Optional[str]:
        """
        Place a stop-market order for the position's stop loss.
        For longs: stop-sell order below entry.
        For shorts: stop-buy order above entry.
        """
        if self.paper:
            order_id = f"PAPER-STOP-{symbol}-{int(time.time())}"
            logger.info(f"[PAPER] Stop order placed: {stop_price:.2f} for {symbol}")
            return order_id

        try:
            if direction == "long":
                order = rh.order_sell_stop_loss(symbol, shares, stop_price)
            else:
                order = rh.order_buy_stop_loss(symbol, shares, stop_price)

            if order and "id" in order:
                logger.info(f"Stop order placed: {order['id']} @ ${stop_price:.2f}")
                return order["id"]
            else:
                logger.error(f"Stop order failed for {symbol}: {order}")
                return None
        except Exception as e:
            logger.error(f"Stop order exception: {e}")
            return None

    # ─── Close Position ───────────────────────────────────────────────────

    def close_position(self, symbol: str, reason: str = "manual") -> bool:
        """
        Close a position at market.
        Also cancels any open stop order.
        """
        pos = self.risk._positions.get(symbol)
        if not pos or not pos.is_open:
            logger.warning(f"No open position to close: {symbol}")
            return False

        # Get current price
        try:
            prices = rh.get_latest_price(symbol)
            exit_price = float(prices[0]) if prices else pos.entry_price
        except Exception:
            exit_price = pos.entry_price

        if self.paper:
            logger.info(f"[PAPER] Close {symbol} @ ${exit_price:.2f} | Reason: {reason}")
        else:
            try:
                if pos.direction == "long":
                    rh.order_sell_market(symbol, pos.shares)
                else:
                    rh.order_buy_market(symbol, pos.shares)

                # Cancel stop order
                if pos.rh_order_id:
                    try:
                        rh.cancel_stock_order(pos.rh_order_id)
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Close order failed for {symbol}: {e}")
                return False

        closed = self.risk.record_close(symbol, exit_price, reason)
        if closed:
            self._audit.append({
                "time": datetime.now(tz=EASTERN).isoformat(),
                "action": "CLOSE",
                "symbol": symbol,
                "exit_price": exit_price,
                "pnl": closed.pnl,
                "reason": reason,
                "paper": self.paper,
            })
        return True

    def close_all_positions(self, reason: str = "eod"):
        """Close all open positions. Called at 3:45 PM."""
        logger.info(f"Closing all positions. Reason: {reason}")
        for symbol in list(self.risk._positions.keys()):
            pos = self.risk._positions.get(symbol)
            if pos and pos.is_open:
                self.close_position(symbol, reason)

    # ─── Monitor Targets ──────────────────────────────────────────────────

    def check_targets(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if a position has hit its stop or target.
        Robinhood doesn't bracket-order, so we monitor manually.
        Returns action taken ("stop_hit", "target_hit", None).
        """
        pos = self.risk._positions.get(symbol)
        if not pos or not pos.is_open:
            return None

        if pos.direction == "long":
            if current_price <= pos.stop_loss:
                logger.warning(f"{symbol}: STOP HIT @ ${current_price:.2f}")
                self.close_position(symbol, "stopped_out")
                return "stop_hit"
            if current_price >= pos.target_2:
                logger.info(f"{symbol}: TARGET 2 HIT @ ${current_price:.2f} 🎯")
                self.close_position(symbol, "target_hit")
                return "target_hit"
        else:
            if current_price >= pos.stop_loss:
                logger.warning(f"{symbol}: STOP HIT @ ${current_price:.2f}")
                self.close_position(symbol, "stopped_out")
                return "stop_hit"
            if current_price <= pos.target_2:
                logger.info(f"{symbol}: TARGET 2 HIT @ ${current_price:.2f} 🎯")
                self.close_position(symbol, "target_hit")
                return "target_hit"

        return None

    def get_audit_log(self) -> list:
        return self._audit.copy()
