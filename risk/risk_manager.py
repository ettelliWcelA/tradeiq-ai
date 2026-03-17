"""
TradeIQ AI -- Risk Manager (Robinhood Edition)
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class Position:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    shares: float
    entry_time: datetime
    strategy: str
    quality_score: float
    status: str = "open"
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    rh_order_id: Optional[str] = None

    @property
    def is_open(self): return self.status == "open"
    @property
    def dollar_risk(self): return abs(self.entry_price - self.stop_loss) * self.shares


@dataclass
class DailyState:
    date: date
    starting_equity: float
    current_equity: float = 0.0
    realized_pnl: float = 0.0
    trades_taken: int = 0
    trades_won: int = 0
    kill_switch_hit: bool = False
    day_trades_used: int = 0

    @property
    def pnl_pct(self):
        return self.realized_pnl / self.starting_equity * 100 if self.starting_equity else 0.0
    @property
    def win_rate(self):
        return self.trades_won / self.trades_taken if self.trades_taken else 0.0


class RiskManager:
    def __init__(self, max_risk_pct=1.0, max_daily_loss_pct=3.0,
                 max_concurrent=3, default_rr=3.0, min_rr=2.0,
                 account_size=10000.0, pdt_safe_mode=True, paper_mode=False):
        self.max_risk_pct = max_risk_pct / 100
        self.max_daily_loss_pct = max_daily_loss_pct / 100
        self.max_concurrent = max_concurrent
        self.default_rr = default_rr
        self.min_rr = min_rr
        self.account_size = account_size
        self.pdt_safe_mode = pdt_safe_mode
        self.paper_mode = paper_mode
        self._state: Optional[DailyState] = None
        self._positions: Dict[str, Position] = {}
        self._history: List[Position] = []

    def can_trade(self, symbol, entry, stop, target, equity, quality_score=0.5):
        if self._state and self._state.kill_switch_hit:
            return False, "Kill switch active"
        if self._state:
            loss_pct = self._state.realized_pnl / self._state.starting_equity
            if loss_pct <= -self.max_daily_loss_pct:
                self._state.kill_switch_hit = True
                return False, f"Kill switch: {loss_pct:.2%} daily loss"
        if not self.paper_mode and self.pdt_safe_mode and equity < 25000:
            if self._state and self._state.day_trades_used >= 3:
                return False, "PDT limit reached (3 day trades used)"
        open_count = sum(1 for p in self._positions.values() if p.is_open)
        if open_count >= self.max_concurrent:
            return False, f"Max {self.max_concurrent} concurrent positions"
        if symbol in self._positions and self._positions[symbol].is_open:
            return False, f"Already in {symbol}"
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk <= 0:
            return False, "Zero risk distance"
        rr = reward / risk
        if rr < self.min_rr:
            return False, f"R:R {rr:.2f} below minimum {self.min_rr}"
        if not self.paper_mode and equity < 25000 and quality_score < 0.55:
            return False, f"Quality {quality_score:.0%} too low for PDT account"
        return True, "APPROVED"

    def calculate_shares(self, entry, stop, equity, quality_score=0.5):
        risk_dist = abs(entry - stop)
        if risk_dist <= 0 or entry <= 0:
            return 0.0, 0.0
        risk_pct = self.max_risk_pct * (1.0 if quality_score >= 0.75 else 0.75 if quality_score >= 0.60 else 0.50)
        dollar_risk = equity * risk_pct
        shares = dollar_risk / risk_dist
        shares = min(shares, (equity * 0.20) / entry)
        if equity < 25000:
            shares *= 0.8
        return round(shares, 6), round(dollar_risk, 2)

    def initialize_day(self, equity):
        self._state = DailyState(date=date.today(), starting_equity=equity, current_equity=equity)
        logger.info(f"Risk init: equity=${equity:,.2f} max_loss=${equity*self.max_daily_loss_pct:,.2f}")

    def record_open(self, pos):
        self._positions[pos.symbol] = pos
        if self._state:
            self._state.trades_taken += 1
            self._state.day_trades_used += 1

    def record_close(self, symbol, exit_price, reason="manual"):
        pos = self._positions.get(symbol)
        if not pos or not pos.is_open:
            return None
        pos.exit_price = exit_price
        pos.exit_time = datetime.now()
        pos.status = reason
        mult = 1 if pos.direction == "long" else -1
        pos.pnl = mult * (exit_price - pos.entry_price) * pos.shares
        pos.pnl_pct = mult * (exit_price - pos.entry_price) / pos.entry_price * 100
        if self._state:
            self._state.realized_pnl += pos.pnl
            self._state.current_equity += pos.pnl
            if pos.pnl > 0:
                self._state.trades_won += 1
        self._history.append(pos)
        return pos

    def get_summary(self):
        s = self._state
        if not s: return {}
        return {"date": str(s.date), "starting_equity": s.starting_equity,
                "realized_pnl": s.realized_pnl, "pnl_pct": s.pnl_pct,
                "trades_taken": s.trades_taken, "trades_won": s.trades_won,
                "win_rate": s.win_rate, "kill_switch": s.kill_switch_hit,
                "day_trades_used": s.day_trades_used}

    def get_open_positions(self): return [p for p in self._positions.values() if p.is_open]
    def get_history(self): return self._history.copy()
    def is_kill_switch(self): return self._state is not None and self._state.kill_switch_hit
