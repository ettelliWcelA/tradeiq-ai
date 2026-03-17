"""
TradeIQ AI -- Unified Strategy Engine (v3)
==========================================
v3 adds: VWAP, RSI, EMA Trend Filter, Relative Volume,
Market Regime Detector, Stocks-in-Play, VWAP Pullback strategy,
RSI Divergence strategy, EMA Momentum Breakout strategy.
All times in ET internally; schedule fires on CT (your local clock).
"""
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
import pytz

EASTERN = pytz.timezone("America/New_York")


# ── Enums ──────────────────────────────────────────────────────────────────

class SessionPhase(Enum):
    PRE_MARKET   = "pre_market"
    ORB_FORMING  = "orb_forming"
    PRIME_WINDOW = "prime_window"
    MID_SESSION  = "mid_session"
    POWER_HOUR   = "power_hour"
    CLOSED       = "closed"

class MarketRegime(Enum):
    TRENDING_UP   = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING       = "ranging"
    VOLATILE      = "volatile"


# ── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class ORBLevels:
    high: float; low: float; open_price: float; close_price: float
    range_size: float; formed_at: object
    @property
    def midpoint(self): return (self.high + self.low) / 2

@dataclass
class FVGZone:
    high: float; low: float; direction: str; size: float
    formed_at: object; tested: bool = False
    @property
    def midpoint(self): return (self.high + self.low) / 2
    def contains(self, price, tol=0.002):
        m = (self.high - self.low) * tol
        return (self.low - m) <= price <= (self.high + m)

@dataclass
class TradeSetup:
    symbol: str; direction: str; strategy: str
    entry_price: float; stop_loss: float; target_1: float; target_2: float
    quality_score: float; signals: List[str] = field(default_factory=list)
    fvg: Optional[FVGZone] = None; regime: str = "unknown"
    vwap: Optional[float] = None; rsi: Optional[float] = None

    @property
    def risk_per_share(self): return abs(self.entry_price - self.stop_loss)
    @property
    def reward_per_share(self): return abs(self.target_2 - self.entry_price)
    @property
    def rr_ratio(self):
        r = self.risk_per_share
        return self.reward_per_share / r if r > 0 else 0

    def describe(self):
        return (f"{self.direction.upper()} {self.symbol} | {self.strategy} | "
                f"Q={self.quality_score:.0%} R:R={self.rr_ratio:.1f} | "
                f"Entry={self.entry_price:.2f} Stop={self.stop_loss:.2f} T2={self.target_2:.2f} | "
                f"RSI={self.rsi or 0:.0f} VWAP={self.vwap or 0:.2f} | "
                f"Signals: {' + '.join(self.signals)}")


# ── Indicators ─────────────────────────────────────────────────────────────

class Indicators:

    # Basic
    @staticmethod
    def body(r): return abs(float(r["close"]) - float(r["open"]))
    @staticmethod
    def rng(r): return float(r["high"]) - float(r["low"])
    @staticmethod
    def bullish(r): return float(r["close"]) > float(r["open"])
    @staticmethod
    def bearish(r): return float(r["close"]) < float(r["open"])

    # VWAP
    @staticmethod
    def vwap(df):
        if "volume" not in df.columns or df["volume"].astype(float).sum() == 0:
            return df["close"].astype(float)
        tp  = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3
        vol = df["volume"].astype(float)
        return (tp * vol).cumsum() / vol.cumsum()

    @staticmethod
    def vwap_bands(df, vwap_series, mult=1.5):
        tp  = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3
        vol = df["volume"].astype(float)
        cv  = vol.cumsum()
        var = ((tp - vwap_series)**2 * vol).cumsum() / cv
        sd  = np.sqrt(var)
        return vwap_series + mult*sd, vwap_series - mult*sd

    # RSI
    @staticmethod
    def rsi(df, period=14):
        close = df["close"].astype(float)
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period).mean()
        loss  = (-delta).clip(lower=0).ewm(alpha=1/period, min_periods=period).mean()
        rs    = gain / loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def rsi_divergence(df, rsi_series, lookback=5):
        if len(df) < lookback + 2: return None
        prices = df["close"].astype(float).values[-lookback:]
        rsis   = rsi_series.values[-lookback:]
        if prices[-1] < prices[:-1].min() and rsis[-1] > rsis[:-1].min(): return "bullish"
        if prices[-1] > prices[:-1].max() and rsis[-1] < rsis[:-1].max(): return "bearish"
        return None

    # EMA
    @staticmethod
    def ema(df, period):
        return df["close"].astype(float).ewm(span=period, adjust=False).mean()

    @staticmethod
    def ema_direction(df):
        if len(df) < 60: return "sideways"
        e8 = Indicators.ema(df,8).iloc[-1]; e21 = Indicators.ema(df,21).iloc[-1]
        e55 = Indicators.ema(df,55).iloc[-1]; p = float(df["close"].iloc[-1])
        if p > e8 > e21 > e55: return "up"
        if p < e8 < e21 < e55: return "down"
        return "sideways"

    @staticmethod
    def ema55_side(df):
        if len(df) < 60: return "unknown"
        e55 = Indicators.ema(df,55).iloc[-1]; p = float(df["close"].iloc[-1])
        return "above" if p > e55 else "below"

    # Volume
    @staticmethod
    def rvol(df, period=20):
        if "volume" not in df.columns or len(df) < period+1: return 1.0
        vols = df["volume"].astype(float)
        avg  = vols.iloc[-period-1:-1].mean()
        curr = vols.iloc[-1]
        return float(curr/avg) if avg > 0 else 1.0

    @staticmethod
    def in_play(df, thresh=2.0):
        return Indicators.rvol(df) >= thresh

    @staticmethod
    def vol_contracting(df, bars=3):
        if "volume" not in df.columns or len(df) < bars+1: return True
        v = df["volume"].astype(float).tail(bars).values
        return bool(v[-1] < v[0])

    # Regime
    @staticmethod
    def regime(df, atr):
        if len(df) < 20: return MarketRegime.RANGING
        p = float(df["close"].iloc[-1]); atr_pct = atr/p if p > 0 else 0
        if atr_pct > 0.025: return MarketRegime.VOLATILE
        d = Indicators.ema_direction(df)
        if d == "up":   return MarketRegime.TRENDING_UP
        if d == "down": return MarketRegime.TRENDING_DOWN
        highs = df["high"].astype(float).tail(10); lows = df["low"].astype(float).tail(10)
        if highs.max() - lows.min() < atr * 1.5: return MarketRegime.RANGING
        return MarketRegime.RANGING

    # Patterns
    @staticmethod
    def displacement(df, body_pct=0.65):
        if len(df) < 2: return None
        r = df.iloc[-1]; b = Indicators.body(r); t = Indicators.rng(r)
        if t == 0: return None
        if b/t >= body_pct: return "bullish" if Indicators.bullish(r) else "bearish"
        return None

    @staticmethod
    def structure_break(df, lookback=5):
        if len(df) < lookback+2: return None
        w = df.iloc[-lookback-1:-1]; l = df.iloc[-1]
        if float(l["close"]) > w["high"].astype(float).max(): return "bullish"
        if float(l["close"]) < w["low"].astype(float).min():  return "bearish"
        return None

    @staticmethod
    def momentum(df, candles=3, body_pct=0.60):
        if len(df) < candles: return None
        up = dn = 0
        for _, r in df.iloc[-candles:].iterrows():
            b = Indicators.body(r); t = Indicators.rng(r)
            if t == 0: continue
            if b/t >= body_pct:
                if Indicators.bullish(r): up += 1
                else: dn += 1
        if up >= candles: return "bullish"
        if dn >= candles: return "bearish"
        return None

    @staticmethod
    def swing_failure(df, orb, lookback=3):
        if not orb or len(df) < lookback+2: return None
        recent = df.iloc[-lookback:]; l = df.iloc[-1]
        if recent["high"].astype(float).max() > orb.high and float(l["close"]) < orb.high: return "bearish"
        if recent["low"].astype(float).min()  < orb.low  and float(l["close"]) > orb.low:  return "bullish"
        return None

    @staticmethod
    def fvg(df, atr, min_atr=0.35, body_pct=0.40):
        if len(df) < 3: return None
        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        min_sz = atr * min_atr
        bull_gap = float(c3["low"]) - float(c1["high"])
        if bull_gap > min_sz:
            b2 = Indicators.body(c2); r2 = Indicators.rng(c2)
            if r2 > 0 and b2/r2 >= body_pct and Indicators.bullish(c2):
                return FVGZone(float(c3["low"]), float(c1["high"]), "bullish", bull_gap, c3.name)
        bear_gap = float(c1["low"]) - float(c3["high"])
        if bear_gap > min_sz:
            b2 = Indicators.body(c2); r2 = Indicators.rng(c2)
            if r2 > 0 and b2/r2 >= body_pct and Indicators.bearish(c2):
                return FVGZone(float(c1["low"]), float(c3["high"]), "bearish", bear_gap, c3.name)
        return None

    @staticmethod
    def engulfing(df, min_ratio=1.10):
        if len(df) < 2: return None
        prev = df.iloc[-2]; curr = df.iloc[-1]
        pb = Indicators.body(prev); cb = Indicators.body(curr)
        if pb == 0: return None
        if cb/pb >= min_ratio:
            if Indicators.bullish(curr) and Indicators.bearish(prev): return "bullish"
            if Indicators.bearish(curr) and Indicators.bullish(prev): return "bearish"
        return None

    @staticmethod
    def box_confirm(df, direction):
        if len(df) < 2: return False
        c = df.iloc[-1]; p = df.iloc[-2]
        if direction == "long":  return Indicators.bullish(c) and float(c["close"]) > float(p["high"])
        return Indicators.bearish(c) and float(c["close"]) < float(p["low"])

    @staticmethod
    def vwap_retest(df, vwap_s, direction, tol=0.003):
        if len(df) < 3 or len(vwap_s) < 3: return False
        vw = float(vwap_s.iloc[-1]); p = float(df["close"].iloc[-1])
        lo = float(df["low"].iloc[-1]); hi = float(df["high"].iloc[-1])
        near = abs(p - vw) / vw <= tol
        touched = (lo <= vw*(1+tol)) if direction == "long" else (hi >= vw*(1-tol))
        return near or touched


@dataclass
class PreMarketLevels:
    high: float; low: float; symbol: str
    @property
    def range_size(self): return self.high - self.low
    @property
    def midpoint(self): return (self.high + self.low) / 2
    def direction_bias(self, price) -> str:
        """Returns 'long' if price above PM midpoint, 'short' if below."""
        return "long" if price >= self.midpoint else "short"
    def was_swept(self, df) -> Optional[str]:
        """Returns 'high' or 'low' if PM level was swept (taken then rejected) in session."""
        if len(df) < 3: return None
        highs = df["high"].astype(float)
        lows  = df["low"].astype(float)
        close = float(df["close"].iloc[-1])
        if highs.max() > self.high and close < self.high: return "high"
        if lows.min()  < self.low  and close > self.low:  return "low"
        return None


# ── Main Strategy Engine ───────────────────────────────────────────────────

class UnifiedStrategy:

    def __init__(self, config=None):
        cfg = config or {}
        self.min_rr         = cfg.get("min_rr", 2.0)
        self.default_rr     = cfg.get("default_rr", 3.0)
        self.min_fvg_atr    = cfg.get("min_fvg_atr", 0.45)   # Opt 1: raised from 0.35
        self.min_quality    = cfg.get("min_quality", 0.55)
        self.engulf_ratio   = cfg.get("engulf_ratio", 1.10)
        self.disp_body_pct  = cfg.get("disp_body_pct", 0.65)
        self.box_threshold  = cfg.get("box_threshold", 0.10)
        self.box_accept_n   = int(cfg.get("box_acceptance_candles", 2))  # Opt 7
        self.stop_atr_mult  = cfg.get("stop_atr_mult", 0.75)
        self.rvol_threshold = cfg.get("rvol_threshold", 1.5)
        self.fvg_rvol_min   = cfg.get("fvg_rvol_min", 1.5)   # Opt 2
        self.swing_require_disp = bool(cfg.get("swing_failure_require_displacement", 1))  # Opt 5
        self.premarket_enabled  = bool(cfg.get("premarket_track_enabled", 1))             # Opt 6
        self.premarket_bonus    = cfg.get("premarket_sweep_quality_bonus", 0.08)          # Opt 6

        self._orb:        Dict[str, ORBLevels]       = {}
        self._prev_high:  Dict[str, float]            = {}
        self._prev_low:   Dict[str, float]            = {}
        self._premarket:  Dict[str, PreMarketLevels]  = {}  # Opt 6

    def initialize_symbol(self, symbol, prev_high, prev_low):
        self._prev_high[symbol] = prev_high
        self._prev_low[symbol]  = prev_low

    def set_orb(self, symbol, orb):
        self._orb[symbol] = orb

    def set_premarket_levels(self, symbol, pm_high: float, pm_low: float):
        """Opt 6: Store pre-market high/low for directional bias."""
        if pm_high > 0 and pm_low > 0 and pm_high > pm_low:
            self._premarket[symbol] = PreMarketLevels(high=pm_high, low=pm_low, symbol=symbol)
            logger.info(f"{symbol}: PM levels set H={pm_high:.2f} L={pm_low:.2f}")

    def reset_symbol(self, symbol):
        self._orb.pop(symbol, None)

    # ── Main analyze ─────────────────────────────────────────────────

    def analyze(self, symbol, df, atr, price, phase):
        if df.empty or len(df) < 10: return None

        vwap_s  = Indicators.vwap(df)
        vwap    = float(vwap_s.iloc[-1])
        rsi_s   = Indicators.rsi(df)
        rsi     = float(rsi_s.iloc[-1])
        ema_dir = Indicators.ema_direction(df)
        e55     = Indicators.ema55_side(df)
        rv      = Indicators.rvol(df)
        sip     = Indicators.in_play(df)
        reg     = Indicators.regime(df, atr)
        rsi_div = Indicators.rsi_divergence(df, rsi_s)
        vb_u, vb_l = Indicators.vwap_bands(df, vwap_s)

        ctx = {
            "vwap": vwap, "vwap_series": vwap_s,
            "vwap_upper": float(vb_u.iloc[-1]), "vwap_lower": float(vb_l.iloc[-1]),
            "rsi": rsi, "rsi_divergence": rsi_div,
            "ema_direction": ema_dir, "ema55_side": e55,
            "rvol": rv, "in_play": sip, "regime": reg, "atr": atr,
            "premarket": self._premarket.get(symbol),  # Opt 6
        }

        candidates = []

        if phase in (SessionPhase.PRIME_WINDOW, SessionPhase.ORB_FORMING):
            s = self._orb_fvg(symbol, df, price, ctx)
            if s: candidates.append(s)

        if reg in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            s = self._vwap_pullback(symbol, df, price, ctx)
            if s: candidates.append(s)

        s = self._box_theory(symbol, df, price, ctx)
        if s: candidates.append(s)

        s = self._swing_failure(symbol, df, price, ctx)
        if s: candidates.append(s)

        s = self._rsi_divergence(symbol, df, price, ctx)
        if s: candidates.append(s)

        if ema_dir in ("up", "down"):
            s = self._ema_momentum(symbol, df, price, ctx)
            if s: candidates.append(s)

        if not candidates: return None
        best = max(candidates, key=lambda x: x.quality_score)
        if best.quality_score < self.min_quality: return None
        logger.info(f"Setup: {best.describe()}")
        return best

    # ── Quality Scorer ────────────────────────────────────────────────

    def _score(self, direction, ctx, extra):
        score = 0.50; signals = []
        vwap = ctx["vwap"]; rsi = ctx["rsi"]; e55 = ctx["ema55_side"]
        rv = ctx["rvol"]; reg = ctx["regime"]; sip = ctx["in_play"]
        rsi_div = ctx["rsi_divergence"]

        if direction == "long"  and e55 == "above": score += 0.10; signals.append("VWAP_ABOVE")
        if direction == "short" and e55 == "below": score += 0.10; signals.append("VWAP_BELOW")

        if 30 < rsi < 70: score += 0.05; signals.append(f"RSI_{rsi:.0f}")
        if direction == "long"  and rsi > 70: score -= 0.08; signals.append("RSI_OVERBOUGHT")
        if direction == "short" and rsi < 30: score -= 0.08; signals.append("RSI_OVERSOLD")

        if direction == "long"  and e55 == "above": score += 0.08; signals.append("EMA55_BULL")
        elif direction == "short" and e55 == "below": score += 0.08; signals.append("EMA55_BEAR")

        if rv >= 2.0: score += 0.07; signals.append(f"RVOL_{rv:.1f}x")
        elif rv >= self.rvol_threshold: score += 0.04; signals.append(f"RVOL_{rv:.1f}x")

        if sip: score += 0.15; signals.append("IN_PLAY")

        if rsi_div == "bullish" and direction == "long":  score += 0.08; signals.append("RSI_BULL_DIV")
        if rsi_div == "bearish" and direction == "short": score += 0.08; signals.append("RSI_BEAR_DIV")

        if reg == MarketRegime.TRENDING_UP   and direction == "long":  score += 0.05; signals.append("TREND_UP")
        if reg == MarketRegime.TRENDING_DOWN and direction == "short": score += 0.05; signals.append("TREND_DOWN")
        if reg == MarketRegime.VOLATILE: score -= 0.05

        # Opt 6: Pre-market sweep alignment bonus
        if self.premarket_enabled:
            pm: Optional[PreMarketLevels] = ctx.get("premarket")
            if pm:
                swept = pm.was_swept(ctx.get("_df")) if ctx.get("_df") is not None else None
                if swept == "high" and direction == "short":
                    score += self.premarket_bonus; signals.append("PM_HIGH_SWEPT")
                elif swept == "low" and direction == "long":
                    score += self.premarket_bonus; signals.append("PM_LOW_SWEPT")
                # Directional bias bonus (softer signal)
                elif pm.direction_bias(vwap) == direction:
                    score += self.premarket_bonus * 0.5; signals.append("PM_BIAS_ALIGN")

        signals.extend(extra)
        return min(score, 1.0), signals

    def _stops(self, entry, direction, atr, rr=None):
        rr = rr or self.default_rr
        sd = atr * self.stop_atr_mult
        if direction == "long":
            return entry - sd, entry + sd*1.5, entry + sd*rr
        return entry + sd, entry - sd*1.5, entry - sd*rr

    # ── Strategy 1: ORB + FVG + Engulfing ────────────────────────────

    def _orb_fvg(self, symbol, df, price, ctx):
        orb = self._orb.get(symbol)
        if not orb: return None
        atr = ctx["atr"]
        direction = None
        if price > orb.high * 1.001: direction = "long"
        elif price < orb.low  * 0.999: direction = "short"
        if not direction: return None

        fvg_z = Indicators.fvg(df, atr, self.min_fvg_atr)
        if not fvg_z: return None
        if fvg_z.direction != ("bullish" if direction == "long" else "bearish"): return None

        # Opt 2: FVG must come with volume surge (institutional = volume)
        if ctx["rvol"] < self.fvg_rvol_min:
            logger.debug(f"{symbol}: FVG rejected — RVOL {ctx['rvol']:.1f}x < {self.fvg_rvol_min}x minimum")
            return None

        eng = Indicators.engulfing(df, self.engulf_ratio)
        if not eng: return None

        extra = ["ORB_BREAK", f"FVG_{fvg_z.size:.2f}", "ENGULFING"]
        sb = Indicators.structure_break(df)
        if sb and sb == ("bullish" if direction=="long" else "bearish"): extra.append("STRUCT_BREAK")
        mom = Indicators.momentum(df, 3, self.disp_body_pct)
        if mom and mom == ("bullish" if direction=="long" else "bearish"): extra.append("MOMENTUM")

        ctx["_df"] = df  # for premarket sweep check
        score, signals = self._score(direction, ctx, extra)
        if fvg_z.size >= atr * 0.50: score = min(score+0.10, 1.0); signals.append("FVG_LARGE")

        stop, t1, t2 = self._stops(price, direction, atr)
        return TradeSetup(symbol, direction, "ORB_FVG_ENGULF", price, stop, t1, t2,
                          score, signals, fvg_z, ctx["regime"].value, ctx["vwap"], ctx["rsi"])

    # ── Strategy 2: VWAP Pullback ─────────────────────────────────────

    def _vwap_pullback(self, symbol, df, price, ctx):
        reg = ctx["regime"]; ema_dir = ctx["ema_direction"]; rsi = ctx["rsi"]; atr = ctx["atr"]
        direction = None
        if reg == MarketRegime.TRENDING_UP   and ema_dir == "up":   direction = "long"
        if reg == MarketRegime.TRENDING_DOWN and ema_dir == "down": direction = "short"
        if not direction: return None

        if not Indicators.vwap_retest(df, ctx["vwap_series"], direction): return None
        if not Indicators.vol_contracting(df, 3): return None
        if direction == "long"  and rsi > 65: return None
        if direction == "short" and rsi < 35: return None

        extra = ["VWAP_PULLBACK", "VOL_CONTRACT", "EMA_TREND"]
        score, signals = self._score(direction, ctx, extra)
        prox = abs(price - ctx["vwap"]) / ctx["vwap"]
        if prox < 0.002: score = min(score+0.05, 1.0); signals.append("TIGHT_VWAP")

        stop, t1, t2 = self._stops(price, direction, atr)
        return TradeSetup(symbol, direction, "VWAP_PULLBACK", price, stop, t1, t2,
                          score, signals, None, ctx["regime"].value, ctx["vwap"], ctx["rsi"])

    # ── Strategy 3: Box Theory ────────────────────────────────────────

    def _box_theory(self, symbol, df, price, ctx):
        ph = self._prev_high.get(symbol, 0); pl = self._prev_low.get(symbol, 0)
        if not ph or not pl: return None
        box_range = ph - pl
        if box_range <= 0: return None
        atr = ctx["atr"]; thresh = box_range * self.box_threshold
        direction = None
        if price <= pl + thresh: direction = "long"
        elif price >= ph - thresh: direction = "short"
        if not direction: return None
        if not Indicators.box_confirm(df, direction): return None

        # Opt 7: Acceptance filter — require N candle closes beyond box edge
        if self.box_accept_n > 1:
            if direction == "long":
                # Need closes above prev_low level (acceptance of support)
                recent_closes = df["close"].astype(float).tail(self.box_accept_n)
                if not all(c > pl - (atr * 0.1) for c in recent_closes):
                    logger.debug(f"{symbol}: BOX_THEORY rejected — insufficient acceptance candles")
                    return None
            else:
                recent_closes = df["close"].astype(float).tail(self.box_accept_n)
                if not all(c < ph + (atr * 0.1) for c in recent_closes):
                    logger.debug(f"{symbol}: BOX_THEORY rejected — insufficient acceptance candles")
                    return None

        extra = ["BOX_THEORY", f"BOX_{box_range:.2f}"]
        ctx["_df"] = df  # for premarket sweep check
        score, signals = self._score(direction, ctx, extra)
        if ctx["regime"] == MarketRegime.RANGING:
            score = min(score+0.07, 1.0); signals.append("RANGING_BONUS")

        stop, t1, t2 = self._stops(price, direction, atr)
        return TradeSetup(symbol, direction, "BOX_THEORY", price, stop, t1, t2,
                          score, signals, None, ctx["regime"].value, ctx["vwap"], ctx["rsi"])

    # ── Strategy 4: Swing Failure ─────────────────────────────────────

    def _swing_failure(self, symbol, df, price, ctx):
        orb = self._orb.get(symbol)
        sf  = Indicators.swing_failure(df, orb)
        if not sf: return None

        # Opt 5: Require displacement candle as secondary confirmation
        if self.swing_require_disp:
            disp = Indicators.displacement(df, self.disp_body_pct)
            if not disp or disp != ("bullish" if sf == "bullish" else "bearish"):
                logger.debug(f"{symbol}: SWING_FAILURE rejected — no displacement candle confirmation")
                return None

        extra = ["SWING_FAIL"]
        if ctx["rvol"] >= 1.5: extra.append("SWEEP_VOL")
        if self.swing_require_disp: extra.append("DISP_CONFIRMED")
        ctx["_df"] = df
        score, signals = self._score(sf, ctx, extra)
        stop, t1, t2   = self._stops(price, sf, ctx["atr"])
        return TradeSetup(symbol, sf, "SWING_FAILURE", price, stop, t1, t2,
                          score, signals, None, ctx["regime"].value, ctx["vwap"], ctx["rsi"])

    # ── Strategy 5: RSI Divergence Reversal ───────────────────────────

    def _rsi_divergence(self, symbol, df, price, ctx):
        div = ctx["rsi_divergence"]
        if not div: return None
        atr = ctx["atr"]; vwap = ctx["vwap"]; rsi = ctx["rsi"]
        direction = "long" if div == "bullish" else "short"
        near_vwap = abs(price - vwap) / vwap < 0.005
        near_prev = (abs(price - self._prev_high.get(symbol,0)) < atr or
                     abs(price - self._prev_low.get(symbol,0))  < atr)
        if not (near_vwap or near_prev): return None
        if direction == "long"  and rsi >= 35: return None
        if direction == "short" and rsi <= 65: return None
        extra = [f"RSI_DIV_{div.upper()}"]
        if near_vwap: extra.append("NEAR_VWAP")
        if near_prev: extra.append("KEY_LEVEL")
        score, signals = self._score(direction, ctx, extra)
        stop, t1, t2   = self._stops(price, direction, atr, rr=2.5)
        return TradeSetup(symbol, direction, "RSI_DIVERGENCE", price, stop, t1, t2,
                          score, signals, None, ctx["regime"].value, vwap, rsi)

    # ── Strategy 6: EMA Momentum Breakout ─────────────────────────────

    def _ema_momentum(self, symbol, df, price, ctx):
        if len(df) < 60: return None
        e8 = Indicators.ema(df,8); e21 = Indicators.ema(df,21); e55 = Indicators.ema(df,55)
        cross_up   = float(e8.iloc[-1]) > float(e21.iloc[-1]) and float(e8.iloc[-2]) <= float(e21.iloc[-2])
        cross_down = float(e8.iloc[-1]) < float(e21.iloc[-1]) and float(e8.iloc[-2]) >= float(e21.iloc[-2])
        if not (cross_up or cross_down): return None
        direction = "long" if cross_up else "short"
        if direction == "long"  and ctx["ema55_side"] != "above": return None
        if direction == "short" and ctx["ema55_side"] != "below": return None
        if ctx["rvol"] < self.rvol_threshold: return None
        extra = ["EMA_CROSS", f"EMA55_{float(e55.iloc[-1]):.2f}"]
        score, signals = self._score(direction, ctx, extra)
        stop, t1, t2   = self._stops(price, direction, ctx["atr"])
        return TradeSetup(symbol, direction, "EMA_MOMENTUM", price, stop, t1, t2,
                          score, signals, None, ctx["regime"].value, ctx["vwap"], ctx["rsi"])

    # ── Session Phase ──────────────────────────────────────────────────

    def get_session_phase(self):
        t = datetime.now(tz=EASTERN).time()
        if t < time(9,30):  return SessionPhase.PRE_MARKET
        if t < time(9,35):  return SessionPhase.ORB_FORMING
        if t < time(10,30): return SessionPhase.PRIME_WINDOW
        if t < time(15,0):  return SessionPhase.MID_SESSION
        if t < time(16,0):  return SessionPhase.POWER_HOUR
        return SessionPhase.CLOSED
