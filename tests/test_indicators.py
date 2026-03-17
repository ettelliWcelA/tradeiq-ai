"""
tests/test_indicators.py
========================
Unit tests for strategies/unified_strategy.py — Indicators class, data classes,
enums, and the TradeSetup property calculations.
"""
import sys
import os
import unittest
import pandas as pd
import numpy as np

# ── Ensure stubs are registered before importing TradeIQ modules ──────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import mock_deps  # noqa: F401

from strategies.unified_strategy import (
    Indicators, TradeSetup, FVGZone, ORBLevels,
    MarketRegime, SessionPhase,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_df(closes, highs=None, lows=None, opens=None, volumes=None):
    """Build a minimal OHLCV DataFrame for testing."""
    n = len(closes)
    closes = np.array(closes, dtype=float)
    highs   = np.array(highs,   dtype=float) if highs   is not None else closes + 0.5
    lows    = np.array(lows,    dtype=float) if lows    is not None else closes - 0.5
    opens   = np.array(opens,   dtype=float) if opens   is not None else closes - 0.1
    volumes = np.array(volumes, dtype=float) if volumes is not None else np.ones(n) * 1_000_000
    return pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    })


# ── TestFVGZone ───────────────────────────────────────────────────────────────

class TestFVGZone(unittest.TestCase):
    def _zone(self, hi=10.0, lo=8.0, direction="bullish"):
        return FVGZone(high=hi, low=lo, direction=direction, size=hi-lo, formed_at=None)

    def test_midpoint(self):
        z = self._zone(10.0, 8.0)
        self.assertAlmostEqual(z.midpoint, 9.0)

    def test_contains_exact_midpoint(self):
        z = self._zone(10.0, 8.0)
        self.assertTrue(z.contains(9.0))

    def test_contains_within_tolerance(self):
        z = self._zone(10.0, 8.0)
        # tol=0.002 → margin = (10-8)*0.002 = 0.004
        self.assertTrue(z.contains(7.997))   # just below low within 0.004 margin
        self.assertTrue(z.contains(10.003))  # just above high within 0.004 margin

    def test_not_contains_far_outside(self):
        z = self._zone(10.0, 8.0)
        self.assertFalse(z.contains(5.0))
        self.assertFalse(z.contains(15.0))


# ── TestORBLevels ─────────────────────────────────────────────────────────────

class TestORBLevels(unittest.TestCase):
    def _orb(self):
        return ORBLevels(high=105.0, low=100.0, open_price=101.0,
                         close_price=104.0, range_size=5.0, formed_at=None)

    def test_midpoint(self):
        orb = self._orb()
        self.assertAlmostEqual(orb.midpoint, 102.5)

    def test_range_size(self):
        orb = self._orb()
        self.assertEqual(orb.range_size, 5.0)


# ── TestTradeSetupProperties ─────────────────────────────────────────────────

class TestTradeSetupProperties(unittest.TestCase):
    def _setup(self, direction="long", entry=100.0, stop=98.0, t2=106.0):
        return TradeSetup(
            symbol="TEST", direction=direction, strategy="TEST_STRAT",
            entry_price=entry, stop_loss=stop, target_1=102.0, target_2=t2,
            quality_score=0.65,
        )

    def test_risk_per_share_long(self):
        s = self._setup("long", 100.0, 98.0)
        self.assertAlmostEqual(s.risk_per_share, 2.0)

    def test_risk_per_share_short(self):
        s = self._setup("short", 100.0, 102.0)
        self.assertAlmostEqual(s.risk_per_share, 2.0)

    def test_reward_per_share_long(self):
        s = self._setup("long", 100.0, 98.0, t2=106.0)
        self.assertAlmostEqual(s.reward_per_share, 6.0)

    def test_rr_ratio_long(self):
        s = self._setup("long", 100.0, 98.0, t2=106.0)
        self.assertAlmostEqual(s.rr_ratio, 3.0)

    def test_rr_ratio_zero_risk(self):
        s = self._setup("long", 100.0, 100.0, t2=106.0)  # entry == stop
        self.assertEqual(s.rr_ratio, 0)

    def test_describe_contains_key_info(self):
        s = self._setup()
        desc = s.describe()
        self.assertIn("LONG", desc)
        self.assertIn("TEST", desc)
        self.assertIn("TEST_STRAT", desc)


# ── TestIndicatorsBasic ───────────────────────────────────────────────────────

class TestIndicatorsBasic(unittest.TestCase):
    def _row(self, o, h, l, c):
        return pd.Series({"open": o, "high": h, "low": l, "close": c})

    def test_body_bullish(self):
        self.assertAlmostEqual(Indicators.body(self._row(100, 105, 98, 103)), 3.0)

    def test_body_bearish(self):
        self.assertAlmostEqual(Indicators.body(self._row(103, 105, 98, 100)), 3.0)

    def test_rng(self):
        self.assertAlmostEqual(Indicators.rng(self._row(100, 110, 90, 105)), 20.0)

    def test_bullish_true(self):
        self.assertTrue(Indicators.bullish(self._row(100, 110, 95, 108)))

    def test_bullish_false(self):
        self.assertFalse(Indicators.bullish(self._row(108, 110, 95, 100)))

    def test_bearish_true(self):
        self.assertTrue(Indicators.bearish(self._row(108, 110, 95, 100)))

    def test_bearish_false(self):
        self.assertFalse(Indicators.bearish(self._row(100, 110, 95, 108)))


# ── TestVWAP ─────────────────────────────────────────────────────────────────

class TestVWAP(unittest.TestCase):
    def test_vwap_constant_price_equals_price(self):
        df = make_df([100]*20, highs=[101]*20, lows=[99]*20, volumes=[1_000_000]*20)
        vwap = Indicators.vwap(df)
        self.assertTrue(all(abs(vwap - 100.0) < 0.01))

    def test_vwap_no_volume_column(self):
        df = pd.DataFrame({"open": [100]*5, "high": [101]*5, "low": [99]*5, "close": [100]*5})
        vwap = Indicators.vwap(df)
        self.assertEqual(len(vwap), 5)

    def test_vwap_zero_volume(self):
        df = make_df([100]*5, volumes=[0]*5)
        vwap = Indicators.vwap(df)
        self.assertEqual(len(vwap), 5)

    def test_vwap_weighted_higher_price_shifts_up(self):
        # Two candles: one at 100, one at 200 with double volume
        df = make_df([100, 200], highs=[101, 201], lows=[99, 199], volumes=[1, 2])
        vwap = Indicators.vwap(df)
        self.assertGreater(float(vwap.iloc[-1]), 150.0)

    def test_vwap_bands_upper_above_lower(self):
        df = make_df(list(range(90, 110)), volumes=[1_000_000]*20)
        vwap_s = Indicators.vwap(df)
        upper, lower = Indicators.vwap_bands(df, vwap_s)
        self.assertTrue(all(upper >= lower))


# ── TestRSI ──────────────────────────────────────────────────────────────────

class TestRSI(unittest.TestCase):
    def test_rsi_range(self):
        closes = list(range(100, 130))  # 30 bars, consistently up
        df = make_df(closes)
        rsi = Indicators.rsi(df)
        vals = rsi.dropna().values
        self.assertTrue(all(0 <= v <= 100 for v in vals))

    def test_rsi_high_on_uptrend(self):
        closes = list(range(100, 130))
        df = make_df(closes)
        rsi = Indicators.rsi(df)
        self.assertGreater(float(rsi.iloc[-1]), 60)

    def test_rsi_low_on_downtrend(self):
        closes = list(range(130, 100, -1))
        df = make_df(closes)
        rsi = Indicators.rsi(df)
        self.assertLess(float(rsi.iloc[-1]), 40)

    def test_rsi_divergence_bullish(self):
        # Price makes lower lows, RSI makes higher lows → bullish divergence
        closes = [100, 105, 103, 107, 104, 98, 101, 97]  # new low at end
        df = make_df(closes)
        rsi_s = Indicators.rsi(df)
        # Just test it returns a string or None (structure test)
        result = Indicators.rsi_divergence(df, rsi_s)
        self.assertIn(result, [None, "bullish", "bearish"])

    def test_rsi_divergence_too_short(self):
        df = make_df([100, 101, 102])
        rsi_s = Indicators.rsi(df)
        result = Indicators.rsi_divergence(df, rsi_s)
        self.assertIsNone(result)


# ── TestEMA ──────────────────────────────────────────────────────────────────

class TestEMA(unittest.TestCase):
    def test_ema_length_matches_df(self):
        df = make_df(list(range(100, 160)))
        ema = Indicators.ema(df, 21)
        self.assertEqual(len(ema), len(df))

    def test_ema_direction_up_on_uptrend(self):
        closes = list(range(50, 170))  # 120 bars of uptrend
        df = make_df(closes, highs=[c+1 for c in closes], lows=[c-1 for c in closes])
        result = Indicators.ema_direction(df)
        self.assertEqual(result, "up")

    def test_ema_direction_down_on_downtrend(self):
        closes = list(range(170, 50, -1))
        df = make_df(closes, highs=[c+1 for c in closes], lows=[c-1 for c in closes])
        result = Indicators.ema_direction(df)
        self.assertEqual(result, "down")

    def test_ema_direction_sideways_too_short(self):
        df = make_df([100]*10)
        result = Indicators.ema_direction(df)
        self.assertEqual(result, "sideways")

    def test_ema55_side_above(self):
        # Price > EMA55 → above
        closes = list(range(80, 170))  # end much higher
        df = make_df(closes, highs=[c+1 for c in closes], lows=[c-1 for c in closes])
        result = Indicators.ema55_side(df)
        self.assertEqual(result, "above")

    def test_ema55_side_below(self):
        closes = list(range(170, 80, -1))
        df = make_df(closes, highs=[c+1 for c in closes], lows=[c-1 for c in closes])
        result = Indicators.ema55_side(df)
        self.assertEqual(result, "below")

    def test_ema55_side_unknown_too_short(self):
        df = make_df([100]*5)
        result = Indicators.ema55_side(df)
        self.assertEqual(result, "unknown")


# ── TestRelativeVolume ────────────────────────────────────────────────────────

class TestRelativeVolume(unittest.TestCase):
    def test_rvol_high_volume_spike(self):
        # 20 bars of baseline 1M volume, then 3M spike
        vols = [1_000_000]*20 + [3_000_000]
        df = make_df([100]*21, volumes=vols)
        rv = Indicators.rvol(df)
        self.assertAlmostEqual(rv, 3.0, places=1)

    def test_rvol_average_volume(self):
        vols = [1_000_000]*21
        df = make_df([100]*21, volumes=vols)
        rv = Indicators.rvol(df)
        self.assertAlmostEqual(rv, 1.0, places=1)

    def test_rvol_too_short_returns_one(self):
        df = make_df([100]*5)
        rv = Indicators.rvol(df)
        self.assertEqual(rv, 1.0)

    def test_in_play_true(self):
        vols = [1_000_000]*20 + [3_000_000]
        df = make_df([100]*21, volumes=vols)
        self.assertTrue(Indicators.in_play(df, thresh=2.0))

    def test_in_play_false(self):
        vols = [1_000_000]*21
        df = make_df([100]*21, volumes=vols)
        self.assertFalse(Indicators.in_play(df, thresh=2.0))

    def test_vol_contracting_true(self):
        # Volume decreasing over last 3 bars
        vols = [1_000_000]*10 + [900_000, 800_000, 700_000]
        df = make_df([100]*13, volumes=vols)
        self.assertTrue(Indicators.vol_contracting(df, bars=3))

    def test_vol_contracting_false(self):
        vols = [1_000_000]*10 + [700_000, 800_000, 900_000]
        df = make_df([100]*13, volumes=vols)
        self.assertFalse(Indicators.vol_contracting(df, bars=3))


# ── TestRegime ────────────────────────────────────────────────────────────────

class TestRegime(unittest.TestCase):
    def test_regime_volatile_high_atr(self):
        df = make_df([100]*25)
        regime = Indicators.regime(df, atr=5.0)  # 5% of 100 = 5% > 2.5% threshold
        self.assertEqual(regime, MarketRegime.VOLATILE)

    def test_regime_trending_up(self):
        closes = list(range(50, 170))
        df = make_df(closes, highs=[c+1 for c in closes], lows=[c-1 for c in closes])
        regime = Indicators.regime(df, atr=1.0)  # small atr, not volatile
        self.assertEqual(regime, MarketRegime.TRENDING_UP)

    def test_regime_trending_down(self):
        closes = list(range(170, 50, -1))
        df = make_df(closes, highs=[c+1 for c in closes], lows=[c-1 for c in closes])
        regime = Indicators.regime(df, atr=1.0)
        self.assertEqual(regime, MarketRegime.TRENDING_DOWN)

    def test_regime_too_short(self):
        df = make_df([100]*5)
        regime = Indicators.regime(df, atr=1.0)
        self.assertEqual(regime, MarketRegime.RANGING)


# ── TestPatternIndicators ─────────────────────────────────────────────────────

class TestDisplacement(unittest.TestCase):
    def test_displacement_bullish(self):
        # Last candle: big body bullish (close >> open)
        closes = [100]*5 + [105]
        opens  = [100]*5 + [100]
        highs  = [101]*5 + [106]
        lows   = [99] *5 + [99]
        df = make_df(closes, highs=highs, lows=lows, opens=opens)
        result = Indicators.displacement(df, body_pct=0.65)
        self.assertEqual(result, "bullish")

    def test_displacement_none_small_body(self):
        # Doji — no displacement
        closes = [100]*5 + [100.1]
        opens  = [100]*5 + [100.0]
        highs  = [101]*5 + [105]
        lows   = [99] *5 + [95]
        df = make_df(closes, highs=highs, lows=lows, opens=opens)
        result = Indicators.displacement(df)
        self.assertIsNone(result)

    def test_displacement_too_short(self):
        df = make_df([100])
        self.assertIsNone(Indicators.displacement(df))


class TestEngulfing(unittest.TestCase):
    def test_engulfing_bullish(self):
        # prev = bearish small, curr = bullish large
        opens  = [102, 100]
        closes = [100, 103]
        highs  = [103, 104]
        lows   = [99,  99]
        df = make_df(closes, highs=highs, lows=lows, opens=opens)
        result = Indicators.engulfing(df, min_ratio=1.10)
        self.assertEqual(result, "bullish")

    def test_engulfing_bearish(self):
        opens  = [100, 103]
        closes = [102, 99]
        highs  = [103, 104]
        lows   = [99,  98]
        df = make_df(closes, highs=highs, lows=lows, opens=opens)
        result = Indicators.engulfing(df, min_ratio=1.10)
        self.assertEqual(result, "bearish")

    def test_no_engulfing_same_direction(self):
        # Both bullish — no engulfing pattern
        opens  = [100, 101]
        closes = [101, 103]
        highs  = [102, 104]
        lows   = [99,  100]
        df = make_df(closes, highs=highs, lows=lows, opens=opens)
        result = Indicators.engulfing(df, min_ratio=1.10)
        self.assertIsNone(result)

    def test_engulfing_too_short(self):
        df = make_df([100])
        self.assertIsNone(Indicators.engulfing(df))


class TestFVGDetection(unittest.TestCase):
    def test_fvg_bullish_gap(self):
        # c1 high < c3 low → bullish FVG
        opens  = [100, 101, 103]
        closes = [101, 103, 105]
        highs  = [102, 104, 106]
        lows   = [99,  100, 104]  # c3 low (104) > c1 high (102) → gap
        df = make_df(closes, highs=highs, lows=lows, opens=opens)
        atr = 1.0
        result = Indicators.fvg(df, atr, min_atr=0.35)
        # Gap = 104-102 = 2.0, min_sz = 0.35, so should detect
        self.assertIsNotNone(result)
        if result:
            self.assertEqual(result.direction, "bullish")

    def test_fvg_too_short(self):
        df = make_df([100, 101])
        self.assertIsNone(Indicators.fvg(df, 1.0))

    def test_fvg_no_gap(self):
        # Candles overlap, no FVG
        opens  = [100, 101, 100]
        closes = [101, 102, 101]
        highs  = [102, 103, 102]
        lows   = [99,  100, 99]
        df = make_df(closes, highs=highs, lows=lows, opens=opens)
        result = Indicators.fvg(df, 1.0, min_atr=0.35)
        self.assertIsNone(result)


class TestSwingFailure(unittest.TestCase):
    def test_swing_failure_bearish(self):
        orb = ORBLevels(high=105.0, low=95.0, open_price=100.0,
                        close_price=102.0, range_size=10.0, formed_at=None)
        # Recent highs exceed ORB high, last close below ORB high → bearish SF
        closes = [102, 103, 106, 104, 103]  # exceeded 105, then pulled back
        highs  = [103, 104, 107, 105, 104]
        lows   = [101, 102, 105, 103, 102]
        df = make_df(closes, highs=highs, lows=lows)
        result = Indicators.swing_failure(df, orb)
        self.assertEqual(result, "bearish")

    def test_swing_failure_no_orb(self):
        df = make_df([100]*5)
        self.assertIsNone(Indicators.swing_failure(df, None))

    def test_swing_failure_too_short(self):
        orb = ORBLevels(high=105.0, low=95.0, open_price=100.0,
                        close_price=102.0, range_size=10.0, formed_at=None)
        df = make_df([100, 101])
        self.assertIsNone(Indicators.swing_failure(df, orb))


class TestMomentum(unittest.TestCase):
    def test_momentum_bullish(self):
        # Each candle must have body >= 60% of range
        # body=2, range=2.5 → ratio=0.80 ✓
        opens  = [100, 101, 102]
        closes = [102, 103, 104]
        highs  = [102.5, 103.5, 104.5]
        lows   = [99.5,  100.5, 101.5]
        df = make_df(closes, highs=highs, lows=lows, opens=opens)
        result = Indicators.momentum(df, candles=3, body_pct=0.60)
        self.assertEqual(result, "bullish")

    def test_momentum_none_mixed(self):
        opens  = [100, 103, 100]
        closes = [103, 100, 103]
        highs  = [104, 104, 104]
        lows   = [99,  99,  99]
        df = make_df(closes, highs=highs, lows=lows, opens=opens)
        result = Indicators.momentum(df, candles=3, body_pct=0.60)
        self.assertIsNone(result)


class TestStructureBreak(unittest.TestCase):
    def test_structure_break_bullish(self):
        # structure_break needs at least lookback+2 = 7 rows (default lookback=5)
        # window = df.iloc[-6:-1], last = df.iloc[-1]
        # last close must exceed max high of prior 5 rows
        closes = [100, 101, 102, 103, 104, 103, 110]  # 7 rows; last=110 > max_high(105)
        highs  = [101, 102, 103, 104, 105, 104, 111]
        lows   = [99,  100, 101, 102, 103, 102, 109]
        df = make_df(closes, highs=highs, lows=lows)
        result = Indicators.structure_break(df)
        self.assertEqual(result, "bullish")

    def test_structure_break_bearish(self):
        closes = [100, 99,  98,  97,  96,  97,  89]   # 7 rows; last=89 < min_low(95)
        highs  = [101, 100, 99,  98,  97,  98,  90]
        lows   = [99,  98,  97,  96,  95,  96,  88]
        df = make_df(closes, highs=highs, lows=lows)
        result = Indicators.structure_break(df)
        self.assertEqual(result, "bearish")

    def test_structure_break_none(self):
        closes = [100]*8  # flat, no break
        df = make_df(closes)
        result = Indicators.structure_break(df)
        self.assertIsNone(result)

    def test_structure_break_too_short(self):
        closes = [100, 101, 102, 103, 104, 108]  # only 6 rows → fails lookback+2 check
        df = make_df(closes)
        result = Indicators.structure_break(df)
        self.assertIsNone(result)


class TestBoxConfirm(unittest.TestCase):
    def test_box_confirm_long(self):
        opens  = [100, 101]
        closes = [101, 103]
        highs  = [102, 104]
        lows   = [99,  100]
        df = make_df(closes, highs=highs, lows=lows, opens=opens)
        self.assertTrue(Indicators.box_confirm(df, "long"))

    def test_box_confirm_short(self):
        opens  = [102, 100]
        closes = [101, 98]
        highs  = [103, 101]
        lows   = [100, 97]
        df = make_df(closes, highs=highs, lows=lows, opens=opens)
        self.assertTrue(Indicators.box_confirm(df, "short"))

    def test_box_confirm_too_short(self):
        df = make_df([100])
        self.assertFalse(Indicators.box_confirm(df, "long"))


class TestVWAPRetest(unittest.TestCase):
    def test_vwap_retest_near(self):
        df = make_df([100]*5, highs=[101]*5, lows=[99]*5)
        vwap_s = pd.Series([100.0]*5)
        # price at 100.0, VWAP at 100.0 → definitely near
        result = Indicators.vwap_retest(df, vwap_s, "long", tol=0.003)
        self.assertTrue(result)

    def test_vwap_retest_far(self):
        df = make_df([110]*5, highs=[111]*5, lows=[109]*5)
        vwap_s = pd.Series([100.0]*5)
        result = Indicators.vwap_retest(df, vwap_s, "long", tol=0.003)
        self.assertFalse(result)

    def test_vwap_retest_too_short(self):
        df = make_df([100, 101])
        vwap_s = pd.Series([100.0, 100.0])
        result = Indicators.vwap_retest(df, vwap_s, "long")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
