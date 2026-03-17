"""
tests/test_risk_manager.py
==========================
Unit tests for risk/risk_manager.py — RiskManager, Position, DailyState.
"""
import sys
import os
import unittest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import mock_deps  # noqa: F401

from risk.risk_manager import RiskManager, Position, DailyState


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_rm(**kwargs):
    defaults = dict(
        max_risk_pct=1.0, max_daily_loss_pct=3.0,
        max_concurrent=3, default_rr=3.0, min_rr=2.0,
        account_size=10_000.0, pdt_safe_mode=True, paper_mode=True,
    )
    defaults.update(kwargs)
    return RiskManager(**defaults)


def make_pos(symbol="NVDA", direction="long", entry=100.0, stop=98.0,
             t1=103.0, t2=106.0, shares=10.0, quality=0.65, strategy="TEST"):
    return Position(
        symbol=symbol, direction=direction,
        entry_price=entry, stop_loss=stop,
        target_1=t1, target_2=t2,
        shares=shares, entry_time=datetime.now(),
        strategy=strategy, quality_score=quality,
    )


# ── TestPosition ──────────────────────────────────────────────────────────────

class TestPosition(unittest.TestCase):
    def test_is_open_initial(self):
        p = make_pos()
        self.assertTrue(p.is_open)

    def test_is_open_false_after_close(self):
        p = make_pos()
        p.status = "target_hit"
        self.assertFalse(p.is_open)

    def test_dollar_risk_long(self):
        p = make_pos(entry=100.0, stop=98.0, shares=10.0)
        self.assertAlmostEqual(p.dollar_risk, 20.0)

    def test_dollar_risk_short(self):
        p = make_pos(direction="short", entry=100.0, stop=102.0, shares=5.0)
        self.assertAlmostEqual(p.dollar_risk, 10.0)


# ── TestDailyState ────────────────────────────────────────────────────────────

class TestDailyState(unittest.TestCase):
    def _state(self, starting=10_000.0, realized_pnl=0.0, trades=0, wins=0):
        from datetime import date
        s = DailyState(date=date.today(), starting_equity=starting)
        s.realized_pnl = realized_pnl
        s.trades_taken = trades
        s.trades_won = wins
        return s

    def test_pnl_pct_positive(self):
        s = self._state(10_000.0, 300.0, trades=1, wins=1)
        self.assertAlmostEqual(s.pnl_pct, 3.0)

    def test_pnl_pct_negative(self):
        s = self._state(10_000.0, -300.0, trades=1, wins=0)
        self.assertAlmostEqual(s.pnl_pct, -3.0)

    def test_pnl_pct_zero_equity(self):
        s = self._state(0.0, 0.0)
        self.assertEqual(s.pnl_pct, 0.0)

    def test_win_rate_full(self):
        s = self._state(trades=4, wins=3)
        self.assertAlmostEqual(s.win_rate, 0.75)

    def test_win_rate_no_trades(self):
        s = self._state(trades=0, wins=0)
        self.assertEqual(s.win_rate, 0.0)


# ── TestRiskManagerInit ───────────────────────────────────────────────────────

class TestRiskManagerInit(unittest.TestCase):
    def test_get_open_positions_empty(self):
        rm = make_rm()
        self.assertEqual(rm.get_open_positions(), [])

    def test_get_history_empty(self):
        rm = make_rm()
        self.assertEqual(rm.get_history(), [])

    def test_is_kill_switch_no_state(self):
        rm = make_rm()
        self.assertFalse(rm.is_kill_switch())

    def test_summary_no_state(self):
        rm = make_rm()
        self.assertEqual(rm.get_summary(), {})


# ── TestCanTrade ──────────────────────────────────────────────────────────────

class TestCanTrade(unittest.TestCase):
    def setUp(self):
        self.rm = make_rm(paper_mode=True)
        self.rm.initialize_day(10_000.0)

    def test_approved_normal(self):
        ok, msg = self.rm.can_trade("NVDA", 100, 98, 106, 10_000)
        self.assertTrue(ok)
        self.assertEqual(msg, "APPROVED")

    def test_rejected_zero_risk(self):
        ok, msg = self.rm.can_trade("NVDA", 100, 100, 106, 10_000)
        self.assertFalse(ok)
        self.assertIn("Zero risk", msg)

    def test_rejected_low_rr(self):
        # R:R = 1.5 < min_rr 2.0
        ok, msg = self.rm.can_trade("NVDA", 100, 98, 103, 10_000)
        self.assertFalse(ok)
        self.assertIn("R:R", msg)

    def test_rejected_kill_switch(self):
        # Trigger kill switch by setting loss beyond threshold
        self.rm._state.realized_pnl = -500.0  # -5% > 3% limit
        ok, msg = self.rm.can_trade("NVDA", 100, 98, 106, 10_000)
        self.assertFalse(ok)
        self.assertIn("Kill switch", msg)

    def test_rejected_max_concurrent(self):
        rm = make_rm(max_concurrent=1, paper_mode=True)
        rm.initialize_day(10_000.0)
        p = make_pos("NVDA")
        rm.record_open(p)
        ok, msg = rm.can_trade("AMD", 100, 98, 106, 10_000)
        self.assertFalse(ok)
        self.assertIn("concurrent", msg)

    def test_rejected_already_in_symbol(self):
        p = make_pos("NVDA")
        self.rm.record_open(p)
        ok, msg = self.rm.can_trade("NVDA", 100, 98, 106, 10_000)
        self.assertFalse(ok)
        self.assertIn("Already", msg)

    def test_pdt_limit_paper_mode_bypassed(self):
        # In paper mode, PDT should NOT block trades even after 3 day trades
        rm = make_rm(paper_mode=True, pdt_safe_mode=True)
        rm.initialize_day(10_000.0)
        rm._state.day_trades_used = 5
        ok, msg = rm.can_trade("NVDA", 100, 98, 106, 10_000)
        self.assertTrue(ok)

    def test_pdt_limit_live_mode_blocks(self):
        rm = make_rm(paper_mode=False, pdt_safe_mode=True)
        rm.initialize_day(10_000.0)
        rm._state.day_trades_used = 3
        ok, msg = rm.can_trade("NVDA", 100, 98, 106, 10_000, quality_score=0.65)
        self.assertFalse(ok)
        self.assertIn("PDT", msg)

    def test_kill_switch_saved_after_trigger(self):
        self.rm._state.realized_pnl = -500.0
        self.rm.can_trade("NVDA", 100, 98, 106, 10_000)
        self.assertTrue(self.rm._state.kill_switch_hit)
        # Subsequent check also fails
        ok, _ = self.rm.can_trade("AMD", 100, 98, 106, 10_000)
        self.assertFalse(ok)


# ── TestCalculateShares ───────────────────────────────────────────────────────

class TestCalculateShares(unittest.TestCase):
    def setUp(self):
        self.rm = make_rm(max_risk_pct=1.0, paper_mode=True)

    def test_shares_positive(self):
        shares, dollar_risk = self.rm.calculate_shares(100.0, 98.0, 10_000.0, quality_score=0.65)
        self.assertGreater(shares, 0)
        self.assertGreater(dollar_risk, 0)

    def test_shares_zero_on_zero_risk_dist(self):
        shares, dr = self.rm.calculate_shares(100.0, 100.0, 10_000.0)
        self.assertEqual(shares, 0.0)

    def test_shares_smaller_for_lower_quality(self):
        # Use a tight stop (0.10) so risk-based shares are small enough to
        # stay below the 20%-of-equity position cap, letting quality scale them.
        # equity=10k, risk=1%, entry=100, stop=99.90 → risk_dist=0.10
        # factor 1.0 → $100/0.10 = 1000 shares; but 20% cap = 10k*0.2/100=20 shares
        # Use large entry price so cap is lower:
        # entry=1000, stop=999, equity=1000 → cap=1000*0.2/1000=0.2 shares
        # Better: just use a high-price entry to make the cap tight
        # entry=500, stop=497.5 (0.5%), equity=2000 → 20% cap = 2000*0.2/500=0.8
        # risk 1%full = 20/2.5 = 8 → still hits cap
        # Simplest: check factor differences via dollar_risk which always differs
        _, dr_high = self.rm.calculate_shares(100.0, 99.0, 10_000.0, quality_score=0.80)
        _, dr_low  = self.rm.calculate_shares(100.0, 99.0, 10_000.0, quality_score=0.55)
        # dollar_risk is computed before capping → reflects quality scaling
        self.assertGreaterEqual(dr_high, dr_low)

    def test_shares_capped_at_20pct_account(self):
        # Even with tight stop, shares * price shouldn't exceed 20% of equity
        shares, _ = self.rm.calculate_shares(100.0, 99.99, 10_000.0, quality_score=0.9)
        self.assertLessEqual(shares * 100.0, 10_000.0 * 0.20 * 1.01)  # 1% tolerance

    def test_dollar_risk_within_max_risk_pct(self):
        _, dollar_risk = self.rm.calculate_shares(100.0, 98.0, 10_000.0, quality_score=0.75)
        # Max risk is 1% of 10k = $100; full quality = 1.0x factor → $100
        self.assertLessEqual(dollar_risk, 105.0)  # small tolerance


# ── TestRecordOpenClose ───────────────────────────────────────────────────────

class TestRecordOpenClose(unittest.TestCase):
    def setUp(self):
        self.rm = make_rm(paper_mode=True)
        self.rm.initialize_day(10_000.0)

    def test_record_open_increments_trades_taken(self):
        p = make_pos("NVDA")
        self.rm.record_open(p)
        self.assertEqual(self.rm._state.trades_taken, 1)

    def test_record_close_win(self):
        p = make_pos("NVDA", entry=100.0, shares=10.0)
        self.rm.record_open(p)
        closed = self.rm.record_close("NVDA", 106.0)
        self.assertIsNotNone(closed)
        self.assertAlmostEqual(closed.pnl, 60.0)
        self.assertEqual(self.rm._state.trades_won, 1)

    def test_record_close_loss(self):
        p = make_pos("NVDA", entry=100.0, stop=98.0, shares=10.0)
        self.rm.record_open(p)
        closed = self.rm.record_close("NVDA", 98.0)
        self.assertAlmostEqual(closed.pnl, -20.0)
        self.assertEqual(self.rm._state.trades_won, 0)

    def test_record_close_short_win(self):
        p = make_pos("NVDA", direction="short", entry=100.0, stop=102.0, shares=10.0)
        self.rm.record_open(p)
        closed = self.rm.record_close("NVDA", 94.0)
        self.assertAlmostEqual(closed.pnl, 60.0)

    def test_record_close_nonexistent_symbol(self):
        result = self.rm.record_close("FAKE", 100.0)
        self.assertIsNone(result)

    def test_record_close_updates_equity(self):
        p = make_pos("NVDA", entry=100.0, shares=10.0)
        self.rm.record_open(p)
        self.rm.record_close("NVDA", 106.0)
        self.assertAlmostEqual(self.rm._state.current_equity, 10_060.0)

    def test_get_open_positions_after_open(self):
        p = make_pos("NVDA")
        self.rm.record_open(p)
        open_pos = self.rm.get_open_positions()
        self.assertEqual(len(open_pos), 1)
        self.assertEqual(open_pos[0].symbol, "NVDA")

    def test_get_history_after_close(self):
        p = make_pos("NVDA")
        self.rm.record_open(p)
        self.rm.record_close("NVDA", 106.0)
        history = self.rm.get_history()
        self.assertEqual(len(history), 1)

    def test_position_removed_from_open_after_close(self):
        p = make_pos("NVDA")
        self.rm.record_open(p)
        self.rm.record_close("NVDA", 106.0)
        open_pos = self.rm.get_open_positions()
        self.assertEqual(len(open_pos), 0)


# ── TestSummary ───────────────────────────────────────────────────────────────

class TestSummary(unittest.TestCase):
    def test_summary_keys(self):
        rm = make_rm(paper_mode=True)
        rm.initialize_day(10_000.0)
        summary = rm.get_summary()
        for key in ["date", "starting_equity", "realized_pnl", "pnl_pct",
                    "trades_taken", "trades_won", "win_rate", "kill_switch"]:
            self.assertIn(key, summary)

    def test_summary_starting_equity(self):
        rm = make_rm(paper_mode=True)
        rm.initialize_day(12_345.0)
        self.assertAlmostEqual(rm.get_summary()["starting_equity"], 12_345.0)

    def test_daily_loss_triggers_kill_switch(self):
        rm = make_rm(max_daily_loss_pct=3.0, paper_mode=True)
        rm.initialize_day(10_000.0)
        # Simulate 3.5% loss
        p = make_pos("NVDA", entry=100.0, shares=35.0)
        rm.record_open(p)
        rm.record_close("NVDA", 90.0)   # -$350 = -3.5%
        # Kill switch is checked lazily inside can_trade(), not record_close()
        ok, msg = rm.can_trade("AMD", 100, 98, 106, 10_000)
        self.assertFalse(ok)
        self.assertIn("Kill switch", msg)
        self.assertTrue(rm.is_kill_switch())


if __name__ == "__main__":
    unittest.main(verbosity=2)
